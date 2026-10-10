"""CPU source/RNG contracts; these synthetic fixtures are not physics evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig, PacerTrajectoryCurriculum


def corpus(tmp_path, seed=19):
    c = AMASSCorpus.__new__(AMASSCorpus)
    c.split, c.device, c.ids = 'train', torch.device('cpu'), torch.arange(3)
    c.diffusion_ids = c.ids
    c.generator = torch.Generator().manual_seed(seed)
    c._convert_dof_velocity = False
    c.output_dof_velocity_frame = 'child_joint'
    c.manifest_path = tmp_path/'synthetic_mixed_manifest.json'
    c.manifest_path.write_text('{"synthetic_CPU_contract_bank":true}')
    counts = torch.tensor([73, 169, 401])
    c.entry = {'sha256': 'synthetic_not_a_real_asset_hash',
               'clips': [{'motion_id': i, 'frames': int(n)} for i, n in enumerate(counts)]}
    starts = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]))
    c.data = {'motion_num_frames': counts, 'motion_dt': torch.full((3,), 1/30), 'length_starts': starts}
    roots = []
    for motion, count in enumerate(counts):
        frame = torch.arange(int(count)).float()
        roots.append(torch.stack((motion*5+.007*frame+.00001*frame.square(), .003*frame, torch.ones_like(frame)), -1))
    root = torch.cat(roots)
    total = len(root)
    q = torch.zeros(total, 24, 4)
    q[..., 3] = 1
    # Every frame has a unique DOF marker; tests can prove ordered next/history.
    dps = torch.arange(total).float()[:, None].expand(total, 69).clone()/1000
    c.data.update(gts=root[:, None].expand(total, 24, 3).clone(), grs=q,
                  gvs=torch.zeros(total, 24, 3), gavs=torch.zeros(total, 24, 3), dps=dps, dvs=dps+1)
    return c


def make(tmp_path, seed=7, corpus_seed=19):
    config = PacerTrajectoryConfig(speed_max_mps=2, scalar_accel_max_mps2=1.5, vector_accel_max_mps2=1.5)
    return MixedTrajectoryCurriculum(corpus(tmp_path, corpus_seed), 4, device='cpu',
        bounds_xy=(-100, -100, 100, 100), procedural_config=config, seed=seed)


def setup(mixed):
    ids = torch.tensor([3, 0, 2, 1])
    reference, history, next_dps, mid, frame = mixed.sample_reference(ids)
    origins = torch.tensor([[10., 10.], [20., 10.], [30., 10.], [40., 10.]])
    mixed.bind_reset(ids, mid, frame, reference.root_pos[:, :2], origins[ids], torch.ones(4, dtype=torch.long))
    state = SimpleNamespace(root_pos=torch.cat((origins, torch.ones(4, 1)), -1),
                            root_vel=torch.zeros(4, 3), root_rot=torch.tensor([[0., 0., 0., 1.]]).repeat(4, 1))
    ages = torch.zeros(4, dtype=torch.long)
    episodes = torch.ones(4, dtype=torch.long)
    plan = mixed.build_plan(state, ages, episode_ids=episodes)
    return state, origins, ages, episodes, plan


def advance(mixed, ages, steps=1, terminated=None, cap=None):
    terminated = torch.zeros(4, dtype=torch.bool) if terminated is None else terminated
    cap = torch.zeros(4, dtype=torch.bool) if cap is None else cap
    for _ in range(steps):
        ages += 1
        done = mixed.completed_step(ages, terminated, cap)
    return done


def assert_reference_equal(left, right):
    for i in (0, 1):
        for name, value in vars(left[i]).items():
            assert torch.equal(value, getattr(right[i], name)), name
    for i in (2, 3, 4):
        assert torch.equal(left[i], right[i])


def test_shares_bank_not_rng_and_preserves_requested_order_and_legal_source_windows(tmp_path):
    mixed = make(tmp_path)
    assert mixed.private_corpus.data is mixed.corpus.data
    assert mixed.private_corpus.generator is not mixed.corpus.generator
    assert mixed.generator is not mixed.private_corpus.generator
    original = corpus(tmp_path)
    real = corpus(tmp_path, seed=8)
    global_rng = torch.get_rng_state().clone()
    ids = torch.tensor([3, 0, 2, 1])
    actual = mixed.sample_reference(ids)
    real_expected = real.sample_reference(2, future_frames=31)
    proc_expected = original.sample_reference(2, future_frames=1)
    for slot, expected in ((torch.tensor([1, 3]), real_expected), (torch.tensor([0, 2]), proc_expected)):
        assert torch.equal(actual[3][slot], expected[3])
        assert torch.equal(actual[4][slot], expected[4])
        assert torch.equal(actual[0].root_pos[slot], expected[0].root_pos)
        assert torch.equal(actual[1].root_pos.reshape(4, 10, 3)[slot], expected[1].root_pos.reshape(2, 10, 3))
        assert torch.equal(actual[2][slot], expected[2])
    assert torch.equal(mixed.corpus.generator.get_state(), original.generator.get_state())
    assert torch.equal(mixed.private_corpus.generator.get_state(), real.generator.get_state())
    assert torch.equal(torch.get_rng_state(), global_rng)
    only_real = torch.tensor([1, 0])
    rng = mixed.corpus.generator.get_state().clone()
    mixed.sample_reference(only_real)
    assert torch.equal(mixed.corpus.generator.get_state(), rng)


def test_source_native30hz_and_original_procedural_formula_for_non_grid_queries(tmp_path):
    mixed = make(tmp_path)
    state, origins, ages, episodes, plan = setup(mixed)
    legacy = PacerTrajectoryCurriculum(mixed.procedural.config, seed=7).build_plan(state, actor_ids=torch.tensor([2, 3]))
    assert torch.equal(plan.xy[torch.tensor([2, 3])], legacy.xy)
    assert torch.equal(plan.native_xy[:2, ::3], plan.xy[:2])
    mid, frame = mixed.amass.motion_ids.cpu(), mixed.amass.initial_frames.cpu()
    absolute = mixed.corpus.data['length_starts'][mid, None]+(frame[:, None]+torch.arange(151)).minimum(mixed.corpus.data['motion_num_frames'][mid, None]-1)
    expected = mixed.corpus.data['gts'][absolute, 0, :2]+mixed.amass.offset_xy[:, None]
    raw_frame = frame[:, None]+torch.arange(151)
    beyond = (raw_frame-mixed.corpus.data['motion_num_frames'][mid, None]+1).clamp_min(0)
    expected += beyond[..., None]/30*mixed.amass.end_velocity_xy[:, None]
    assert torch.allclose(plan.native_xy[:2], expected, atol=1e-6, rtol=0)
    advance(mixed, ages, 4)
    local_age = torch.full((4,), 4)
    task = mixed.future_targets(ages, local_age, plan.native_xy, episodes)
    assert torch.equal(task[:2], mixed.amass.future_targets(ages[:2], episode_ids=episodes[:2]))
    seconds = torch.arange(1, 11)[None]*.5+4/30
    f = (seconds*10).clamp(0, 50)
    lo = f.floor().long().clamp_max(49)
    rows = torch.arange(2)[:, None]
    expected_proc = torch.lerp(plan.xy[2:][rows, lo], plan.xy[2:][rows, lo+1], (f-lo)[..., None])
    assert torch.equal(task[2:], expected_proc)
    query = torch.tensor([[1/30, 2/30], [3/30, 4/30], [.123456, 5.1], [.078923, 4.998]])
    out = mixed.interpolate_reference(query, plan.native_xy)
    assert torch.equal(out[:2], plan.native_xy[:2][rows, torch.tensor([[1, 2], [3, 4]])])
    f = (query[2:]*10).clamp(0, 50)
    lo = f.floor().long().clamp_max(49)
    assert torch.equal(out[2:], torch.lerp(plan.xy[2:][rows, lo], plan.xy[2:][rows, lo+1], (f-lo)[..., None]))
    shifted_physical = deepcopy(state)
    shifted_physical.root_pos += 20
    previous = mixed.build_plan(shifted_physical, ages, actor_ids=torch.tensor([1, 0]), episode_ids=episodes)
    assert not torch.equal(previous.native_xy[:, 0], shifted_physical.root_pos[torch.tensor([1, 0]), :2])
    assert torch.equal(mixed.amass.offset_xy, origins[:2]-mixed.corpus.data['gts'][mixed.corpus.data['length_starts'][mid]+frame, 0, :2])


def test_real_clip_end_only_and_exact_group_control_exposure(tmp_path):
    mixed = make(tmp_path)
    ids = torch.arange(4)
    motion = torch.tensor([0, 0, 0, 0])
    frame = torch.tensor([41, 41, 71, 71])  # 31 steps real, one step proc reference only.
    source = mixed.corpus.data['gts'][frame, 0, :2]
    origins = torch.tensor([[10., 10.], [20., 10.], [30., 10.], [40., 10.]])
    mixed.bind_reset(ids, motion, frame, source, origins, torch.ones(4, dtype=torch.long))
    ages = torch.zeros(4, dtype=torch.long)
    assert not advance(mixed, ages, 30).any()
    done = advance(mixed, ages, terminated=torch.tensor([False, True, False, True]), cap=torch.ones(4, dtype=torch.bool))
    assert done.tolist() == [True, True, False, False]
    assert mixed.statistics['amass_actor_transitions'] == mixed.statistics['procedural_actor_transitions'] == 62
    assert mixed.statistics['control_steps'] == 31
    assert mixed.statistics['procedural_fall_terminations'] == mixed.statistics['procedural_cap_truncations'] == 1
    with pytest.raises(ValueError):
        mixed.completed_step(ages, torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))


def test_checkpoint_all_three_rng_streams_next_plan_and_actual_runner_buffers(tmp_path):
    original = make(tmp_path)
    state, origins, ages, episodes, plan = setup(original)
    advance(original, ages, 3)
    saved = deepcopy(original.state_dict())
    original_rng = original.corpus.generator.get_state().clone()
    restored = make(tmp_path, corpus_seed=987)
    restored.load_state_dict(saved)
    # Original corpus RNG is separately runner-owned; curriculum restore cannot consume it.
    untouched = restored.corpus.generator.get_state().clone()
    restored.load_state_dict(saved)
    assert torch.equal(restored.corpus.generator.get_state(), untouched)
    restored.corpus.generator.set_state(original_rng)
    flags = dict(replan_steps=15, origins_xy=origins)
    restored.validate_runner_state(ages, torch.full((4,), 3), plan.native_xy,
        torch.cat((original.amass.motion_ids, original.proc_reference_ids)), episodes, **flags)
    bad = plan.native_xy.clone()
    bad[3, 8, 1] += .01
    with pytest.raises(ValueError, match='canonical51'):
        restored.validate_runner_state(ages, torch.full((4,), 3), bad,
            torch.cat((original.amass.motion_ids, original.proc_reference_ids)), episodes, **flags)
    assert_reference_equal(original.sample_reference(torch.tensor([3, 1, 0, 2])), restored.sample_reference(torch.tensor([3, 1, 0, 2])))
    a = original.build_plan(state, ages, actor_ids=torch.tensor([2, 1, 3, 0]), episode_ids=episodes)
    b = restored.build_plan(state, ages, actor_ids=torch.tensor([2, 1, 3, 0]), episode_ids=episodes)
    assert torch.equal(a.native_xy, b.native_xy) and torch.equal(a.xy, b.xy)



def test_partial_reset_skips_only_declared_stale_plan_and_requires_replan(tmp_path):
    mixed = make(tmp_path)
    state, origins, ages, episodes, plan = setup(mixed)
    advance(mixed, ages, 3)
    ids = torch.tensor([3, 0])
    ref, _, _, mid, frame = mixed.sample_reference(ids)
    episodes[ids] += 1
    mixed.bind_reset(ids, mid, frame, ref.root_pos[:, :2], origins[ids], episodes[ids])
    ages[ids] = 0
    plan_age = torch.tensor([15, 3, 3, 15])
    plan.native_xy[ids] += 100  # Valid stale buffer remains until forced replan.
    refs = torch.cat((mixed.amass.motion_ids, mixed.proc_reference_ids))
    mixed.validate_runner_state(ages, plan_age, plan.native_xy, refs, episodes, replan_steps=15, origins_xy=origins)
    with pytest.raises(ValueError, match='must replan'):
        mixed.validate_runner_state(ages, torch.tensor([15, 3, 3, 0]), plan.native_xy, refs, episodes, replan_steps=15, origins_xy=origins)
    refreshed = mixed.build_plan(state, ages, actor_ids=ids, episode_ids=episodes)
    plan.native_xy[ids] = refreshed.native_xy
    plan_age[ids] = 0
    advance(mixed, ages)
    mixed.validate_runner_state(ages, plan_age+1, plan.native_xy, refs, episodes, replan_steps=15, origins_xy=origins)


@pytest.mark.parametrize('actors', [[0., 1.], [True], [-1], [4], [0, 0], []])
def test_invalid_global_ids_do_not_consume_any_rng(tmp_path, actors):
    mixed = make(tmp_path)
    states = [g.get_state().clone() for g in (mixed.corpus.generator, mixed.private_corpus.generator, mixed.generator)]
    with pytest.raises(ValueError):
        mixed.sample_reference(actors)
    assert all(torch.equal(s, g.get_state()) for s, g in zip(states, (mixed.corpus.generator, mixed.private_corpus.generator, mixed.generator), strict=True))


@pytest.mark.parametrize('field', ['mapping', 'rng', 'source', 'counters', 'clock', 'missing', 'dtype'])
def test_strict_restore_rejects_corruption_without_mutating_rng_or_actor_state(tmp_path, field):
    mixed = make(tmp_path)
    setup(mixed)
    saved = deepcopy(mixed.state_dict())
    if field == 'mapping':
        saved['config']['amass_actor_ids'] = [0, 2]
    elif field == 'rng':
        saved['generator_state'] = torch.zeros(4, dtype=torch.uint8)
    elif field == 'source':
        saved['procedural_actors']['source_xy'][0, 0] += .1
    elif field == 'counters':
        saved['statistics']['procedural_actor_transitions'] += 1
    elif field == 'clock':
        saved['procedural_actors']['last_plan_age'][0] = 1
    elif field == 'missing':
        del saved['procedural_actors']['reference_ids']
    elif field == 'dtype':
        saved['procedural_actors']['reference_ids'] = saved['procedural_actors']['reference_ids'].float()
    rng = [g.get_state().clone() for g in (mixed.corpus.generator, mixed.private_corpus.generator, mixed.generator)]
    actors = mixed.proc_reference_ids.clone()
    with pytest.raises(ValueError):
        mixed.load_state_dict(saved)
    assert torch.equal(actors, mixed.proc_reference_ids)
    assert all(torch.equal(s, g.get_state()) for s, g in zip(rng, (mixed.corpus.generator, mixed.private_corpus.generator, mixed.generator), strict=True))


@pytest.mark.parametrize('num_envs', [True, 1, 3, 4., 0])
def test_even_fixed_assignment_is_required(tmp_path, num_envs):
    with pytest.raises(ValueError):
        MixedTrajectoryCurriculum(corpus(tmp_path), num_envs)


def test_non_grid_real_time_and_float_runner_ids_rejected(tmp_path):
    mixed = make(tmp_path)
    _, origins, ages, episodes, plan = setup(mixed)
    with pytest.raises(ValueError, match='source grid'):
        mixed.interpolate_reference(torch.full((4, 1), .123), plan.native_xy)
    with pytest.raises(ValueError, match='reference IDs'):
        mixed.validate_runner_state(ages, torch.full((4,), 15), plan.native_xy,
            torch.cat((mixed.amass.motion_ids, mixed.proc_reference_ids)).float(), episodes,
            replan_steps=15, origins_xy=origins)


def test_safe_torch_checkpoint_payload_replays_future_reference_and_preserves_global_rng(tmp_path):
    import io

    original = make(tmp_path)
    setup(original)
    payload = {'curriculum': original.state_dict(), 'corpus_rng': original.corpus.generator.get_state()}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = make(tmp_path)
    restored.load_state_dict(loaded['curriculum'])
    restored.corpus.generator.set_state(loaded['corpus_rng'])
    global_rng = torch.get_rng_state().clone()
    for ids in (torch.tensor([0]), torch.tensor([3, 2]), torch.tensor([1, 3, 0])):
        assert_reference_equal(original.sample_reference(ids), restored.sample_reference(ids))
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize('branch', ['amass_state', 'procedural_state'])
def test_child_rng_failure_is_atomic_and_requires_complete_cpu_state(tmp_path, branch):
    mixed = make(tmp_path)
    setup(mixed)
    saved = deepcopy(mixed.state_dict())
    saved[branch]['generator_state'] = torch.zeros(4, dtype=torch.uint8)
    before = mixed.state_dict()
    with pytest.raises(ValueError):
        mixed.load_state_dict(saved)
    after = mixed.state_dict()
    for name in ('generator_state',):
        assert torch.equal(before[name], after[name])
    for child in ('amass_state', 'procedural_state'):
        assert torch.equal(before[child]['generator_state'], after[child]['generator_state'])
    assert torch.equal(before['procedural_actors']['reference_ids'], after['procedural_actors']['reference_ids'])


def test_invalid_source_bind_and_wrong_group_clock_fail_without_partial_reset(tmp_path):
    mixed = make(tmp_path)
    ids = torch.tensor([3, 0])
    ref, _, _, mid, frame = mixed.sample_reference(ids)
    source = ref.root_pos[:, :2].clone()
    source[0] += .01
    with pytest.raises(ValueError, match='actual source frame'):
        mixed.bind_reset(ids, mid, frame, source, torch.ones(2, 2), torch.ones(2, dtype=torch.long))
    assert not mixed.amass.initialized.any() and not mixed.proc_initialized.any()
    _, _, ages, _, plan = setup(mixed)
    with pytest.raises(ValueError, match='exactly once'):
        mixed.completed_step(torch.tensor([1, 1, 2, 1]), torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    assert mixed.statistics['control_steps'] == 0 and mixed.amass.statistics['control_actor_steps'] == 0
    with pytest.raises(ValueError, match='bound resets'):
        mixed.future_targets(ages, torch.zeros(4, dtype=torch.long), plan.native_xy, torch.tensor([1, 1, 2, 1]))


def test_external_state_and_description_cannot_change_live_expected_source_config(tmp_path):
    mixed = make(tmp_path)
    setup(mixed)
    expected = deepcopy(mixed.config)
    expected_child = deepcopy(mixed.amass.config)
    saved = mixed.state_dict()
    saved['config']['amass_config']['motion_ids'][0] = 999
    saved['config']['amass_actor_ids'][0] = 3
    saved['amass_state']['config']['source_bank_sha256'] = 'tampered'
    assert mixed.config == expected and mixed.amass.config == expected_child
    with pytest.raises(ValueError, match='source/config'):
        mixed.load_state_dict(saved)
    described = mixed.describe()
    described['config']['amass_actor_ids'][0] = 3
    described['amass']['config']['motion_ids'][0] = 999
    assert mixed.config == expected and mixed.amass.config == expected_child
    assert mixed.plans_generated == 4


def test_missing_procedural_recipe_cannot_silently_select_other_training_distribution(tmp_path):
    with pytest.raises(ValueError, match='procedural_config'):
        MixedTrajectoryCurriculum(corpus(tmp_path), 4)
