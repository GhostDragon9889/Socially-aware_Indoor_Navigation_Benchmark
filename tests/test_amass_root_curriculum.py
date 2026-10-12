"""CPU source-clock contracts; synthetic banks are not PhysX evidence."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.training.amass_trajectories import AMASSRootTrajectoryCurriculum


ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT/'resources/runtime_sources/style_improvement_20261002/amass_root_curriculum_before'


def corpus(tmp_path, seed=19, counts=(91, 169)):
    """Real corpus methods over a small declared synthetic tensor bank."""
    c = AMASSCorpus.__new__(AMASSCorpus)
    c.split, c.device, c.ids = 'train', torch.device('cpu'), torch.arange(len(counts))
    c.diffusion_ids = c.ids
    c.generator = torch.Generator().manual_seed(seed)
    c._convert_dof_velocity = False
    c.output_dof_velocity_frame = 'child_joint'
    c.manifest_path = tmp_path/'synthetic_manifest.json'
    c.manifest_path.write_text('{"synthetic_CPU_contract_bank":true}')
    c.entry = {'sha256': 'synthetic_not_a_real_weight_or_motion_hash',
               'clips': [{'motion_id': i, 'frames': n} for i, n in enumerate(counts)]}
    c.data = {'motion_num_frames': torch.tensor(counts), 'motion_dt': torch.full((len(counts),), 1/30),
              'length_starts': torch.tensor([0]+list(torch.tensor(counts).cumsum(0)[:-1].tolist()))}
    parts = []
    for i, n in enumerate(counts):
        f = torch.arange(n).float()
        root = torch.stack((10*i+.02*f+.0002*f.square(), .01*f, torch.ones(n)), -1)
        parts.append(root[:, None].repeat(1, 24, 1))
    c.data['gts'] = torch.cat(parts)
    total = sum(counts)
    q = torch.zeros(total, 24, 4)
    q[..., 3] = 1
    c.data.update(grs=q, gvs=torch.zeros(total, 24, 3), gavs=torch.zeros(total, 24, 3),
                  dps=torch.zeros(total, 69), dvs=torch.zeros(total, 69))
    return c


def bind(curriculum, *, ids=(0, 1), motions=(0, 1), frames=(9, 15), episodes=(1, 1), origins=None):
    c = curriculum.corpus
    mid, phase = torch.tensor(motions), torch.tensor(frames)
    source = c.data['gts'][c.data['length_starts'][mid]+phase, 0, :2]
    origins = torch.tensor([[20., 20.], [24., 20.]])[:len(ids)] if origins is None else origins
    curriculum.bind_reset(torch.tensor(ids), mid, phase, source, origins, torch.tensor(episodes))
    return origins


def advance(curriculum, ages, steps=1):
    for _ in range(steps):
        ages += 1
        done = curriculum.completed_step(ages, torch.zeros(len(ages), dtype=torch.bool), torch.zeros(len(ages), dtype=torch.bool))
    return done


def before_corpus():
    name = 'CrowdSim.humanoid.amass_root_before'
    spec = importlib.util.spec_from_file_location(name, BEFORE/'CrowdSim/humanoid/amass_corpus.py')
    old = importlib.util.module_from_spec(spec)
    sys.modules[name] = old
    spec.loader.exec_module(old)
    return old.AMASSCorpus


def test_default_sampler_matches_before_all_tensors_and_shared_rng(tmp_path):
    old = before_corpus()
    a, b, explicit = corpus(tmp_path), corpus(tmp_path), corpus(tmp_path)
    for batch in (2, 19, 1, 23):
        left = old.sample_reference(a, batch, 10)
        right, one = b.sample_reference(batch, 10), explicit.sample_reference(batch, 10, future_frames=1)
        for i in (0, 1):
            for key in vars(left[i]):
                assert torch.equal(getattr(left[i], key), getattr(right[i], key)), key
                assert torch.equal(getattr(left[i], key), getattr(one[i], key)), key
        for i in (2, 3, 4):
            assert torch.equal(left[i], right[i]) and torch.equal(left[i], one[i])
        assert torch.equal(a.generator.get_state(), b.generator.get_state())
        assert torch.equal(a.generator.get_state(), explicit.generator.get_state())


@pytest.mark.parametrize('future', [True, False, 0, -1, 1.5, None])
def test_invalid_future_window_rejected_without_consuming_rng(tmp_path, future):
    c = corpus(tmp_path)
    rng = c.generator.get_state().clone()
    with pytest.raises(ValueError):
        c.sample_reference(2, future_frames=future)
    assert torch.equal(rng, c.generator.get_state())


def test_true_reset_windows_keep_all_sources_and_history_newest_first(tmp_path):
    c = corpus(tmp_path)
    current, history, next_dof, mid, phase = c.sample_reference(1024, future_frames=31)
    assert set(mid.tolist()) == {0, 1}
    assert bool((phase >= 9).all())
    assert bool((c.data['motion_num_frames'][mid]-1-phase >= 31).all())
    absolute = c.data['length_starts'][mid]+phase
    expected = c.data['gts'][absolute[:, None]-torch.arange(10), 0]
    assert torch.equal(history.root_pos.reshape(1024, 10, 3), expected)
    assert torch.equal(current.root_pos, expected[:, 0]) and next_dof.shape == (1024, 69)


def test_actual_phase_native_targets_and_subsets_never_reanchor_or_draw_rng(tmp_path):
    curriculum = AMASSRootTrajectoryCurriculum(corpus(tmp_path), 2)
    origins = bind(curriculum)
    ages = torch.zeros(2, dtype=torch.long)
    first = curriculum.build_plan(ages, episode_ids=torch.ones(2, dtype=torch.long))
    assert first.xy.shape == (2, 51, 2) and first.native_xy.shape == (2, 151, 2)
    assert torch.equal(first.native_xy[:, ::3], first.xy)
    offset, rng = curriculum.offset_xy.clone(), curriculum.corpus.generator.get_state().clone()
    advance(curriculum, ages, 8)
    bind(curriculum, ids=(1,), motions=(1,), frames=(30,), episodes=(2,), origins=origins[1:])
    ages[1] = 0
    advance(curriculum, ages, 7)
    prediction = curriculum.build_plan(ages, episode_ids=torch.tensor([1, 2]))
    assert prediction.source_phase.tolist() == [24, 37]
    assert torch.equal(curriculum.offset_xy[0], offset[0])
    assert torch.equal(rng, curriculum.corpus.generator.get_state())
    c = curriculum.corpus
    expected = c.data['gts'][c.data['length_starts'][prediction.motion_ids]+prediction.source_phase+1, 0, :2]+curriculum.offset_xy
    assert torch.equal(prediction.native_xy[:, 1], expected)
    assert not torch.equal(prediction.native_xy[:, 0], origins)
    assert curriculum.statistics['held_future_knots'] > 0


def test_clip_end_real_truncation_counts_separately_and_never_steps_padded_tail(tmp_path):
    curriculum = AMASSRootTrajectoryCurriculum(corpus(tmp_path), 2)
    bind(curriculum, frames=(59, 137))  # Both have exactly31 real remaining steps.
    ages = torch.zeros(2, dtype=torch.long)
    plan = curriculum.build_plan(ages)
    assert plan.tail_hold[:, 11:].all() and not plan.tail_hold[:, :11].any()
    assert torch.equal(plan.native_xy[:, 31], plan.native_xy[:, -1])
    assert not advance(curriculum, ages, 30).any()
    ages += 1
    end = curriculum.completed_step(ages, torch.tensor([False, True]), torch.tensor([False, False]))
    assert end.tolist() == [True, True]
    assert curriculum.statistics['clip_end_truncations'] == 1
    assert curriculum.statistics['fall_terminations'] == curriculum.statistics['fall_at_clip_end'] == 1
    with pytest.raises(ValueError, match='reset a completed clip'):
        curriculum.completed_step(ages+1, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))


def test_resume_reproduces_next_plan_rng_and_rejects_shifted_source_or_plan(tmp_path):
    c = corpus(tmp_path)
    original = AMASSRootTrajectoryCurriculum(c, 2)
    origins = bind(original)
    ages = torch.zeros(2, dtype=torch.long)
    advance(original, ages, 15)
    plan = original.build_plan(ages)
    advance(original, ages, 3)
    saved = original.state_dict()
    restored = AMASSRootTrajectoryCurriculum(corpus(tmp_path, seed=989), 2)
    restored.load_state_dict(saved)
    flags = dict(replan_steps=15, origins_xy=origins)
    restored.validate_runner_state(ages, torch.tensor([3, 3]), plan.native_xy,
        original.motion_ids, original.episode_ids, **flags)
    with pytest.raises(ValueError, match='world reference IDs'):
        restored.validate_runner_state(ages, torch.tensor([3, 3]), plan.native_xy,
            original.motion_ids.float(), original.episode_ids, **flags)
    assert torch.equal(original.corpus.generator.get_state(), restored.corpus.generator.get_state())
    for left, right in zip(original.corpus.sample_reference(5)[3:], restored.corpus.sample_reference(5)[3:], strict=True):
        assert torch.equal(left, right)
    for i in (0, 1):
        bad_plan = plan.native_xy.clone()
        bad_plan[i] += .1
        with pytest.raises(ValueError, match='plan differs'):
            restored.validate_runner_state(ages, torch.tensor([3, 3]), bad_plan, original.motion_ids, original.episode_ids, **flags)
    bad = deepcopy(saved)
    bad['actors']['offset_xy'][0] += .2
    restored.load_state_dict(bad)
    with pytest.raises(ValueError, match='world origin'):
        restored.validate_runner_state(ages, torch.tensor([3, 3]), plan.native_xy, original.motion_ids, original.episode_ids, **flags)


def test_resume_newly_reset_actor_requires_next_step_replan_with_stale_plan(tmp_path):
    curriculum = AMASSRootTrajectoryCurriculum(corpus(tmp_path), 2)
    origins = bind(curriculum)
    ages = torch.zeros(2, dtype=torch.long)
    stale_plans = curriculum.build_plan(ages).native_xy
    advance(curriculum, ages, 3)
    bind(curriculum, ids=(0,), motions=(1,), frames=(20,), episodes=(2,), origins=origins[:1])
    ages[0] = 0
    stale_plans[0] += 100  # A reset legitimately retains the prior episode's unused buffer.
    flags = dict(replan_steps=15, origins_xy=origins)
    curriculum.validate_runner_state(ages, torch.tensor([15, 3]), stale_plans,
        curriculum.motion_ids, curriculum.episode_ids, **flags)
    with pytest.raises(ValueError, match='must replan before'):
        curriculum.validate_runner_state(ages, torch.tensor([0, 3]), stale_plans,
            curriculum.motion_ids, curriculum.episode_ids, **flags)


@pytest.mark.parametrize('field', ['initial_frames', 'remaining_steps', 'last_observed_age', 'last_plan_age', 'episode_ids', 'initialized'])
def test_incomplete_or_inconsistent_resume_is_rejected(tmp_path, field):
    curriculum = AMASSRootTrajectoryCurriculum(corpus(tmp_path), 2)
    bind(curriculum)
    state = curriculum.state_dict()
    del state['actors'][field]
    before = curriculum.motion_ids.clone()
    with pytest.raises(ValueError):
        curriculum.load_state_dict(state)
    assert torch.equal(before, curriculum.motion_ids)


def test_floor_source_and_rng_resume_guards_are_fail_closed(tmp_path):
    c = corpus(tmp_path)
    guarded = AMASSRootTrajectoryCurriculum(c, 2, bounds_xy=[0, 0, 25, 25])
    rng = c.generator.get_state().clone()
    with pytest.raises(ValueError, match='floor margin'):
        bind(guarded)
    assert torch.equal(rng, c.generator.get_state()) and not guarded.initialized.any()
    normal = AMASSRootTrajectoryCurriculum(c, 2)
    bind(normal)
    state = normal.state_dict()
    state['config']['source_bank_sha256'] = 'different'
    with pytest.raises(ValueError, match='source/config'):
        normal.load_state_dict(state)
    state = normal.state_dict()
    state['generator_state'] = torch.zeros(4, dtype=torch.uint8)
    with pytest.raises(ValueError, match='RNG'):
        normal.load_state_dict(state)


def runner_function(name, namespace):
    tree = ast.parse((ROOT/'scripts/train_paper_amp.py').read_text())
    fn = deepcopy(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name))
    fn.body = [ast.Global(names=n.names) if isinstance(n, ast.Nonlocal) else n for n in fn.body]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'real_runner_function', 'exec'), namespace)
    return namespace[name]


def test_real_runner_native_replan_and_next_step_target_source_phase(tmp_path):
    curriculum = AMASSRootTrajectoryCurriculum(corpus(tmp_path), 2)
    origins = bind(curriculum)
    ages = torch.zeros(2, dtype=torch.long)
    advance(curriculum, ages, 15)
    namespace = dict(torch=torch, args=SimpleNamespace(trajectory_source='amass_root', num_envs=2), curriculum=curriculum,
        world=SimpleNamespace(age=ages, episode_ids=torch.ones(2, dtype=torch.long), device=torch.device('cpu')),
        plans=torch.zeros(2, 151, 2), goals=torch.zeros(2, 2), plan_age=torch.full((2,), 15),
        plan_frequency_hz=30, replan_count=0, planner_forward_calls=0)
    replan = runner_function('replan', namespace)
    replan(torch.tensor([1, 0]))
    assert namespace['plan_age'].tolist() == [0, 0] and namespace['replan_count'] == 2
    interpolate = runner_function('interpolate', namespace)
    mid = curriculum.motion_ids
    for step in range(1, 16):
        expected = curriculum.corpus.data['gts'][curriculum.corpus.data['length_starts'][mid]+curriculum.initial_frames+15+step, 0, :2]+curriculum.offset_xy
        actual = interpolate(torch.full((2, 1), step/30))[:, 0]
        assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match='source grid'):
        interpolate(torch.full((2, 1), .012))
    assert not torch.equal(namespace['plans'][:, 0], origins)


def test_complete_training_physical_control_loop_ast_remains_exact_before():
    class TrainingBranch(ast.NodeTransformer):
        # The maintained runner now omits discriminator/value collection in
        # pure evaluation. Compare the actual training branch, without dropping
        # any physics, action, observation, reward or route statements.
        def visit_If(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.test) == 'not pure_evaluation':
                assert not node.orelse
                return node.body
            return node

        def visit_BoolOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.And) and ast.unparse(node.values[0]) == 'not pure_evaluation':
                assert len(node.values) == 2
                return node.values[1]
            return node

    def loop(path):
        tree = ast.parse(path.read_text())
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.target) == 'step']
        assert len(nodes) == 1
        return ast.dump(TrainingBranch().visit(nodes[0]), include_attributes=False)
    assert loop(ROOT/'scripts/train_paper_amp.py') == loop(BEFORE/'scripts/train_paper_amp.py')


def world_methods(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AMPWorld')
    methods = [deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('reset', 'step', 'planar')]
    namespace = dict(torch=torch, ResetState=SimpleNamespace, StateConversion=SimpleNamespace(COMMON='common'),
        build_amp_observation=lambda state, mapping: torch.zeros(len(state.root_pos), 195),
        action_to_pd=lambda actions, *unused, **kwargs: actions)
    exec(compile(ast.Module(body=methods, type_ignores=[]), 'actual_world_methods_CPU_contract', 'exec'), namespace)
    return namespace


def minimal_world(tmp_path, *, enabled, before=False):
    from types import MethodType
    methods = world_methods((BEFORE if before else ROOT)/'CrowdSim/humanoid/amp_world.py')
    c = corpus(tmp_path, counts=(91,))
    state = c.state_at(torch.tensor([9]))
    state.dof_forces = torch.zeros(1, 69)
    writes, steps = [], []

    def reset_envs(values, env_ids):
        writes.append(env_ids.clone())
        for name in ('root_pos', 'root_rot', 'root_vel', 'root_ang_vel', 'dof_pos', 'dof_vel'):
            getattr(state, name)[env_ids] = getattr(values, name)

    def step(target):
        steps.append(target.clone())  # Does not emulate simulation or claim SDK correctness.

    world = SimpleNamespace(align_spawn_heading_to_route=False, initial_reference_max_speed_m_s=None,
        initial_reference_arm_cone_deg=None, initialize_at_rest=False, corpus=c, device=torch.device('cpu'), num_envs=1, history_frames=10, episode_steps=300,
        origins=torch.tensor([[20., 20.]]), lower=torch.zeros(69), upper=torch.ones(69),
        mapping=SimpleNamespace(dof_names=[]), simulator=SimpleNamespace(reset_envs=reset_envs, step=step),
        age=torch.zeros(1, dtype=torch.long), episode_ids=torch.zeros(1, dtype=torch.long),
        reference_ids=torch.zeros(1, dtype=torch.long), amp_history=torch.zeros(1, 10, 195),
        amp_history_episode_ids=torch.zeros(1, 10, dtype=torch.long), planar_history=torch.zeros(1, 91, 5),
        history_available=torch.zeros(1, 91, dtype=torch.bool),
        ground_height_at=lambda xy: torch.zeros(xy.shape[:-1]), state=lambda: state)
    if enabled:
        world.amass_root_curriculum = AMASSRootTrajectoryCurriculum(c, 1, bounds_xy=(0., 0., 64., 64.))
    for name in ('reset', 'step', 'planar'):
        setattr(world, name, MethodType(methods[name], world))
    return world, writes, steps


def test_disabled_actual_world_reset_step_matches_before_rng_outputs_and_pose_write_count(tmp_path):
    old, old_writes, old_steps = minimal_world(tmp_path, enabled=False, before=True)
    current, writes, steps = minimal_world(tmp_path, enabled=False)
    for world in (old, current):
        world.reset(torch.tensor([0]))
    for _ in range(35):
        a, b = old.step(torch.zeros(1, 69)), current.step(torch.zeros(1, 69))
        for field in vars(a[0]):
            assert torch.equal(getattr(a[0], field), getattr(b[0], field))
        assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    assert len(writes) == len(old_writes) == 1 and len(steps) == len(old_steps) == 35
    assert torch.equal(old.corpus.generator.get_state(), current.corpus.generator.get_state())
    for name in ('age', 'amp_history', 'planar_history', 'history_available', 'episode_ids'):
        assert torch.equal(getattr(old, name), getattr(current, name))


def test_actual_world_opt_in_clip_end_truncates_without_runtime_pose_writes(tmp_path):
    world, writes, steps = minimal_world(tmp_path, enabled=True)
    world.reset(torch.tensor([0]))
    remaining = int(world.amass_root_curriculum.remaining_steps[0])
    assert remaining >= 31
    for index in range(remaining):
        _, fallen, truncated = world.step(torch.zeros(1, 69))
        assert not fallen.any() and bool(truncated[0]) == (index == remaining-1)
    assert len(writes) == 1 and len(steps) == remaining
    assert world.amass_root_curriculum.statistics['clip_end_truncations'] == 1
    assert world.amass_root_curriculum.statistics['cap_only_truncations'] == 0


def test_reference_end_bootstraps_final_state_without_crossing_next_episode():
    from CrowdSim.humanoid.training.contracts import generalized_advantages
    rewards, values = torch.tensor([[1.], [50.]]), torch.zeros(2, 1)
    next_values = torch.tensor([[7.], [11.]])  # Last actual physical state, never reset state100.
    terminated, truncated = torch.zeros(2, 1, dtype=torch.bool), torch.tensor([[True], [False]])
    advantages, _ = generalized_advantages(rewards, values, next_values, terminated, truncated, gamma=.9, gae_lambda=1.)
    assert float(advantages[0]) == pytest.approx(7.3)
    terminated[0] = True
    advantages, _ = generalized_advantages(rewards, values, next_values, terminated, truncated, gamma=.9, gae_lambda=1.)
    assert float(advantages[0]) == 1.


def test_linear_tail_preserves_all_real_frames_and_exact_ten_interval_velocity(tmp_path):
    c = corpus(tmp_path)
    curriculum = AMASSRootTrajectoryCurriculum(c, 2, tail_policy='linear_velocity')
    bind(curriculum, frames=(59, 137))
    ages = torch.zeros(2, dtype=torch.long)
    plan = curriculum.build_plan(ages)
    actual = c.data['gts'][c.data['length_starts'][curriculum.motion_ids, None]+curriculum.initial_frames[:, None]+torch.arange(32), 0, :2]+curriculum.offset_xy[:, None]
    assert torch.equal(plan.native_xy[:, :32], actual)
    last = c.data['length_starts'][curriculum.motion_ids]+c.data['motion_num_frames'][curriculum.motion_ids]-1
    velocity = (c.data['gts'][last, 0, :2]-c.data['gts'][last-10, 0, :2])/(10/30)
    assert torch.equal(curriculum.end_velocity_xy, velocity)
    torch.testing.assert_close(plan.native_xy[:, 32], actual[:, -1]+velocity/30, atol=1e-6, rtol=0)
    assert not plan.tail_hold.any() and plan.beyond_source[:, 11:].all()
    assert curriculum.statistics['held_future_knots'] == 0
    assert curriculum.statistics['extrapolated_future_knots'] == curriculum.statistics['beyond_source_future_knots'] > 0
    assert torch.equal(plan.xy, plan.native_xy[:, ::3])
    advance(curriculum, ages, 31)
    future = curriculum.future_targets(ages)
    torch.testing.assert_close(future[:, -1], actual[:, -1]+5*velocity, atol=1e-6, rtol=0)


def test_current_phase_task_targets_are_not_clamped_to_old_plan_window(tmp_path):
    curriculum = AMASSRootTrajectoryCurriculum(corpus(tmp_path), 2, tail_policy='linear_velocity')
    bind(curriculum)
    ages = torch.zeros(2, dtype=torch.long)
    old = curriculum.build_plan(ages)
    advance(curriculum, ages, 14)
    snapshot = curriculum.state_dict()
    namespace = dict(torch=torch, args=SimpleNamespace(trajectory_source='amass_root'), curriculum=curriculum,
        world=SimpleNamespace(age=ages, episode_ids=torch.ones(2, dtype=torch.long)))
    targets = runner_function('task_trajectory', namespace)()
    assert targets.shape == (2, 10, 2)
    assert torch.equal(targets, curriculum.future_targets(ages))
    assert not torch.equal(targets[:, -1], old.native_xy[:, -1])
    assert snapshot['statistics'] == curriculum.state_dict()['statistics']
    assert torch.equal(snapshot['generator_state'], curriculum.state_dict()['generator_state'])


def test_linear_tail_bounds_and_saved_velocity_are_fail_closed(tmp_path):
    c = corpus(tmp_path)
    # Actual source fits this floor, but the declared end+5s extension does not.
    hold = AMASSRootTrajectoryCurriculum(c, 2, bounds_xy=[0, 0, 35, 35])
    bind(hold)
    linear = AMASSRootTrajectoryCurriculum(c, 2, bounds_xy=[0, 0, 35, 35], tail_policy='linear_velocity')
    with pytest.raises(ValueError, match='Extrapolated five-second'):
        bind(linear)
    linear = AMASSRootTrajectoryCurriculum(c, 2, tail_policy='linear_velocity')
    bind(linear)
    state = linear.state_dict()
    state['actors']['end_velocity_xy'][0] += .1
    with pytest.raises(ValueError, match='endpoint velocity'):
        linear.load_state_dict(state)
    with pytest.raises(ValueError, match='source/config'):
        linear.load_state_dict(hold.state_dict())


@pytest.mark.parametrize('policy', ['unknown', None, True, 1])
def test_unknown_tail_policy_rejected_before_sampling(tmp_path, policy):
    c = corpus(tmp_path)
    rng = c.generator.get_state().clone()
    with pytest.raises(ValueError, match='tail_policy'):
        AMASSRootTrajectoryCurriculum(c, 2, tail_policy=policy)
    assert torch.equal(rng, c.generator.get_state())


def test_original_procedural_provenance_exact_before_and_amass_metadata_truthful():
    def nodes(path):
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        body = next(n for n in main.body if isinstance(n, ast.Try)).body
        start = next(i for i,n in enumerate(body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'curriculum is not None'
                     and 'provenance' in ast.unparse(n))
        end = next(i for i in range(start+1,len(body)) if isinstance(body[i], ast.If)
                   and ast.unparse(body[i].test) == 'args.resume')
        return deepcopy(body[start:end])

    def evaluate(path, source):
        namespace = dict(args=SimpleNamespace(trajectory_source=source), report={}, provenance={},
                         curriculum=SimpleNamespace(describe=lambda: {'config': {'tail_policy':'linear_velocity'}}))
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes(path), type_ignores=[])), 'actual_provenance', 'exec'), namespace)
        return namespace['provenance']

    assert evaluate(ROOT/'scripts/train_paper_amp.py', 'pacer_curriculum') == evaluate(BEFORE/'scripts/train_paper_amp.py', 'pacer_curriculum')
    new = evaluate(ROOT/'scripts/train_paper_amp.py', 'amass_root')
    assert new['navigation_guidance']['mode'] == 'amass_root_training_commands'
    assert new['task_tail'] == 'linear_velocity'
    assert new['amass_root_command_contract']['live_root_reanchoring'] is False
    assert 'fixed episode translation' in new['goal_generation']
