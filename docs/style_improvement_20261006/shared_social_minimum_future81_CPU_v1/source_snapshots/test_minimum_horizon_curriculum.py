"""Future81 source/RNG/ABI contracts, with no claim of actual physics quality."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.training.minimum_horizon_curriculum import MinimumHorizonMixedCurriculum
from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig


def corpus(tmp_path, counts=(91, 169, 401)):
    path = Path(__file__).with_name("test_amass_root_curriculum.py")
    spec = importlib.util.spec_from_file_location("existing_horizon_CPU_fixtures", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.corpus(tmp_path, counts=counts)


def pair(tmp_path):
    config = PacerTrajectoryConfig()
    return tuple(cls(corpus(tmp_path), 4, procedural_config=config, seed=7)
                 for cls in (MixedTrajectoryCurriculum, MinimumHorizonMixedCurriculum))


def test_same_ID_prior_draw_order_and_procedural_reset_but_longer_AMASS_frames(tmp_path):
    baseline, candidate = pair(tmp_path)
    global_rng = torch.get_rng_state().clone()
    actors = torch.tensor([3, 0, 2, 1])
    old, new = baseline.sample_reference(actors), candidate.sample_reference(actors)
    real, proc = actors < 2, actors >= 2
    assert torch.equal(old[3], new[3])
    assert torch.equal(old[4][proc], new[4][proc])
    assert torch.equal(old[0].root_pos[proc], new[0].root_pos[proc])
    remaining = candidate.corpus.data["motion_num_frames"][new[3][real]] - 1 - new[4][real]
    assert bool((remaining >= 81).all())
    assert torch.equal(baseline.private_corpus.generator.get_state(), candidate.private_corpus.generator.get_state())
    assert torch.equal(baseline.corpus.generator.get_state(), candidate.corpus.generator.get_state())
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(candidate.corpus.ids, torch.arange(3))
    assert candidate.private_corpus.data is candidate.corpus.data


def test_same_legal_binding_produces_identical_native_commands_and_procedural_plans(tmp_path):
    baseline, candidate = pair(tmp_path)
    actors = torch.arange(4)
    reference, _, _, mid, frame = candidate.sample_reference(actors)
    origins = torch.tensor([[10., 10.], [20., 10.], [30., 10.], [40., 10.]])
    state = SimpleNamespace(root_pos=torch.cat((origins, torch.ones(4, 1)), -1),
        root_vel=torch.zeros(4, 3), root_rot=torch.tensor([[0., 0., 0., 1.]]).repeat(4, 1))
    episodes, ages = torch.ones(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)
    for mixed in (baseline, candidate):
        mixed.bind_reset(actors, mid, frame, reference.root_pos[:, :2], origins, episodes)
    plans = [mixed.build_plan(state, ages, episode_ids=episodes) for mixed in (baseline, candidate)]
    assert torch.equal(plans[0].xy, plans[1].xy)
    seconds = torch.tensor([[.5, 1., 2.7, 5.]]).repeat(4, 1)
    commands = [mixed.interpolate_reference(seconds, plans[i].native_xy)
                for i, mixed in enumerate((baseline, candidate))]
    assert torch.equal(*commands)
    targets = [mixed.future_targets(ages, ages, plans[i].native_xy, episode_ids=episodes)
               for i, mixed in enumerate((baseline, candidate))]
    assert torch.equal(*targets)
    assert baseline.config["real_reset_future_frames"] == 31
    assert candidate.config["real_reset_future_frames"] == candidate.config["amass_config"]["future_frames"] == 81
    assert baseline.SCHEMA != candidate.SCHEMA


def test_infeasible_short_clip_rejects_whole_corpus_without_RNG_or_ID_changes(tmp_path):
    c = corpus(tmp_path, counts=(90, 169))
    rng = c.generator.get_state().clone()
    ids = c.ids.clone()
    with pytest.raises(ValueError, match="future81"):
        MinimumHorizonMixedCurriculum(c, 4, procedural_config=PacerTrajectoryConfig())
    assert torch.equal(c.ids, ids) and torch.equal(c.generator.get_state(), rng)


def test_remaining80_AMASS_rejected_before_any_partial_binding(tmp_path):
    _, candidate = pair(tmp_path)
    actors = torch.tensor([0, 2])
    mid, frame = torch.tensor([0, 0]), torch.tensor([10, 89])
    root = candidate.corpus.state_at(candidate.corpus.data["length_starts"][mid] + frame).root_pos[:, :2]
    with pytest.raises(ValueError, match="future81"):
        candidate.bind_reset(actors, mid, frame, root, root, torch.ones(2, dtype=torch.long))
    assert not bool(candidate.amass.initialized.any()) and not bool(candidate.proc_initialized.any())
    assert candidate.statistics["amass_resets"] == candidate.statistics["procedural_resets"] == 0


def test_remaining81_AMASS_and_remaining1_procedural_are_legal(tmp_path):
    _, candidate = pair(tmp_path)
    actors = torch.tensor([0, 2])
    mid, frame = torch.tensor([0, 0]), torch.tensor([9, 89])
    root = candidate.corpus.state_at(candidate.corpus.data["length_starts"][mid] + frame).root_pos[:, :2]
    candidate.bind_reset(actors, mid, frame, root, root, torch.ones(2, dtype=torch.long))
    assert int(candidate.amass.remaining_steps[0]) == 81
    assert bool(candidate.proc_initialized[0])


def test_invalid_source_ID_fails_as_contract_error_not_indexing_error(tmp_path):
    _, candidate = pair(tmp_path)
    with pytest.raises(ValueError, match="source ID"):
        candidate.bind_reset(torch.tensor([0]), torch.tensor([999]), torch.tensor([9]),
            torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, dtype=torch.long))


def test_candidate_state_roundtrip_and_baseline_resume_rejection(tmp_path):
    baseline, candidate = pair(tmp_path)
    actors = torch.arange(4)
    reference, _, _, mid, frame = candidate.sample_reference(actors)
    candidate.bind_reset(actors, mid, frame, reference.root_pos[:, :2], reference.root_pos[:, :2], torch.ones(4, dtype=torch.long))
    state = deepcopy(candidate.state_dict())
    _, recovered = pair(tmp_path)
    recovered.load_state_dict(state)
    assert torch.equal(recovered.amass.remaining_steps, candidate.amass.remaining_steps)
    assert recovered.describe()["config"] == candidate.describe()["config"]
    with pytest.raises(ValueError, match="schema"):
        baseline.load_state_dict(state)
