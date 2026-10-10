from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.training.episode_trajectories import EpisodeProceduralReference
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig, PacerTrajectoryCurriculum


def state():
    return SimpleNamespace(root_pos=torch.zeros(2, 3), root_vel=torch.tensor([[1., 0., 0.]]).repeat(2, 1),
        root_rot=torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1))


def config():
    return PacerTrajectoryConfig(speed_min_mps=1., speed_max_mps=1., heading_rate_max_rad_s=1e-8,
                                 sharp_turn_probability=0.)


def test_absolute_reference_progresses_and_live_root_cannot_erase_error():
    physical = state()
    reference = EpisodeProceduralReference(2, config=config())
    episodes = torch.ones(2, dtype=torch.long)
    reference.reset(physical, episodes)
    physical.root_pos[:, 0] = 100  # Owned reference is independent of later physical state.
    generator = reference.source.generator.get_state().clone()
    for age in (15, 30, 60, 300):
        target = reference.query(torch.full((2,), age), episodes, [0.])
        torch.testing.assert_close(target[:, 0, 0], torch.full((2,), age / 30), atol=1e-6, rtol=1e-6)
    assert torch.equal(generator, reference.source.generator.get_state())
    assert reference.source.plans_generated == 2
    endpoint = reference.query(torch.full((2,), 300), episodes, [5.])
    torch.testing.assert_close(endpoint[:, 0, 0], torch.full((2,), 15.))


def test_same_initial_smooth_five_second_path_and_global_rng_preserved():
    saved = torch.random.get_rng_state().clone()
    reference = EpisodeProceduralReference(2)
    original = PacerTrajectoryCurriculum(seed=7).build_plan(state())
    reference.reset(state(), torch.ones(2, dtype=torch.long))
    assert torch.equal(reference.paths[:, :51], original.xy)
    assert torch.equal(saved, torch.random.get_rng_state())


def test_partial_reset_does_not_regenerate_other_actor_and_resume_continues_exactly():
    reference = EpisodeProceduralReference(2)
    episodes = torch.ones(2, dtype=torch.long)
    reference.reset(state(), episodes)
    previous = reference.paths[0].clone()
    saved = reference.state_dict()
    repeated = EpisodeProceduralReference(2, seed=123)
    repeated.load_state_dict(saved)
    episodes[1] = 2
    physical = state()
    physical.root_pos[1, :2] = 10
    reference.reset(physical, episodes, [1])
    repeated.reset(physical, episodes, [1])
    assert torch.equal(reference.paths[0], previous)
    assert torch.equal(reference.paths, repeated.paths)
    assert reference.source.plans_generated == repeated.source.plans_generated == 3
    torch.testing.assert_close(reference.query(torch.zeros(2, dtype=torch.long), episodes, [0.])[1, 0], torch.full((2,), 10.))


@pytest.mark.parametrize("case", ["stale_episode", "outside_offset", "negative_age", "after_cap", "uninitialized"])
def test_invalid_queries_leave_saved_reference_and_rng_unchanged(case):
    reference = EpisodeProceduralReference(2)
    episodes, ages, offsets = torch.ones(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), [0.]
    if case != "uninitialized":
        reference.reset(state(), episodes)
    if case == "stale_episode":
        episodes[1] = 2
    elif case == "outside_offset":
        offsets = [5.01]
    elif case == "negative_age":
        ages[0] = -1
    elif case == "after_cap":
        ages[0] = 301
    saved = reference.state_dict()
    with pytest.raises(ValueError):
        reference.query(ages, episodes, offsets)
    assert torch.equal(saved["paths"], reference.paths)
    assert torch.equal(saved["source"]["generator_state"], reference.source.generator.get_state())


def test_invalid_resume_rejected_before_live_path_or_rng_mutation():
    reference = EpisodeProceduralReference(2)
    reference.reset(state(), torch.ones(2, dtype=torch.long))
    saved = reference.state_dict()
    invalid = deepcopy(saved)
    invalid["paths"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        reference.load_state_dict(invalid)
    assert torch.equal(saved["paths"], reference.paths)
    assert torch.equal(saved["source"]["generator_state"], reference.source.generator.get_state())
