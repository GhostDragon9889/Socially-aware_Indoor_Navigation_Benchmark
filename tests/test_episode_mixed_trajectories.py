"""CPU command/reset/state contracts, without PhysX or policy-quality claims."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

from CrowdSim.humanoid.training.episode_mixed_trajectories import EpisodeAnchoredMixedCurriculum
from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig

SPEC = importlib.util.spec_from_file_location("episode_mixed_CPU_fixture", Path(__file__).with_name("test_mixed_trajectories.py"))
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def make(tmp_path, *, original=False, cap=300):
    corpus = HELPER.corpus(tmp_path)
    kwargs = {"procedural_config": PacerTrajectoryConfig(), "bounds_xy": (-100., -100., 100., 100.)}
    if original:
        return MixedTrajectoryCurriculum(corpus, 4, **kwargs)
    return EpisodeAnchoredMixedCurriculum(corpus, 4, episode_controls=cap, **kwargs)


def test_all_original_reset_sources_histories_and_AMASS_targets_remain_exact(tmp_path):
    baseline, candidate = make(tmp_path, original=True), make(tmp_path)
    ids = torch.tensor([3, 0, 2, 1])
    HELPER.assert_reference_equal(baseline.sample_reference(ids), candidate.sample_reference(ids))
    state, _, ages, episodes, first = HELPER.setup(candidate)
    _, _, old_ages, old_episodes, old = HELPER.setup(baseline)
    assert torch.equal(candidate.private_corpus.generator.get_state(), baseline.private_corpus.generator.get_state())
    assert torch.equal(candidate.corpus.generator.get_state(), baseline.corpus.generator.get_state())
    assert candidate.config["real_reset_future_frames"] == 31
    assert torch.equal(first.native_xy, old.native_xy)
    assert torch.equal(candidate.episode_reference.paths[:, :51], old.xy[2:])
    assert torch.equal(candidate.future_targets(ages, ages, first.native_xy, episodes),
                       baseline.future_targets(old_ages, old_ages, old.native_xy, old_episodes))
    assert torch.equal(state.root_pos[:, :2], torch.tensor([[10., 10.], [20., 10.], [30., 10.], [40., 10.]]))


def test_replanning_does_not_reanchor_or_redraw_but_original_AMASS_plan_moves_forward(tmp_path):
    candidate = make(tmp_path)
    state, origins, ages, episodes, first = HELPER.setup(candidate)
    saved_rng = candidate.generator.get_state().clone()
    paths = candidate.episode_reference.paths.clone()
    HELPER.advance(candidate, ages, 15)
    state.root_pos[2:, :2] += 100
    second = candidate.build_plan(state, ages, episode_ids=episodes)
    assert torch.equal(saved_rng, candidate.generator.get_state())
    assert torch.equal(paths, candidate.episode_reference.paths)
    assert not torch.equal(second.native_xy[2:, 0], state.root_pos[2:, :2])
    assert not torch.equal(first.native_xy[:2, 0], second.native_xy[:2, 0])
    assert candidate.statistics["procedural_plans"] == 2
    assert candidate.statistics["procedural_plan_views"] == 4
    HELPER.advance(candidate, ages)
    target = candidate.interpolate_reference(torch.ones(4, 1) / 30, second.native_xy)
    assert torch.equal(target[2:], candidate.episode_reference.query(ages[2:], episodes[2:], [0.]))
    assert torch.equal(candidate.future_targets(ages, torch.ones(4, dtype=torch.long), second.native_xy, episodes)[2:],
                       candidate.episode_reference.query(ages[2:], episodes[2:], torch.arange(1, 11).float() * .5))
    reference_ids = torch.cat((candidate.amass.motion_ids, candidate.proc_reference_ids))
    candidate.validate_runner_state(ages, torch.ones(4, dtype=torch.long), second.native_xy, reference_ids, episodes,
                                    replan_steps=15, origins_xy=origins)
    broken = second.native_xy.clone()
    broken[2, 0, 0] += 1
    with pytest.raises(ValueError, match="native view"):
        candidate.validate_runner_state(ages, torch.ones(4, dtype=torch.long), broken, reference_ids, episodes,
                                        replan_steps=15, origins_xy=origins)


def test_mixed_state_restores_paths_views_source_clocks_and_future_reset_rng(tmp_path):
    candidate = make(tmp_path)
    state, _, ages, episodes, _ = HELPER.setup(candidate)
    HELPER.advance(candidate, ages, 15)
    view = candidate.build_plan(state, ages, episode_ids=episodes)
    saved = candidate.state_dict()
    external_rng = candidate.corpus.generator.get_state().clone()
    restored = make(tmp_path)
    caller_rng_before = restored.corpus.generator.get_state().clone()
    restored.load_state_dict(saved)
    assert torch.equal(caller_rng_before, restored.corpus.generator.get_state())
    restored.corpus.generator.set_state(external_rng)  # Explicit caller-owned expert/reset source continuation.
    assert torch.equal(restored.episode_reference.paths, candidate.episode_reference.paths)
    assert torch.equal(restored.generator.get_state(), candidate.generator.get_state())
    assert torch.equal(restored.future_targets(ages, torch.zeros(4, dtype=torch.long), view.native_xy, episodes),
                       candidate.future_targets(ages, torch.zeros(4, dtype=torch.long), view.native_xy, episodes))
    HELPER.assert_reference_equal(restored.sample_reference([2]), candidate.sample_reference([2]))


@pytest.mark.parametrize("bad", ["RNG_owner", "view", "episode", "field_dtype"])
def test_malformed_mixed_state_rejected_without_mutating_live_sources_or_paths(tmp_path, bad):
    candidate = make(tmp_path)
    HELPER.setup(candidate)
    before = candidate.state_dict()
    invalid = deepcopy(before)
    if bad == "RNG_owner":
        invalid["episode_reference"]["source"]["plans_generated"] += 1
    elif bad == "view":
        invalid["procedural_actors"]["canonical_last_plans"][0, 0, 0] += 1
    elif bad == "episode":
        invalid["episode_reference"]["episode_ids"][0] += 1
    else:
        invalid["procedural_actors"]["plan_valid"] = invalid["procedural_actors"]["plan_valid"].long()
    with pytest.raises(ValueError):
        candidate.load_state_dict(invalid)
    assert torch.equal(before["generator_state"], candidate.generator.get_state())
    assert torch.equal(before["episode_reference"]["paths"], candidate.episode_reference.paths)
    assert before["statistics"] == candidate.statistics


def test_partial_reset_regenerates_only_the_fresh_episode(tmp_path):
    candidate = make(tmp_path)
    state, origins, ages, episodes, _ = HELPER.setup(candidate)
    old_path = candidate.episode_reference.paths[1].clone()
    HELPER.advance(candidate, ages, 15)
    reference, _, _, mid, frames = candidate.sample_reference([2])
    state.root_pos[2, :2] = origins[2] + 10
    candidate.bind_reset([2], mid, frames, reference.root_pos[:, :2], state.root_pos[2:3, :2], torch.tensor([2]))
    episodes[2], ages[2] = 2, 0
    candidate.build_plan(state, ages, actor_ids=[2], episode_ids=episodes)
    assert torch.equal(old_path, candidate.episode_reference.paths[1])
    assert torch.equal(candidate.episode_reference.paths[0, 0], state.root_pos[2, :2])
    assert candidate.statistics["procedural_plans"] == 3


def test_four_control_smoke_preserves_physical_cap_and_full_future_observation(tmp_path):
    candidate = make(tmp_path, cap=4)
    _, _, ages, episodes, plan = HELPER.setup(candidate)
    HELPER.advance(candidate, ages, 4)
    assert candidate.episode_controls == candidate.config["physical_episode_controls"] == 4
    assert candidate.episode_reference.episode_controls == 6
    targets = candidate.future_targets(ages, ages, plan.native_xy, episodes)
    assert targets.shape == (4, 10, 2) and torch.isfinite(targets).all()
    HELPER.advance(candidate, ages)
    with pytest.raises(ValueError, match="physical episode cap"):
        candidate.future_targets(ages, ages, plan.native_xy, episodes)
