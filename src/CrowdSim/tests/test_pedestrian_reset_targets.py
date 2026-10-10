"""CPU contracts for standing navigation targets and checked walking resets."""

from protomotions.repository import repository_root

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import yaml

from CrowdSim.humanoid.motion_reset import configure_pedestrian_reset_motions
from CrowdSim.nav_manager import CrowdNavigationConfig, CrowdNavigationManager
from CrowdSim.tests.test_masked_mimic_navigation_subsets import manager_and_context
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager
from protomotions.simulator.base_simulator.simulator_state import StateConversion


def test_standing_target_restores_height_above_future_ground_for_packed_ids():
    manager, component, ctx = manager_and_context([2, 4, 1])
    manager.config.humanoid_pelvis_height_above_ground = .93
    ctx.current.rigid_body_pos[0, 1, 2] = .4  # Already sagged, unlike desired height.
    ctx.current.rigid_body_pos[2, 1, 2] = .6
    queried = []

    def ground_at(locations):
        queried.append(locations.clone())
        return (.1 * locations[:, 0] + .2 * locations[:, 1])[:, None]

    component.env.terrain = NS(get_ground_heights=ground_at)
    manager._override_masked_mimic_context(component, ctx)
    positions = ctx.masked_mimic.ref_pos[[0, 2], :, 1]
    torch.testing.assert_close(positions[..., 2], .1 * positions[..., 0] + .2 * positions[..., 1] + .93)
    assert queried[0].shape == (4, 3)
    # Frozen filler retains its own holding pose and does not get a walking goal.
    torch.testing.assert_close(ctx.masked_mimic.ref_pos[1, :, 1, 2], ctx.current.rigid_body_pos[1, 1, 2].expand(2))


def test_nonfinite_target_ground_fails_instead_of_silently_holding_falling_height():
    manager, component, ctx = manager_and_context([1])
    manager.config.humanoid_pelvis_height_above_ground = .93
    component.env.terrain = NS(get_ground_heights=lambda points: torch.full((len(points), 1), float("nan")))
    with pytest.raises(ValueError, match="terrain heights"):
        manager._override_masked_mimic_context(component, ctx)


def test_reset_returns_observations_recomputed_after_new_navigation_for_subset():
    manager = object.__new__(CrowdNavigationManager)
    manager.config = CrowdNavigationConfig(Path("unused"), .1, 200, 3, 0, torch.device("cpu"),
        trajectory_recording_enabled=False, car_rl_policy=False)
    manager.sfm_controller = NS(dt=.04)
    manager.task = NS(create_visualization_markers=lambda **kwargs: None)
    manager.path_log_path = Path("unused")
    manager._trajectory_log_file = manager._trajectory_timestamp_file = None
    manager.drive = None
    manager.env_step_count = 0
    manager._has_completed_initial_reset = False
    calls = []
    manager._reset_navigation_agents = lambda ids: calls.append(("replan", ids.tolist()))
    manager._read_agent_state = lambda: (np.zeros((3, 2)), np.ones((3, 2)))
    manager._compute_sfm_reference_waypoints = lambda positions, velocities: calls.append(("new_targets", velocities.tolist()))
    manager._update_local_target_markers = lambda: None
    manager._record_trajectory_frame = lambda *args: None

    def install_hook(env):
        manager._masked_mimic_navigation_hook_installed = True

    manager._attach_masked_mimic_navigation_targets = install_hook
    old_obs = {"goal": torch.tensor(-1.)}
    new_obs = {"goal": torch.tensor(1.)}
    info = {"reset_metadata": "preserve"}

    def original_reset(ids):
        calls.append(("original_reset_old_targets", ids.tolist()))
        return old_obs, info

    def compute_observations(ids, *, context):
        assert calls[-1][0] == "new_targets"
        calls.append(("refresh", ids.tolist(), context.tolist()))

    env = NS(num_envs=3, device=torch.device("cpu"), dt=.04,
        simulator=NS(headless=True, native_quaternion_order="xyzw"),
        reset=original_reset, step=lambda action: None,
        compute_observations=compute_observations, _build_context=lambda ids: ids,
        get_obs=lambda: new_obs, _current_context="stale")
    manager.attach(env)
    result = env.reset(torch.tensor([2, 0]))
    assert result[0] is new_obs and result[1] is info
    assert calls[0] == ("replan", [2, 0])
    assert calls[1] == ("original_reset_old_targets", [2, 0])
    assert calls[-1] == ("refresh", [2, 0], [2, 0])
    assert env._current_context is None


def _tiny_library():
    library = MotionLib(MotionLibConfig(), device="cpu")
    library.motion_file = "test.pt"
    library.get_motion_state_use_blend = True
    library.motion_files = ("source/Walking_1_poses.motion", "source/jiggles.motion", "source/Walking_2_poses.motion")
    library.motion_weights = torch.ones(3)
    library.motion_lengths = torch.ones(3)
    library.motion_num_frames = torch.full((3,), 2, dtype=torch.long)
    library.motion_dt = torch.ones(3)
    library.length_starts = torch.tensor([0, 2, 4])
    library.gts = torch.zeros(6, 1, 3)
    library.gts[:, 0, 2] = .93
    library.grs = torch.tensor([0., 0., 0., 1.]).repeat(6, 1, 1)
    library.gvs = torch.zeros(6, 1, 3)
    library.gavs = torch.zeros(6, 1, 3)
    library.dps = torch.zeros(6, 3)
    library.dvs = torch.zeros(6, 3)
    library.lrs = None
    library.contacts = None
    return library


def _environment(library, **config):
    manager = MimicMotionManager(MimicMotionManagerConfig(init_start_prob=1., **config),
        num_envs=2, env_dt=1 / 30, device=torch.device("cpu"), motion_lib=library)
    return NS(motion_lib=library, motion_manager=manager)


def test_walking_filter_preserves_exclusions_across_weight_updates_and_subset_reset():
    env = _environment(_tiny_library(), exclude_motion_ids=[2])
    report = configure_pedestrian_reset_motions(env, {"reset_motion_patterns": ["*/Walking_*_poses.motion"]})
    assert report["selected_motion_ids"] == [0, 2]
    assert report["eligible_motion_ids"] == [0]
    manager = env.motion_manager
    manager.update_sampling_weights(torch.ones(3))
    assert torch.equal(manager.sample_n_motion_ids(10000), torch.zeros(10000, dtype=torch.long))
    manager.motion_ids[1] = 2
    manager.motion_times[1] = .5
    manager.sample_motions(torch.tensor([0]))
    assert manager.motion_ids.tolist() == [0, 2]
    assert manager.motion_times.tolist() == [0., .5]


@pytest.mark.parametrize("problem", ["missing", "downward", "tilted", "low", "nonfinite", "excluded", "fixed"])
def test_bad_walking_reset_contract_fails_without_mutating_sampler(problem):
    library = _tiny_library()
    env = _environment(library, **({"subset_method": [1, 1]} if problem == "fixed" else {}))
    patterns = ["*/Walking_*_poses.motion"]
    if problem == "missing":
        patterns = ["*/not_installed.motion"]
    elif problem == "downward":
        library.gvs[0, 0, 2] = -.386
    elif problem == "tilted":
        library.grs[0, 0] = torch.tensor([1., 0., 0., 0.])
    elif problem == "low":
        library.gts[0, 0, 2] = .4
    elif problem == "nonfinite":
        library.dps[0, 0] = float("nan")
    elif problem == "excluded":
        env.motion_manager._sampling_eligible_motion_mask[[0, 2]] = False
    original_mask = env.motion_manager._sampling_eligible_motion_mask.clone()
    original_weights = env.motion_manager.motion_weights.clone()
    with pytest.raises(ValueError):
        configure_pedestrian_reset_motions(env, {"reset_motion_patterns": patterns})
    torch.testing.assert_close(env.motion_manager._sampling_eligible_motion_mask, original_mask)
    torch.testing.assert_close(env.motion_manager.motion_weights, original_weights)
    assert not hasattr(env, "_crowdsim_reset_motion_manifest")


@pytest.mark.needs_lfs
def test_actual_amass_validation_walking_reset_distribution_and_original_unchanged():
    root = repository_root(__file__)
    motion = root.parent / "Assets/motion/amass_smpl_validation.pt"
    if not motion.is_file() or motion.stat().st_size < 10000:
        pytest.skip("Local AMASS validation tensor library is not installed")
    original_stat = motion.stat()
    with torch.serialization.safe_globals([StateConversion]):
        packaged = torch.load(motion, weights_only=True, mmap=True, map_location="cpu")
    # Use the real MotionLib query/rotation implementation with mapped tensors;
    # no model unpickling, interpolation mock, full train load or source rewrite.
    library = MotionLib(MotionLibConfig(), device="cpu")
    for name, value in packaged.items():
        setattr(library, name, value)
    library.motion_file = str(motion)
    library.get_motion_state_use_blend = True
    config = yaml.safe_load((root / "CrowdSim/config/pedestrian.yaml").read_text())["humanoid"]
    env = _environment(library)
    report = configure_pedestrian_reset_motions(env, config)
    expected = [0, 10, 19, 26, 234, 235, 244, 253, 254]
    assert report["selected_motion_ids"] == expected
    torch.manual_seed(7)
    sampled = env.motion_manager.sample_n_motion_ids(10000)
    assert set(sampled.tolist()) == set(expected)
    assert 315 not in sampled and 239 not in sampled and 246 not in sampled
    assert max(abs(value) for value in report["vertical_velocities_m_s"]) <= .15
    assert max(report["root_tilt_degrees"]) <= 15.
    np.testing.assert_array_equal(library.dps[packaged["length_starts"].long()].numpy(), packaged["dps"][packaged["length_starts"].long()].numpy())
    assert motion.stat().st_size == original_stat.st_size
    assert motion.stat().st_mtime_ns == original_stat.st_mtime_ns
