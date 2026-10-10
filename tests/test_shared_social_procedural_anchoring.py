import numpy as np
import pytest

from CrowdSim.humanoid.analysis.procedural_anchoring import reference_clocks, reward_summary


def test_post_clock_boundary_does_not_restart_last_reward_early():
    origin, local, episode = reference_clocks()
    assert origin[14] == 0 and origin[15] == 15 and origin[149] == 135
    assert local[14] == .5 and local[149] == .5
    assert local[15] == np.float32(1 / 30)
    assert episode[14] == .5 and episode[149] == 5
    np.testing.assert_array_equal(local[:15], episode[:15])


@pytest.mark.parametrize("controls,period", [(True, 15), (151, 15), (150, 0), (10, 15)])
def test_invalid_reference_clocks(controls, period):
    with pytest.raises(ValueError):
        reference_clocks(controls, period)


def test_complete_population_summary_and_analytic_constant_speed_path():
    _, local, episode = reference_clocks()
    distance = np.stack((local, episode))[:, :, None]
    delta = distance ** 2
    rewards = np.exp(-2 * delta)
    result = reward_summary(delta, rewards)
    assert result[0]["actor_controls"] == 150
    assert result[0]["last_control_mean_trajectory_reward"] == float(np.exp(np.float32(-.5)))
    assert result[1]["mean_root_target_distance_m"] == pytest.approx(151 / 60)
    assert result[0]["mean_trajectory_reward"] > result[1]["mean_trajectory_reward"]


def test_actual_smooth_generator_retains_start_ramp_every_replan():
    from types import SimpleNamespace

    import torch

    from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig
    from scripts.diagnose_shared_social_procedural_anchoring import probe

    state = SimpleNamespace(root_pos=torch.zeros(1, 3), root_rot=torch.tensor([[0., 0., 0., 1.]]),
                            root_vel=torch.zeros(1, 3))
    saved = torch.random.get_rng_state().clone()
    arrays, result = probe(state, torch, PacerTrajectoryConfig(speed_min_mps=1., speed_max_mps=1.,
        heading_rate_max_rad_s=1e-8, sharp_turn_probability=0.))
    assert torch.equal(saved, torch.random.get_rng_state())
    assert torch.count_nonzero(state.root_pos) == 0
    assert result["actual_generator_plans"] == 10
    assert arrays["target_xy"][0, 14, 0, 0] == pytest.approx(.3)
    assert arrays["target_xy"][0, 29, 0, 0] == pytest.approx(.3)
    assert arrays["target_xy"][2, 149, 0, 0] == pytest.approx(4.8)
    assert arrays["trajectory_reward"][0, 29, 0] == pytest.approx(np.exp(-.18))


@pytest.mark.parametrize("invalid", ["negative", "nonfinite", "unbounded", "wrong_dtype"])
def test_invalid_probe_summary(invalid):
    delta, rewards = np.zeros((3, 2, 1), np.float32), np.ones((3, 2, 1), np.float32)
    if invalid == "negative":
        delta[0, 0, 0] = -1
    elif invalid == "nonfinite":
        rewards[0, 0, 0] = np.nan
    elif invalid == "unbounded":
        rewards[0, 0, 0] = 2
    else:
        delta = delta.astype(np.float64)
    with pytest.raises(ValueError):
        reward_summary(delta, rewards)
