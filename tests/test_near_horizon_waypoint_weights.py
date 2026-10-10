"""Execution-window priorities are explicit, normalized and pre-SDK safe."""

from pathlib import Path
import subprocess
import sys

import pytest

from CrowdSim.humanoid.waypoint_weighting import (
    near_horizon_waypoint_weights,
    near_horizon_weighting_description,
    validate_near_horizon_mass,
)


ROOT = Path(__file__).resolve().parents[1]


def test_zero_mass_is_an_actual_noop_sentinel_not_a_different_mean():
    assert near_horizon_waypoint_weights(None) is None
    assert near_horizon_waypoint_weights([2., 5.], 0) is None


def test_timed_ten_goals_redistribute_without_increasing_total_goal_strength():
    times = [.5*index for index in range(1, 11)]
    weights = near_horizon_waypoint_weights(times, .25)
    assert weights == pytest.approx([.2, .2]+[.075]*8)
    assert sum(weights) == pytest.approx(1.)
    assert [20*weight for weight in weights] == pytest.approx([4., 4.]+[1.5]*8)
    metadata = near_horizon_weighting_description(times, .25)
    assert metadata['engineering_variant_enabled'] and not metadata['author_hyperparameters']
    assert metadata['normalized_waypoint_weights'] == list(weights)
    assert not metadata['total_goal_weight_changed']


def test_full_mass_and_repeated_or_permuted_timestamps_retain_each_occurrence():
    times = [2., .5, 1., .5]
    assert near_horizon_waypoint_weights(times, 1.) == pytest.approx([0., 1/3, 1/3, 1/3])
    ordered = near_horizon_waypoint_weights(sorted(times), .25)
    assert sorted(near_horizon_waypoint_weights(times, .25)) == sorted(ordered)


@pytest.mark.parametrize('value', [True, False, -.01, 1.01, float('nan'), float('inf'), '0.25', None])
def test_invalid_mass_is_rejected_without_importing_a_framework(value):
    with pytest.raises(ValueError, match='mass'):
        validate_near_horizon_mass(value)


@pytest.mark.parametrize('times', [[], None, [0., 1.], [-1., .5], [float('nan')], [float('inf')], [True], ['.5']])
def test_enabled_bad_waypoint_times_fail_closed(times):
    with pytest.raises(ValueError, match='times'):
        near_horizon_waypoint_weights(times, .25)


def test_no_near_target_is_not_silently_normalized_into_a_far_goal():
    with pytest.raises(ValueError, match='at or before1s'):
        near_horizon_waypoint_weights([1.1, 5.], .25)


def test_pure_module_import_and_formula_do_not_import_torch_or_sdk():
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from CrowdSim.humanoid.waypoint_weighting import near_horizon_waypoint_weights
assert near_horizon_waypoint_weights([.5,1.,5.], .25)[0] > 1/3
assert not any(name == 'torch' or name.startswith(('isaacsim','isaaclab','omni','pxr')) for name in sys.modules)
"""
    subprocess.run([sys.executable, '-I', '-c', code], check=True, capture_output=True, text=True)
