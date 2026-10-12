"""Validate curriculum boundaries and preserve state on a corrupt resume."""

import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture(params=["crowdsim_ppo.goal_curriculum", "CrowdSim.ppo.goal_curriculum"])
def module(request):
    return importlib.import_module(request.param)


def settings():
    return {
        "enabled": True,
        "window_size": 4,
        "min_episodes": 2,
        "update_interval_episodes": 2,
        "stages": {
            "easy": {"distance": [2.0, 4.0], "heading_degrees": [0.0, 15.0]},
            "hard": {"distance": [4.0, 7.0], "heading_degrees": [15.0, 90.0]},
        },
    }


@pytest.mark.parametrize(
    "defect",
    [
        "window",
        "min",
        "interval",
        "window_small",
        "initial",
        "stage",
        "pair",
        "negative",
        "reverse",
        "heading",
    ],
)
def test_bad_curriculum_config_fails_before_navigation_mutation(module, defect):
    values = settings()
    if defect in ("window", "min", "interval", "window_small", "initial"):
        key, number = {
            "window": ("window_size", 0),
            "min": ("min_episodes", 0),
            "interval": ("update_interval_episodes", 0),
            "window_small": ("min_episodes", 5),
            "initial": ("initial_stage", 9),
        }[defect]
        values[key] = number
    elif defect == "stage":
        values["stages"]["easy"] = []
    else:
        field, raw = {
            "pair": ("distance", [1]),
            "negative": ("distance", [-1, 2]),
            "reverse": ("distance", [3, 2]),
            "heading": ("heading_degrees", [0, 181]),
        }[defect]
        values["stages"]["easy"][field] = raw
    config = SimpleNamespace(min_start_goal_distance=17.0)
    with pytest.raises(ValueError):
        module.PPOGoalCurriculum(config, values)
    assert vars(config) == {"min_start_goal_distance": 17.0}


@pytest.mark.parametrize("field", ["distance", "heading_degrees"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_goal_bounds_are_rejected(module, field, invalid):
    values = settings()
    values["stages"]["easy"][field][1] = invalid
    with pytest.raises(ValueError, match="Invalid curriculum stage"):
        module.GoalCurriculumSpec.from_dict(values)


def test_default_disabled_curriculum_keeps_navigation_and_empty_metrics(module):
    config = SimpleNamespace(min_start_goal_distance=17.0)
    disabled = module.PPOGoalCurriculum(config, None)
    assert not disabled.observe([True]) and disabled.metrics() == {}
    disabled.load_state_dict({"stage_index": 99})
    assert vars(config) == {"min_start_goal_distance": 17.0}
    assert disabled.stage_index == 0 and disabled.success_rate == 0.0


def test_enabled_curriculum_handles_empty_and_short_windows_and_clamped_stages(module):
    curriculum = module.PPOGoalCurriculum(SimpleNamespace(), settings())
    assert not curriculum.observe([]) and curriculum.completed_episodes == 0
    assert not curriculum.observe([True]) and curriculum.completed_episodes == 1
    assert curriculum.observe([True]) and curriculum.stage.name == "hard"
    assert not curriculum.observe([True, True]) and curriculum.stage.name == "hard"
    assert curriculum.observe([False] * 4) and curriculum.stage.name == "easy"
    assert not curriculum.observe([False] * 2) and curriculum.stage.name == "easy"
    metrics = curriculum.metrics()
    assert (
        metrics["curriculum/distance_max_m"] == 4.0
        and metrics["curriculum/completed_episodes"] == 10.0
    )
    before = curriculum.state_dict()
    curriculum.load_state_dict(None)
    assert curriculum.state_dict() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage_index", 99),
        ("outcomes", [float("nan")]),
        ("outcomes", [float("inf")]),
        ("outcomes", [0.5]),
        ("completed_episodes", -1),
        ("episodes_since_update", -1),
    ],
)
def test_invalid_resume_is_rejected_before_any_state_changes(module, field, value):
    curriculum = module.PPOGoalCurriculum(SimpleNamespace(), settings())
    curriculum.observe([True, True])
    state = curriculum.state_dict()
    corrupt = {**state, field: value}
    ranges = vars(curriculum.navigation_config).copy()
    with pytest.raises(ValueError):
        curriculum.load_state_dict(corrupt)
    assert curriculum.state_dict() == state
    assert vars(curriculum.navigation_config) == ranges
