"""CPU analytic tests; fake SDK observations are never actual Office evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("local_Office_gravity_test", Path(__file__).with_name("observe_configured_gravity.py"))
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def fixture():
    class Scene:
        def __init__(self, prim):
            self.prim = prim

        def GetGravityDirectionAttr(self):
            return SimpleNamespace(Get=lambda: self.prim.direction)

        def GetGravityMagnitudeAttr(self):
            return SimpleNamespace(Get=lambda: self.prim.magnitude)

    prim = SimpleNamespace(direction=[0.0, 0.0, -1.0], magnitude=9.8100004196,
                           IsActive=lambda: True, IsA=lambda schema: schema is Scene,
                           GetPath=lambda: "/physicsScene")
    stage = SimpleNamespace(Traverse=lambda: [prim])
    context = SimpleNamespace(stage=stage, cfg=SimpleNamespace(gravity=[0.0, 0.0, -9.81],
                              dt=1.0 / 120.0, physics_prim_path="/physicsScene"),
                              get_physics_step_count=lambda: 4)
    bindings = (SimpleNamespace(instance=lambda: context), lambda: stage,
                SimpleNamespace(Scene=Scene))
    return context, prim, bindings


def test_consistent_config_and_USD_gravity_read_without_physics_claim():
    context, _, bindings = fixture()
    report = TOOL.observe_configured_gravity(context, _bindings=bindings)
    assert report["configured_gravity_m_s2"] == [0.0, 0.0, -9.81]
    assert report["submitted_step_counter_before"] == report["submitted_step_counter_after"] == 4
    assert report["physics_steps_advanced_by_observer"] == 0
    assert report["native_engine_gravity_measured"] is False
    assert report["human_stability_or_navigation_verified"] is False


def test_wrong_context_zero_or_different_gravity_and_clock_advance_rejected():
    context, prim, bindings = fixture()
    with pytest.raises(ValueError, match="singleton"):
        TOOL.observe_configured_gravity(SimpleNamespace(), _bindings=bindings)
    context.cfg.gravity = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="Earth gravity"):
        TOOL.observe_configured_gravity(context, _bindings=bindings)
    context.cfg.gravity = [0.0, 0.0, -9.81]
    prim.magnitude = 1.0
    with pytest.raises(ValueError, match="differs"):
        TOOL.observe_configured_gravity(context, _bindings=bindings)
    prim.magnitude = 9.81
    counter = iter((4, 5))
    context.get_physics_step_count = lambda: next(counter)
    with pytest.raises(ValueError, match="must not advance"):
        TOOL.observe_configured_gravity(context, _bindings=bindings)


def test_instance_adapter_preserves_original_floor_verifier_and_arguments():
    calls = []
    original = {"scene": "Office", "submitted_step_counter_after": 4, "floor_paths": ["/__PaperMapProbeReferenceGround"]}
    snapshot = deepcopy(original)
    provider = SimpleNamespace(scene_name="Office")

    def verify(**kwargs):
        calls.append(("verify", kwargs))
        return original

    provider.verify_live_runtime = verify
    scene = SimpleNamespace(provider=provider)

    def factory(description, *, device):
        calls.append(("factory", description, device))
        return scene

    gravity = {"submitted_step_counter_before": 4}
    wrapped = TOOL.wrap_native_scene_factory(factory, observer=lambda ctx: gravity)
    assert wrapped({"scene": "Office"}, device="cuda:0") is scene
    context = object()
    result = provider.verify_live_runtime(origins_xy=[[1, 2]], simulation_context=context, free_cell_limit=128)
    assert calls == [("factory", {"scene": "Office"}, "cuda:0"),
                     ("verify", {"origins_xy": [[1, 2]], "simulation_context": context, "free_cell_limit": 128})]
    assert result == {**snapshot, "local_configured_gravity_observation": gravity}
    assert original == snapshot
    with pytest.raises(ValueError, match="already"):
        wrapped({"scene": "Office"}, device="cuda:0")
