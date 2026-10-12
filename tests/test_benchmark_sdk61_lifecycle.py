"""Benchmark ownership integration with the real lifecycle and CPU SDK fixtures."""

import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_sdk61_lifecycle import make_world, runtime  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("failure_stage", [None, "config", "construct", "initialize", "step", "cleanup", "frame"])
def test_benchmark_releases_complete_and_partial_sdk_owners(runtime, monkeypatch, tmp_path, failure_stage):  # noqa: F811 - imported pytest fixture
    from CrowdSim.humanoid.runtime import sdk61_lifecycle, trace_pace_scope
    monkeypatch.setattr(sdk61_lifecycle, "uses_sim61", lambda: True)
    monkeypatch.setattr(sdk61_lifecycle, "SDK61Lifecycle", runtime.module.SDK61Lifecycle)
    from protomotions.components.scene_lib import SceneLib
    monkeypatch.setattr(SceneLib, "empty", lambda **kwargs: object())
    terrain = ModuleType("protomotions.components.terrains.terrain")
    terrain.Terrain = lambda **kwargs: SimpleNamespace(sample_valid_locations=lambda count: torch.zeros(count, 2))
    monkeypatch.setitem(sys.modules, terrain.__name__, terrain)

    app = runtime.App()
    launcher = ModuleType("isaaclab.app")
    launcher.AppLauncher = lambda flags: SimpleNamespace(app=app)
    monkeypatch.setitem(sys.modules, launcher.__name__, launcher)
    if failure_stage == "config":
        # No native backend exists before a SimulationContext is constructed.
        runtime.backend.simulation_view = None
        from protomotions.robot_configs import h1_2
        def bad_config():
            raise RuntimeError("benchmark config failure")
        monkeypatch.setattr(h1_2, "H1_2RobotConfig", bad_config)

    class Simulator:
        def __init__(self, config, robot_config, terrain, device, simulation_app, scene_lib):
            runtime.Context.current = runtime.Sim()
            owned, *_ = make_world(runtime)
            self.__dict__.update(vars(owned.simulator))
            self.config, self.robot_config, self.terrain = config, robot_config, terrain
            self._simulation_app = simulation_app
            if failure_stage == "construct":
                native_alias = runtime.backend.simulation_view
                assert native_alias is not None
                try:
                    raise ValueError("original constructor cause")
                except ValueError as cause:
                    raise RuntimeError("benchmark construct failure") from cause
        def _initialize_with_markers(self, markers):
            if failure_stage == "initialize":
                native_alias = runtime.backend.simulation_view
                assert native_alias is not None
                raise RuntimeError("benchmark initialize failure")
        def get_default_robot_reset_state(self):
            return SimpleNamespace(root_pos=torch.zeros(self.config.num_envs, 3))
        def reset_envs(self, state, env_ids):
            pass
        def step(self, action):
            if failure_stage == "step":
                native_alias = runtime.backend.simulation_view
                assert native_alias is not None
                raise RuntimeError("benchmark step failure")
        def close(self):
            pytest.fail("SDK61 owner must close through its lifecycle exactly once")
    module = ModuleType("protomotions.simulator.isaaclab.simulator")
    module.IsaacLabSimulator = Simulator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if failure_stage == "cleanup":
        def bad_stop(self):
            raise RuntimeError("benchmark stop failure")
        monkeypatch.setattr(runtime.Sim, "stop", bad_stop)
    if failure_stage == "frame":
        actual_clear = trace_pace_scope.clear_completed_physics_frame
        def bad_frame(owner, function):
            actual_clear(owner, function)
            raise RuntimeError("benchmark frame receipt failure")
        monkeypatch.setattr(trace_pace_scope, "clear_completed_physics_frame", bad_frame)

    namespace = runpy.run_path(str(ROOT / "examples/benchmark/isaaclab_bench.py"))
    report_path = tmp_path / "result.json"
    args = ["--num-envs", "2", "--warmup-steps", "1", "--steps", "2", "--device", "cpu",
            "--report", str(report_path)]
    if failure_stage:
        message = "Benchmark cleanup failed" if failure_stage in ("cleanup", "frame") else f"benchmark {failure_stage} failure"
        with pytest.raises(RuntimeError, match=message):
            namespace["main"](args)
    else:
        namespace["main"](args)
    report = json.loads(report_path.read_text())
    assert report["app_close_returned"] is True
    closes = [event for event in runtime.events if isinstance(event, tuple) and event[0] == "app"]
    assert len(closes) == 1
    assert closes[0][1] == {"exit_code": int(failure_stage is not None),
                            "wait_for_replicator": False, "skip_cleanup": False}
    assert runtime.backend.simulation_view is None
    assert runtime.Context.instance() is None
    assert report["cleanup_succeeded"] is (failure_stage not in ("cleanup", "frame"))
    if failure_stage not in ("frame",):
        assert report["frame_release"]["owned_completed_frame_cleared"] is True
    if failure_stage == "construct":
        assert report["partial_construction_cleanup"] is True
        assert "original constructor cause" in report["traceback"]
    if failure_stage not in ("config",):
        assert report["public_backend_view_live_before_App_close"] is False
    if failure_stage is None:
        assert report["status"] == "passed"
        assert report["completed_warmup_steps"] == 1
        assert report["completed_measured_steps"] == 2
