"""CPU regression checks; these do not launch Kit or render sensor frames."""

from datetime import datetime
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch

from CrowdSim.utils import sensor_stream
from CrowdSim.tools import test_camera_depth


ROOT = Path(__file__).resolve().parents[1]


def test_camera_restart_in_same_second_preserves_both_recordings(monkeypatch, tmp_path):
    fixed_time = Mock()
    fixed_time.now.return_value = datetime(2026, 10, 11, 12, 0, 0)
    monkeypatch.setattr(sensor_stream, "datetime", fixed_time)
    # An existing session must also remain untouched when a new process starts.
    occupied = tmp_path / "20261011_120000"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("earlier recording")
    data = SimpleNamespace(image_shape=(2, 2), intrinsic_matrices=None, output={
        "rgb": torch.full((1, 2, 2, 3), 42, dtype=torch.uint8),
        "distance_to_image_plane": torch.ones(1, 2, 2, 1),
    })
    recorder = sensor_stream.RobotCameraRecorder(
        SimpleNamespace(crowdsim_robot_camera=SimpleNamespace(data=data), dt=.1),
        tmp_path, 10., [0],
    )
    recorder.toggle()
    recorder.step()
    first = recorder.session_dir
    recorder.toggle()
    data.output["rgb"].fill_(99)
    data.output["distance_to_image_plane"].fill_(2.)
    recorder.toggle()
    recorder.step()
    second = recorder.session_dir
    assert len({occupied, first, second}) == 3
    assert (occupied / "keep.txt").read_text() == "earlier recording"
    for directory, pixel, depth in ((first, 42, 1.), (second, 99, 2.)):
        with Image.open(directory / "env_0000/rgb_000000.png") as frame:
            assert frame.getpixel((0, 0)) == (pixel, pixel, pixel)
        torch.testing.assert_close(
            torch.load(directory / "env_0000/depth_000000.pt", weights_only=True),
            torch.full((2, 2, 1), depth),
        )
        assert (directory / "metadata.pt").is_file()


@pytest.mark.parametrize("depth", [
    np.full((2, 2), np.inf, dtype=np.float32),
    np.array([[np.nan, 0.], [-np.inf, -1.]], dtype=np.float32),
    np.array([[np.inf, 2.], [np.nan, 4.]], dtype=np.float32),
])
def test_camera_capture_preserves_raw_depth_with_no_hits(monkeypatch, tmp_path, capsys, depth):
    monkeypatch.chdir(tmp_path)
    callbacks = {}
    camera = SimpleNamespace(data=SimpleNamespace(output={
        "distance_to_image_plane": torch.from_numpy(depth)[None, ..., None],
    }))
    env = SimpleNamespace(num_envs=1, crowdsim_robot_camera=camera, simulator=SimpleNamespace(
        keyboard_interface=SimpleNamespace(add_callback=lambda key, fn: callbacks.setdefault(key, fn)),
    ))
    test_camera_depth.add_camera_check_key(env)
    callbacks["C"]()
    output = capsys.readouterr().out
    directory = tmp_path / "output/renderings/camera_check"
    np.testing.assert_array_equal(np.load(directory / "depth_env0.npy"), depth)
    assert (directory / "depth_env0.png").is_file()
    if np.any(np.isfinite(depth) & (depth > 0)):
        assert "depth [2.00~4.00]m avg=3.00m" in output
    else:
        assert "no valid depth hits" in output


@pytest.mark.parametrize("failure_stage", [None, "config", "construct", "step"])
def test_isaaclab_benchmark_uses_current_config_and_requested_step_counts(monkeypatch, capsys, failure_stage):
    """Run the CLI body with real CPU config/robot parsing and a fake physics owner."""
    from CrowdSim.humanoid.runtime import sdk61_lifecycle
    monkeypatch.setattr(sdk61_lifecycle, "uses_sim61", lambda: False)
    events = []
    close_codes = []
    def close_app(**kwargs):
        close_codes.append(kwargs.get("exit_code", 0))
        events.append("app_close")
    app = SimpleNamespace(close=close_app)
    app_module = ModuleType("isaaclab.app")
    def launch(flags):
        events.append(("launch", flags))
        return SimpleNamespace(app=app)
    app_module.AppLauncher = launch
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)

    simulator_module = ModuleType("protomotions.simulator.isaaclab.simulator")
    class Simulator:
        def __init__(self, config, robot_config, terrain, device, simulation_app, scene_lib,
                     custom_key_handlers=None):
            assert device == torch.device("cpu")
            assert simulation_app is app
            if failure_stage == "construct":
                raise RuntimeError("benchmark construct failure")
            self.config, self.robot_config = config, robot_config
            self.terrain, self._simulation_app = terrain, simulation_app
            events.append(("construct", config.num_envs))
        def _initialize_with_markers(self, markers):
            events.append("initialize")
        def get_default_robot_reset_state(self):
            return SimpleNamespace(root_pos=torch.zeros(self.config.num_envs, 3))
        def reset_envs(self, state, env_ids):
            assert env_ids.tolist() == [0, 1]
            events.append("reset")
        def step(self, action):
            assert action.shape == (2, self.robot_config.number_of_actions)
            events.append("step")
            if failure_stage == "step":
                raise RuntimeError("benchmark step failure")
        def close(self):
            events.append("simulator_close")
            app.close(exit_code=self._shutdown_exit_code)
    simulator_module.IsaacLabSimulator = Simulator
    monkeypatch.setitem(sys.modules, simulator_module.__name__, simulator_module)

    from protomotions.components.scene_lib import SceneLib
    terrain_module = ModuleType("protomotions.components.terrains.terrain")
    terrain_module.Terrain = lambda **kwargs: SimpleNamespace(
        sample_valid_locations=lambda count: torch.zeros(count, 2))
    monkeypatch.setitem(sys.modules, terrain_module.__name__, terrain_module)
    monkeypatch.setattr(SceneLib, "empty", lambda **kwargs: object())
    if failure_stage == "config":
        from protomotions.robot_configs import h1_2
        def bad_config():
            raise RuntimeError("benchmark config failure")
        monkeypatch.setattr(h1_2, "H1_2RobotConfig", bad_config)
    monkeypatch.setattr(sys, "argv", ["isaaclab_bench.py", "--num-envs", "2", "--steps", "2",
                                      "--warmup-steps", "1", "--device", "cpu"])
    if failure_stage:
        with pytest.raises(RuntimeError, match=f"benchmark {failure_stage} failure"):
            runpy.run_path(str(ROOT / "examples/benchmark/isaaclab_bench.py"), run_name="__main__")
        assert f"benchmark {failure_stage} failure" in capsys.readouterr().err
        assert events[-1] == "app_close" and events.count("app_close") == 1
        assert events.count("simulator_close") == int(failure_stage == "step")
        assert close_codes == [1]
        return
    runpy.run_path(str(ROOT / "examples/benchmark/isaaclab_bench.py"), run_name="__main__")
    assert events[0] == ("launch", {"headless": True, "device": "cpu", "fast_shutdown": False})
    assert events[1:] == [("construct", 2), "initialize", "reset", "step", "step", "step",
                          "simulator_close", "app_close"]
    assert close_codes == [0]
    assert "Simulator close returned" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [["--steps", "0"], ["--num-envs", "0"], ["--warmup-steps", "-1"]])
def test_isaaclab_benchmark_rejects_invalid_counts_before_launch(monkeypatch, arguments):
    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = lambda *args: pytest.fail("invalid CLI must not launch Kit")
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setattr(sys, "argv", ["isaaclab_bench.py", *arguments])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ROOT / "examples/benchmark/isaaclab_bench.py"), run_name="__main__")
    assert error.value.code == 2


def test_isaaclab_benchmark_defaults_keep_parallelism_and_step_counts():
    namespace = runpy.run_path(str(ROOT / "examples/benchmark/isaaclab_bench.py"))
    args = namespace["parse_args"]([])
    assert (args.num_envs, args.warmup_steps, args.steps, args.device) == (4096, 100, 1000, "cuda:0")
