"""CPU checks for captured pixel values and reusable camera buffers."""

from types import SimpleNamespace
import warnings

import numpy as np
from PIL import Image
import pytest
import torch

from CrowdSim.tools import test_camera_depth
from CrowdSim.utils.sensor_stream import RobotCameraRecorder


class ExplicitCameraProxy:
    """Exercise the SDK's explicit Torch view without its deprecated bridge."""

    def __init__(self, tensor):
        self.torch = tensor


def camera_env(depth, rgb=None, wrapper=lambda value: value):
    callbacks = {}
    output = {"distance_to_image_plane": wrapper(depth)}
    if rgb is not None:
        output["rgb"] = wrapper(rgb)
    env = SimpleNamespace(
        num_envs=1, device=torch.device("cpu"),
        robot_config=SimpleNamespace(number_of_actions=1),
        crowdsim_robot_camera=SimpleNamespace(data=SimpleNamespace(output=output)),
        simulator=SimpleNamespace(
            _sim=SimpleNamespace(render=lambda: None),
            keyboard_interface=SimpleNamespace(add_callback=lambda key, fn: callbacks.setdefault(key, fn)),
        ),
        step=lambda action: None,
    )
    return env, callbacks


@pytest.mark.parametrize("wrapper", [lambda value: value, ExplicitCameraProxy], ids=["tensor", "explicit_proxy"])
@pytest.mark.parametrize("dtype,pixel,expected", [
    (torch.uint8, 1, 1), (torch.uint8, 0, 0), (torch.uint8, 128, 128),
    (torch.float32, .5, 127), (torch.float32, 128., 128),
])
def test_capture_respects_rgb_dtype(monkeypatch, tmp_path, wrapper, dtype, pixel, expected):
    monkeypatch.chdir(tmp_path)
    env, callbacks = camera_env(torch.ones(1, 2, 2, 1),
                                torch.full((1, 2, 2, 3), pixel, dtype=dtype), wrapper)
    test_camera_depth.add_camera_check_key(env)
    callbacks["C"]()
    with Image.open(tmp_path / "output/renderings/camera_check/rgb_env0.png") as frame:
        assert frame.getpixel((0, 0)) == (expected,) * 3


@pytest.mark.parametrize("wrapper", [lambda value: value, ExplicitCameraProxy], ids=["tensor", "explicit_proxy"])
def test_burst_freezes_reused_cpu_camera_buffer(monkeypatch, tmp_path, wrapper):
    monkeypatch.chdir(tmp_path)
    depth = torch.full((1, 2, 2, 1), 2.)
    env, _ = camera_env(depth, wrapper=wrapper)
    env.step = lambda action: depth.add_(1.)
    test_camera_depth.record_headless_camera_burst(env, frames=2, save_envs=1)
    directory = tmp_path / "output/renderings/camera_headless"
    for index, expected_depth in enumerate((2., 3.)):
        with Image.open(directory / f"env00_frame{index:04d}.png") as frame:
            assert frame.getpixel((0, 0)) == int(expected_depth / 20. * 255)
        np.testing.assert_array_equal(np.load(directory / f"env00_frame{index:04d}.npy"),
                                      np.full((2, 2), expected_depth))


@pytest.mark.parametrize("wrapper", [lambda value: value, ExplicitCameraProxy], ids=["tensor", "explicit_proxy"])
def test_burst_no_hits_has_clear_statistics_and_preserves_sentinels(monkeypatch, tmp_path, capsys, wrapper):
    monkeypatch.chdir(tmp_path)
    depth = torch.tensor([[[[float("inf")], [float("nan")]], [[0.], [-1.]]]])
    env, _ = camera_env(depth, wrapper=wrapper)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test_camera_depth.record_headless_camera_burst(env, frames=2, save_envs=1)
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert "no valid depth hits" in capsys.readouterr().out
    for index in range(2):
        path = tmp_path / f"output/renderings/camera_headless/env00_frame{index:04d}.npy"
        np.testing.assert_array_equal(np.load(path), depth[0, ..., 0].numpy())


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), float("-inf"), 0., -1.])
def test_recorder_rejects_nonfinite_or_nonpositive_fps(tmp_path, fps):
    with pytest.raises(ValueError, match="fps must be"):
        RobotCameraRecorder(SimpleNamespace(), tmp_path, fps, [0])
