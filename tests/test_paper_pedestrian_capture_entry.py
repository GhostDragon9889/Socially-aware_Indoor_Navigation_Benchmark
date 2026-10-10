"""Pure capture interfaces; fake delegates are not physical or sensor evidence."""

from contextlib import contextmanager
import json
import os
import subprocess
import sys
from types import ModuleType

import pytest

from CrowdSim.humanoid.application import capture, deployment


def cli(tmp_path):
    diffusion = tmp_path / "step_5000.ckpt"
    diffusion.write_bytes(b"parser file-presence fixture, not model weights")
    return ["--amp-checkpoint", str(tmp_path / "step_1000.pt"), "--config", "profile.yaml",
            "--manifest", "manifest.json", "--diffusion-checkpoint", str(diffusion),
            "--duration-seconds", "200", "--output", str(tmp_path / "new_capture")]


def test_capture_changes_only_render_flags_and_interval(tmp_path):
    args, request = capture.parse_capture_args(cli(tmp_path))
    original = vars(deployment.parse_amp_args(deployment.driver_arguments(args)))
    actual = vars(deployment.parse_amp_args(capture.capture_driver_arguments(args, request)))
    assert {key for key in original if original[key] != actual[key]} == {
        "physics_only", "capture", "capture_every_steps"}
    assert actual["capture"] is True and actual["physics_only"] is False
    assert actual["capture_every_steps"] == 2 and actual["evaluate_steps"] == 6000
    assert (actual["guidance_weight"], actual["agent_collision_weight"],
            actual["scene_collision_weight"], actual["near_horizon_waypoint_mass"]) == (20., 5., 10., 0.)


@pytest.mark.parametrize("extra", [["--capture-every-steps", "0"], ["--capture-every-steps", "31"],
    ["--camera-eye", "1", "2", "3"],
    ["--camera-eye", "1", "2", "3", "--camera-target", "1", "2", "3"],
    ["--camera-eye", "nan", "2", "3", "--camera-target", "0", "0", "0"],
    ["--native-dynbench", "--inference-only"]])
def test_bad_capture_interval_or_camera_fails_before_resources(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        capture.parse_capture_args(cli(tmp_path) + extra)
    assert error.value.code == 2 and not (tmp_path / "new_capture").exists()


def test_visual_preflight_restores_override_when_resource_check_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CROWDSIM_SMPL_MODEL_DIR", "/previous/model")
    monkeypatch.delenv("CROWDSIM_SMPL_TEXTURE_DIR", raising=False)
    monkeypatch.setattr(deployment, "preflight_deployment", lambda args: {})
    monkeypatch.setattr(capture, "load_amp_profile", lambda path: {})
    def fail(args, profile):
        assert args.capture and os.environ["CROWDSIM_SMPL_MODEL_DIR"] == str(deployment.ROOT / "data/smpl")
        assert os.environ["CROWDSIM_SMPL_TEXTURE_DIR"] == str(deployment.ROOT / "data/smpl/textures")
        raise ValueError("injected resource failure")
    monkeypatch.setattr(capture, "preflight_amp_resources", fail)
    with pytest.raises(ValueError, match="injected resource failure"):
        capture.preflight_capture(*capture.parse_capture_args(cli(tmp_path)))
    assert os.environ["CROWDSIM_SMPL_MODEL_DIR"] == "/previous/model"
    assert "CROWDSIM_SMPL_TEXTURE_DIR" not in os.environ


@pytest.mark.parametrize("raises", [False, True])
def test_one_delegate_keeps_factories_and_restores_env_and_argv(tmp_path, monkeypatch, raises):
    actor, scene, calls = object(), object(), []
    monkeypatch.setenv("CROWDSIM_SMPL_MODEL_DIR", "/previous/model")
    monkeypatch.setenv("CROWDSIM_SMPL_TEXTURE_DIR", "/previous/textures")
    request = {"driver_argv": ["--capture"], "evaluation_actor_request": {},
               "native_scene_request": {}, "physical_execution_verified": False}
    monkeypatch.setattr(capture, "preflight_capture", lambda *args: request)
    monkeypatch.setattr(deployment, "make_evaluation_actor_factory", lambda request: actor)
    helper, native = ModuleType("capture_rgbd"), ModuleType("native_deployment")
    native.make_native_scene_factory = lambda request: scene
    @contextmanager
    def helpers(output, **kwargs):
        assert kwargs["interval"] == 2 and kwargs["depth"] is True
        yield
    helper.capture_helpers = helpers
    monkeypatch.setitem(sys.modules, "CrowdSim.humanoid.application.capture_rgbd", helper)
    monkeypatch.setitem(sys.modules, "CrowdSim.humanoid.application.native_deployment", native)
    original = sys.argv
    def delegate(**kwargs):
        calls.append(kwargs)
        assert kwargs == {"evaluation_actor_factory": actor, "native_scene_factory": scene}
        assert sys.argv[1:] == request["driver_argv"]
        assert os.environ["CROWDSIM_SMPL_MODEL_DIR"] == str(deployment.ROOT / "data/smpl")
        if raises:
            raise RuntimeError("injected ordinary Python failure; not SDK fast-exit")
        return 7
    if raises:
        with pytest.raises(RuntimeError, match="injected ordinary Python failure"):
            capture.main(cli(tmp_path) + ["--depth"], driver_main=delegate)
    else:
        assert capture.main(cli(tmp_path) + ["--depth"], driver_main=delegate) == 7
    assert len(calls) == 1 and sys.argv is original
    assert os.environ["CROWDSIM_SMPL_MODEL_DIR"] == "/previous/model"
    assert os.environ["CROWDSIM_SMPL_TEXTURE_DIR"] == "/previous/textures"
    assert json.loads((tmp_path / "new_capture/deployment_entry.json").read_text())["physical_execution_verified"] is False


def test_cli_help_imports_no_tensor_or_sdk():
    code = """import json, runpy, sys
sys.argv = ['scripts/capture_paper_pedestrians.py', '--help']
try: runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error: assert error.code == 0
print(json.dumps([n for n in sys.modules if n in {'torch', 'numpy'} or n.startswith(('omni.', 'isaaclab', 'isaacsim'))]))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=deployment.ROOT,
                            check=True, text=True, capture_output=True)
    assert "--depth" in result.stdout and "--camera-eye" in result.stdout
    assert json.loads(result.stdout.splitlines()[-1]) == []
