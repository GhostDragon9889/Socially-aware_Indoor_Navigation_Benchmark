"""Pure entry contracts; injected driver calls are not a physics test."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from CrowdSim.humanoid.application import deployment


GOLDEN = Path(__file__).parent / "fixtures/paper_deployment_navigation_arguments.json"


def cli(tmp_path, *, duration="200", output="new_run"):
    diffusion = tmp_path / "step_5000.ckpt"
    diffusion.write_bytes(b"file-presence parser contract only; not weights")
    return ["--amp-checkpoint", str(tmp_path / "step_1000.pt"),
            "--config", str(tmp_path / "profile.yaml"),
            "--diffusion-checkpoint", str(diffusion), "--manifest", str(tmp_path / "manifest.json"),
            "--duration-seconds", duration, "--output", str(tmp_path / output)]


@pytest.mark.parametrize("duration,steps", [("200", 6000), ("30.0", 900), (".1", 3),
                                           ("0.033333333333333333", 1)])
def test_duration_is_whole_30hz_controls(duration, steps):
    assert deployment.duration_control_steps(duration) == steps


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "Infinity", "0.04", "no", "1e-20"])
def test_invalid_duration_cannot_silently_round_the_physical_budget(duration):
    with pytest.raises(ValueError):
        deployment.duration_control_steps(duration)


def test_forwarded_semantics_match_actual_completed_baseline_for_every_existing_field(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    args = deployment.parse_deployment_args(cli(tmp_path))
    actual = vars(deployment.parse_amp_args(deployment.driver_arguments(args)))
    golden = json.loads(GOLDEN.read_text())["arguments"]
    assert len(golden) == 56
    assert set(actual) == set(golden) | {"near_horizon_waypoint_mass"}
    # Only explicitly user-supplied file/output paths differ; normalize their
    # known names, preserving every other historical argument without filtering.
    for name in ("config", "manifest", "diffusion_checkpoint", "output", "resume", "scene_config"):
        assert Path(actual[name]).is_absolute()
        actual[name] = golden[name]
    assert actual.pop("near_horizon_waypoint_mass") == 0.0  # Original mean guidance, not ETA.
    assert actual == golden


def test_shorter_duration_changes_only_control_count_and_nonterminal_cap(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    long = deployment.parse_deployment_args(cli(tmp_path))
    short = deployment.parse_deployment_args(cli(tmp_path, duration="30"))
    left = vars(deployment.parse_amp_args(deployment.driver_arguments(long)))
    right = vars(deployment.parse_amp_args(deployment.driver_arguments(short)))
    assert {name for name in left if left[name] != right[name]} == {"episode_steps", "evaluate_steps"}
    assert right["evaluate_steps"] == 900 and right["episode_steps"] == 901


@pytest.mark.parametrize("extra", [["--updates", "2000"], ["--capture"],
                                   ["--trajectory-source", "amass_root"], ["--resume", "last.pt"]])
def test_entry_rejects_unused_training_render_or_fallback_flags(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        deployment.parse_deployment_args(cli(tmp_path) + extra)
    assert error.value.code == 2


def test_existing_output_even_empty_and_mutable_controller_are_rejected_before_sdk(tmp_path):
    existing = tmp_path / "new_run"
    existing.mkdir()
    with pytest.raises(SystemExit):
        deployment.parse_deployment_args(cli(tmp_path))
    args = cli(tmp_path, output="other")
    args[1] = str(tmp_path / "last.pt")
    with pytest.raises(SystemExit):
        deployment.parse_deployment_args(args)


def test_fixture_actor_count_is_not_misrepresented_as_arbitrary_scene_support(tmp_path):
    with pytest.raises(SystemExit):
        deployment.parse_deployment_args(cli(tmp_path) + ["--actors", "8"])


class FakeResources:
    def as_dict(self):
        return {"status": "injected CPU resource contract; not SDK validation"}


def mock_resource_chain(monkeypatch):
    seen = []

    def profile(path):
        seen.append(("profile", path))
        return {"mode": "paper_method_reconstruction"}

    def resources(args, document):
        assert document["mode"] == "paper_method_reconstruction"
        seen.append(("resources", args))
        return FakeResources()

    monkeypatch.setattr(deployment, "load_amp_profile", profile)
    monkeypatch.setattr(deployment, "preflight_amp_resources", resources)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    return seen


def test_preflight_only_runs_existing_parser_resource_chain_without_output_or_driver(tmp_path, monkeypatch, capsys):
    seen = mock_resource_chain(monkeypatch)

    def forbidden():
        pytest.fail("Preflight must not invoke the physical driver")

    assert deployment.main(cli(tmp_path) + ["--preflight-only"], driver_main=forbidden) == 0
    report = json.loads(capsys.readouterr().out)
    assert [row[0] for row in seen] == ["profile", "resources"]
    assert report["SDK_started"] is False and report["physical_execution_verified"] is False
    assert report["deployment_contract"]["CPU_threads"] == 4
    assert not (tmp_path / "new_run").exists()
    assert "CUBLAS_WORKSPACE_CONFIG" not in deployment.os.environ  # Preflight remains read-only.


def test_nonfixture_scene_fails_before_driver_even_if_generic_resource_shapes_pass(tmp_path, monkeypatch):
    mock_resource_chain(monkeypatch)
    other = tmp_path / "dynbench.yaml"
    other.write_text("schema: navisaaclab.reconstruction_scene.v1\n")
    args = deployment.parse_deployment_args(cli(tmp_path) + ["--scene-config", str(other)])
    with pytest.raises(ValueError, match="DynBench"):
        deployment.preflight_deployment(args)


def test_dispatch_calls_one_existing_driver_and_restores_argv_on_success_or_failure(tmp_path, monkeypatch):
    mock_resource_chain(monkeypatch)
    original = sys.argv
    calls = []

    def driver():
        calls.append(deepcopy(sys.argv))
        assert sys.argv[0] == str(deployment.DRIVER)
        assert "--evaluate-steps" in sys.argv and "--physics-only" in sys.argv
        return None

    assert deployment.main(cli(tmp_path), driver_main=driver) == 0
    assert len(calls) == 1 and sys.argv is original
    report = json.loads((tmp_path / "new_run/deployment_entry.json").read_text())
    assert calls[0][1:] == report["driver_argv"]
    assert report["physical_execution_verified"] is False
    for name in deployment.THREAD_ENVIRONMENT:
        assert deployment.os.environ[name] == "4"

    def failure():
        raise RuntimeError("original driver failed")

    with pytest.raises(RuntimeError, match="original driver failed"):
        deployment.invoke_existing_driver(["--evaluate-steps", "1"], failure)
    assert sys.argv is original


def test_existing_driver_load_is_lazy_and_imports_no_sdk():
    code = """
import json, sys
from CrowdSim.humanoid.application.deployment import DRIVER, invoke_existing_driver
import importlib.util
spec = importlib.util.spec_from_file_location('pure_original_driver_import', DRIVER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
forbidden = [n for n in sys.modules if n == 'torch' or n == 'numpy' or n.startswith(('omni.', 'isaaclab', 'isaacsim'))]
print(json.dumps(forbidden))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=deployment.ROOT,
                            check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == []


def test_bad_determinism_environment_is_rejected_by_the_existing_pure_parser(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "unsupported")
    args = deployment.parse_deployment_args(cli(tmp_path))
    with pytest.raises(SystemExit):
        deployment.preflight_deployment(args)
    assert not args.output.exists()


def test_existing_sdk_process_cannot_create_a_second_app_and_argv_is_restored(monkeypatch):
    monkeypatch.setitem(sys.modules, "omni.kit.app", object())
    original = sys.argv
    with pytest.raises(RuntimeError, match="fresh Python process"):
        deployment.invoke_existing_driver(["--evaluate-steps", "1"])
    assert sys.argv is original


@pytest.mark.parametrize("extra", [["--arm-style-alpha", "nan"], ["--arm-style-alpha", "1.01"],
                                   ["--arm-style-alpha", "-0.1"], ["--arm-style-alpha", ".25"],
                                   ["--experimental-arm-style-checkpoint", "step_2000.pt"],
                                   ["--inference-only", "--experimental-arm-style-checkpoint", "last.pt"]])
def test_experimental_style_cannot_bypass_explicit_inference_or_artifact_contract(tmp_path, extra):
    with pytest.raises(SystemExit):
        deployment.parse_deployment_args(cli(tmp_path) + extra)


def test_inference_shim_still_calls_the_single_driver_once_and_restores_argv():
    original = sys.argv
    request = {"base_checkpoint": {"path": "/not_loaded/step_1000.pt", "sha256": "a"*64},
               "style_checkpoint": None, "alpha": 0.}
    calls = []
    def fake_driver(*, evaluation_actor_factory):
        calls.append((list(sys.argv), evaluation_actor_factory))
        return 5
    assert deployment.invoke_existing_driver(["--evaluate-steps", "900"], fake_driver,
                                            evaluation_actor_request=request) == 5
    assert len(calls) == 1 and callable(calls[0][1])
    assert calls[0][0][1:] == ["--evaluate-steps", "900"]
    assert sys.argv is original


@pytest.mark.parametrize("steps,resume", [(0, None), (0, "step_1000.pt"), (900, None)])
def test_driver_rejects_pure_actor_training_or_missing_checkpoint_before_sdk(steps, resume):
    import runpy
    from types import SimpleNamespace
    module = runpy.run_path(str(deployment.DRIVER), run_name="pure_driver_CPU_guard")
    main = module["main"]
    main.__globals__["parse_args"] = lambda: SimpleNamespace(evaluate_steps=steps, resume=resume)
    with pytest.raises(ValueError, match="saved controller and explicit evaluation"):
        main(evaluation_actor_factory=lambda *a, **kw: None)


@pytest.mark.parametrize("actors", [0, 16])
def test_native_actor_count_requires_the_measured_source_contract(tmp_path, actors):
    with pytest.raises(SystemExit):
        deployment.parse_deployment_args(cli(tmp_path) + ["--native-dynbench", "--inference-only", "--actors", str(actors)])


def test_native_runtime_verification_precedes_the_first_physical_reset():
    import ast
    source = (deployment.ROOT / "CrowdSim/humanoid/amp_world.py").read_text()
    world = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "AMPWorld")
    init = next(node for node in world.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    calls = {name: [node for node in ast.walk(init) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == name]
             for name in ("bind_native_query", "verify_live_runtime", "reset")}
    assert all(len(rows) == 1 for rows in calls.values())
    assert calls["bind_native_query"][0].lineno < calls["verify_live_runtime"][0].lineno < calls["reset"][0].lineno
    arguments = {arg.arg: ast.unparse(arg.value) for arg in calls["verify_live_runtime"][0].keywords}
    assert arguments["origins_xy"] == "self.origins"
    assert arguments["simulation_context"] == "self.simulator._sim"
