"""Planned single SDK-free seam check: CPU tensors, synthetic states/resources.

No App/NN/checkpoint load or physical loop; no old geometry fixture rerun. Extract
the actual nested replan, keeping its original shared reference body, and use
one small synthetic callback. This is interface evidence, not resource/physics
admission. Execute only after root review/authorization.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from evaluate_sfm import parse_sfm_deployment_args
from sfm_runtime import SFMFactory, prepare_sfm_arguments, sfm_runtime_metadata
from CrowdSim.humanoid.application import deployment, resources
from CrowdSim.humanoid.application.configuration import parse_amp_args
from CrowdSim.humanoid.training.reference import CausalReferenceAdapter, ReferenceAdapterConfig


def record(path):
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def nested_replan(path, namespace):
    """Execute only the actual replan body; counters belong to its closure."""
    source = Path(path).read_text()
    main = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    node = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "replan")
    body = textwrap.dedent("\n".join(source.splitlines()[node.lineno-1:node.end_lineno]))
    wrapper = ("def bind():\n    replan_count = 0\n    planner_forward_calls = 0\n"
               + textwrap.indent(body, "    ")
               + "\n    return replan, lambda: (replan_count, planner_forward_calls)\n")
    scope = dict(namespace)
    exec(compile(wrapper, str(path), "exec"), scope)
    return scope["bind"]()


class SyntheticWorld:
    """One shared four-body snapshot; no simulator/physical progression."""
    def __init__(self):
        self.num_envs, self.shared_world, self.device = 4, True, torch.device("cpu")
        self.states = torch.tensor([[-3., -3., .1, 0., 0.], [-1., -3., 0., .1, 0.],
                                    [1., -3., -.1, 0., 0.], [3., -3., 0., 0., 0.]])
        self.validation_scene = SimpleNamespace(bounds=torch.tensor([-10., -10., 10., 10.]),
            centers=torch.tensor([[0., 0., .5]]), sizes=torch.tensor([[.4, 2., 1.]]))
        self.planar_calls = self.input_calls = 0

    def planar(self):
        self.planar_calls += 1
        return self.states.clone()

    def state(self):
        return SimpleNamespace(root_pos=torch.cat((self.states[:, :2], torch.zeros(4, 1)), 1),
                               root_vel=torch.cat((self.states[:, 2:4], torch.zeros(4, 1)), 1))

    def planner_inputs(self, ids):
        self.input_calls += 1
        return {"states": self.states[ids].clone()}


class DeterministicPlannerStub:
    """Default-DP body comparison only, never a network or Diffusion sampler."""
    def plan(self, *, states, **_kwargs):
        time = torch.arange(51, dtype=states.dtype) / 10
        xy = states[:, None, :2] + time[None, :, None] * states[:, None, 2:4]
        return SimpleNamespace(xy=xy, selected_sample=torch.zeros(len(states), dtype=torch.long))


def replan_environment(args, world, target=None):
    routes = world.states[:, None, :2].repeat(1, 10, 1)
    routes[:, :, 0] += torch.arange(1, 11)[None] * .2
    references = torch.zeros(4, 151, 2)
    return dict(torch=torch, args=args, world=world, target_generator=target, curriculum=None,
        native_scene=None, control_step=17, new_goals=lambda _ids: None,
        route_waypoints=routes, goals=routes[:, -1].clone(), waypoint_times=torch.arange(1, 11) / 2,
        guidance_weight=20., planner=DeterministicPlannerStub(), plans=references,
        raw_plans=torch.zeros(4, 51, 2), reference_velocities=torch.zeros_like(references),
        reference_initialized=torch.zeros(4, dtype=torch.bool), plan_age=torch.full((4,), 3, dtype=torch.long),
        reference_adapter=CausalReferenceAdapter(ReferenceAdapterConfig(max_speed_mps=1.5,
            max_acceleration_mps2=2., initial_speed_policy="brake")),
        reference_plan_records={key: [] for key in ("control_step", "actor_ids", "selected_sample", "raw_xy",
            "reference_xy", "reference_velocity_xy", "initial_overspeed", "speed_violation", "deviation_m")})


def main():
    checks = []
    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    sources = [record(HERE / name) for name in ("sfm_runtime.py", "evaluate_sfm.py", "evaluate_runner.py")]
    parent = ROOT / "scripts/train_paper_amp.py"
    # Independent exact inverse using standard patch, on a disposable copy only.
    with tempfile.TemporaryDirectory(prefix="sfm_seam_source_") as temporary:
        restored = Path(temporary) / "runner.py"
        restored.write_bytes((HERE / "evaluate_runner.py").read_bytes())
        process = subprocess.run(["patch", "--reverse", "--fuzz=0", "--silent", str(restored),
                                  "--input", str(HERE / "runner_exact.patch")], capture_output=True, text=True)
        check("exact_whole_runner_inverse_bytes_and_AST", process.returncode == 0
              and restored.read_bytes() == parent.read_bytes()
              and ast.dump(ast.parse(restored.read_text())) == ast.dump(ast.parse(parent.read_text())))

    args = parse_sfm_deployment_args(["--amp-checkpoint", "/tmp/synthetic_named_actor.ckpt",
        "--config", "/tmp/synthetic_profile.yaml", "--manifest", "/tmp/synthetic_manifest.json",
        "--output", "/tmp/sfm_seam_unused_output", "--duration-seconds", "200", "--inference-only"])
    parent_args = parse_amp_args(deployment.driver_arguments(args))
    sfm_args = prepare_sfm_arguments(deepcopy(parent_args))
    check("SFM_CLI_needs_no_DP_file_actual_source_and_same_clocks", sfm_args.trajectory_source == "sfm"
          and sfm_args.diffusion_checkpoint is None and sfm_args.diffusion_samples == 0
          and sfm_args.replan_steps == 15 and sfm_args.synchronized_replanning
          and sfm_args.evaluate_steps == 6000 and sfm_args.num_envs == 4)
    factory = SFMFactory()
    spec = importlib.util.spec_from_file_location("CPU_only_SFM_guard_check", HERE / "evaluate_runner.py")
    guard_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard_module)
    guard_module.parse_args = lambda: deepcopy(parent_args)
    try:
        guard_module.main(target_generator_factory=factory)
    except ValueError as error:
        check("external_factory_without_saved_evaluation_actor_rejected_before_App",
              "evaluation-only" in str(error))
    else:
        check("external_factory_without_saved_evaluation_actor_rejected_before_App", False)
    inspected = []
    def synthetic_resource(name, path, **_kwargs):
        if name == "diffusion_checkpoint":
            raise AssertionError("Unused DP resource must never be inspected")
        inspected.append(name)
        return resources.ResourceRecord(name, str(path), 128, 0, "synthetic_boundary_stub")
    # Header/weight/manifest stubs only; real original rig XML and fixture YAML are
    # parsed. This does not claim model/data/scene resources passed an actual gate.
    with patch.object(resources, "inspect_resource", synthetic_resource), \
         patch.object(resources, "_manifest_resources", lambda _path: ([], {"train": 1, "validation": 1, "test": 1})):
        factory.preflight(deepcopy(sfm_args), {"training": {}})
    check("only_unused_DP_resource_removed_all_original_roles_kept", inspected == [
        "profile", "manifest", "controller_checkpoint", "smpl_physics_rig", "shared_scene"])

    world = SyntheticWorld()
    runtime = factory(world=world, args=deepcopy(sfm_args), native_scene=None)
    scene_map = runtime.generator.scene_maps[0]
    check("flat_factory_uses_same_raw_cuboid_footprint", not scene_map.known_free[100, 100]
          and scene_map.known_free[130, 70] and scene_map.resolution_m == .1)
    labels = np.array([[254, 255, 0], [254, 205, 254], [254, 254, 254]], dtype=np.uint8)
    native_proxy = SimpleNamespace(provider=SimpleNamespace(_map=labels,
        _map_origin_xy=np.array([-.15, -.15])), resolution=.1)
    native_runtime = factory(world=world, args=deepcopy(sfm_args), native_scene=native_proxy)
    check("native_factory_raw254_only_not_inflated_or_unknown", np.array_equal(
        native_runtime.generator.scene_maps[0].known_free, labels == 254))

    sfm_args.planner_batch_size = 1
    environment = replan_environment(sfm_args, world, runtime)
    original_states, original_routes = world.states.clone(), environment["route_waypoints"].clone()
    replan, counters = nested_replan(HERE / "evaluate_runner.py", environment)
    selected = torch.tensor([2, 0], dtype=torch.long)
    replan(selected)
    check("one_full_scene_snapshot_prediction_then_two_selected_chunks_zero_DP", world.planar_calls == 1
          and world.input_calls == 0 and counters() == (2, 0)
          and runtime.describe()["full_scene_prediction_calls"] == 1
          and runtime.describe()["predicted_actor_plans"] == 4
          and len(environment["reference_plan_records"]["actor_ids"]) == 2)
    check("selected_ids_only_original_reference_body_and_inputs_preserved",
          torch.equal(environment["reference_initialized"], torch.tensor([True, False, True, False]))
          and torch.equal(environment["plan_age"], torch.tensor([0, 3, 0, 3]))
          and torch.equal(world.states, original_states) and torch.equal(environment["route_waypoints"], original_routes)
          and all(np.all(value == 0) for value in environment["reference_plan_records"]["selected_sample"]))

    parent_args.planner_batch_size = 1
    control_environments = [replan_environment(deepcopy(parent_args), SyntheticWorld()) for _ in range(2)]
    counts = []
    for path, scope in zip((parent, HERE / "evaluate_runner.py"), control_environments):
        callback, count = nested_replan(path, scope)
        callback(selected)
        counts.append(count())
    check("factory_None_DP_branch_exact_CPU_outputs_and_counts", counts == [(2, 2), (2, 2)]
          and all(torch.equal(control_environments[0][key], control_environments[1][key]) for key in (
              "plans", "raw_plans", "reference_velocities", "reference_initialized", "plan_age"))
          and all(all(np.array_equal(a, b) for a, b in zip(control_environments[0]["reference_plan_records"][key],
              control_environments[1]["reference_plan_records"][key])) for key in environment["reference_plan_records"]))
    historical = {"trajectory_source": "mixed_amass_procedural", "diffusion_checkpoint": "historical_OLD",
                  "diffusion_sha256": "0" * 64, "physics": {"control_hz": 30}, "complete_fixture_tag": 1}
    history_before = deepcopy(historical)
    report = {"runtime_execution": {}, "navigation_guidance": {"waypoint_times_s": [.5, 1., 5.]}}
    sfm_runtime_metadata(report, runtime, historical)
    check("actual_SFM_zero_DP_metadata_preserves_whole_historical_AMP_provenance",
          historical == history_before and report["trajectory_source"] == "sfm"
          and report["runtime_execution"]["diffusion_forward_calls"] == 0
          and report["runtime_execution"]["diffusion_model_constructions"] == 0
          and report["runtime_execution"]["diffusion_samples_per_actor"] == 0
          and "evaluation_planner_change" not in report
          and report["evaluation_target_generator_change"]["historical_training_diffusion_sha256"] == "0" * 64)

    forbidden = sorted(name for name in sys.modules if name.split(".")[0] in {"isaaclab", "isaacsim", "omni", "pxr"})
    passed = all(item["passed"] for item in checks) and not forbidden and not torch.cuda.is_initialized()
    result = {"status": "passed" if passed else "failed", "scope": "synthetic_CPU_seam_contract_only",
        "checks": checks, "sources": sources, "parent": record(parent), "checker": record(__file__),
        "real_weights_or_datasets_loaded": False, "NN_constructed": False, "SDK_modules_loaded": forbidden,
        "CUDA_initialized": torch.cuda.is_initialized(), "python": sys.version, "executable": sys.executable,
        "SFM_prediction_calls": runtime.describe()["full_scene_prediction_calls"], "replan_counts": list(counters()),
        "synthetic_resource_roles": inspected, "actual_physical_quality_proven": False}
    with (HERE / "actual_CPU_seam.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "failed": [item["name"] for item in checks if not item["passed"]]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
