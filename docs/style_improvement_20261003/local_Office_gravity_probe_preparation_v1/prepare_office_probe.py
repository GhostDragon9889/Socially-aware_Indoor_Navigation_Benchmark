"""Prepare a local Office3/10s observation plan on CPU, without future outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
SDK = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"
OFFICE = ROOT / "docs/style_improvement_20261002/native_map_clock_v9_actual_reader_preparation/configurations3_to15_v1/Office_actors3.yaml"
NATIVE = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_native_preparation_v1"


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular current source/resource required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare():
    spec = importlib.util.spec_from_file_location("Office_registered_native_sources", NATIVE / "validate_native_admission.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    registration = native.read_registration()
    from CrowdSim.humanoid.application.deployment import parse_deployment_args, driver_arguments
    from CrowdSim.humanoid.application.native_deployment import preflight_native_scene
    output = ROOT / "output/diffusion_amp/local_Office3_pacer_reward_gravity10s_v1"
    argv = ["--amp-checkpoint", registration["future_checkpoint"]["path"],
            "--config", str(native.PROFILE),
            "--diffusion-checkpoint", registration["sources"]["diffusion"]["path"],
            "--manifest", registration["sources"]["manifest"]["path"],
            "--scene-config", str(OFFICE), "--actors", "3", "--duration-seconds", "10",
            "--seed", "7", "--inference-only", "--native-dynbench", "--arm-style-alpha", "0",
            "--output", str(output)]
    args = parse_deployment_args(argv)
    request = preflight_native_scene(OFFICE, actors=3)
    current_map = json.loads(Path(request["native_map"]["path"]).read_text())
    source = current_map["source"]
    sdk_root = Path(SDK).parents[2]
    sources = {
        "observation_adapter": record(HERE / "observe_configured_gravity.py"),
        "preparer": record(__file__), "scene_configuration": record(OFFICE),
        "native_admission_core": record(NATIVE / "validate_native_admission.py"),
        "native_registration": record(NATIVE / "registration.json"),
        "deployment_entry": record(ROOT / "CrowdSim/humanoid/application/deployment.py"),
        "native_factory": record(ROOT / "CrowdSim/humanoid/application/native_deployment.py"),
        "existing_live_verifier": record(ROOT / "CrowdSim/humanoid/runtime/dynbench_live.py"),
        "existing_world": record(ROOT / "CrowdSim/humanoid/amp_world.py"),
        "SDK_simulation_cfg": record(sdk_root / "source/isaaclab/isaaclab/sim/simulation_cfg.py"),
        "SDK_simulation_context": record(sdk_root / "source/isaaclab/isaaclab/sim/simulation_context.py"),
        "Office_source_USD": record(source["path"]),
    }
    forbidden = any(name == "torch" or name == "isaacsim" or name == "isaaclab"
                    or name.startswith(("omni.", "isaacsim.", "isaaclab.")) for name in sys.modules)
    if forbidden:
        raise RuntimeError("CPU preparation unexpectedly imported tensor/SDK runtime")
    return {
        "schema": "local_Office.prepared_gravity_support_probe.v1",
        "status": "prepared_source_bound_not_executed", "GPU_used": False, "SDK_started": False,
        "future_checkpoint_read": False, "future_capture_read": False, "sealed34_outputs_read": False,
        "actual_local_Office_human_probe_executed": False, "actual_human_stability_verified": False,
        "author_eight_scene_equivalence": False, "production_source_changed": False,
        "sources": sources, "production_source": registration["current_source_binding"]["production_source"],
        "future_checkpoint": dict(registration["future_checkpoint"]),
        "existing_native_scene_request": request,
        "scene_resource_evidence_limits": {
            "scene": "Office", "map_status": current_map["status"],
            "V9_map_query_gravity_m_s2": current_map["physics"]["gravity"],
            "V9_map_snapshot_steps": current_map["physics"]["snapshot_steps"],
            "V9_physics_metadata": current_map["physics"],
            "reference_ground_recipe": current_map["reference_ground"]["recipe"],
            "reference_ground_overlay": current_map["reference_ground"]["overlay_source"],
            "map_query_is_dynamic_human_stability_evidence": False,
            "floor": "engineering source-bound reference ground; original flat ground collision disabled in session",
            "authored_Office_floor_is_claimed": False,
        },
        "future_deployment_command_without_observer": [SDK, str(ROOT / "scripts/simulate_paper_pedestrians.py"), *argv],
        "future_driver_argv": driver_arguments(args),
        "observation_adapter_integration": {
            "interface": "wrap_native_scene_factory(existing_native_factory)",
            "call_site": "sole driver native_scene_factory keyword before existing AMPWorld construction",
            "callback": "preserve provider.verify_live_runtime; append configured gravity after original read-only floor/source verification",
            "requires_root_owned_future_driver_adapter": True,
            "listed_deployment_command_alone_does_not_attach_observer": True,
            "creates_App_context_views_or_steps": False,
            "changes_USD_or_policy_actions": False,
            "only_mutation": "one newly created provider instance's Python observation callback",
        },
        "expected_counts": {
            "actors": 3, "duration_s": 10.0, "controls": 300,
            "shared_physics_substeps": 1200, "actor_controls": 900,
            "actor_physics_substeps": 3600, "foot_substep_observations": 14400,
            "foot_force_array_shape": [300, 4, 3, 4, 1, 3], "all_failed_controls_retained": True,
        },
        "future_raw_evaluation_interface": {
            "inputs": ["deployment_entry.json", "result.json", "physical_states.npz", "ground_support.npz", "reference_plans.npz", "actual root closed-launch receipt"],
            "gravity": "result.native_scene_live_observation.local_configured_gravity_observation",
            "measurement": "existing pre-reset native static/peer contacts + four120Hz completed substep feet-floor loads and true COM",
            "report": "retain falls, resets, negative heights, contact counts, feet support/loads/airborne durations for all300controls/all3actors",
            "configured_gravity_is_native_engine_measurement": False,
            "existing_4_or_15_actor_200s_registered_reader_accepts_Office3": False,
            "future_Office3_short_capture_reader_adapter_required": True,
            "no_style_score_filter_or_default_promotion": True,
        },
        "execution_order": "Root may schedule only after the current Warehouse comparison; this plan grants no concurrent GPU launch",
        "limitations": [
            "The standalone observer and its CPU tests have not observed an actual Office world.",
            "Configured singleton/USD gravity agrees only when the future callback actually executes; it does not observe native-engine gravity or timestamp.",
            "A ten-second local probe diagnoses source/floor/gravity/foot/contact support and cannot establish author benchmark equivalence or200second stability.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = prepare()
    with args.output.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": document["status"], "plan": record(args.output),
                      "GPU_used": False, "SDK_started": False, "future_checkpoint_read": False}))


if __name__ == "__main__":
    main()
