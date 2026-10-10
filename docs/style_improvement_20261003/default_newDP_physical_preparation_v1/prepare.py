"""Freeze only two same-current185 fixture arrays plus native future path specifications."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from physical_evidence import CPU_ENV, GPU_ENV, HERE, PYTHON, ROOT, RUNS, SOURCE, read, record, require, source_binding, verify_epoch


def main():
    sys.path.insert(0, str(ROOT))
    from CrowdSim.humanoid.application.deployment import parse_deployment_args, preflight_deployment
    require(not (HERE / "registration.json").exists(), "Exclusive preparation; retain earlier failures")
    baseline = read(ROOT / "docs/style_improvement_20261003/local_Office_baseline_probe_preparation_v1/plan.json")
    controller = baseline["selected_checkpoint"]
    old = record(ROOT / "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt")
    new = record(ROOT / "output/diffusion_amp/social_diffusion_maps1000_fresh5000_v1/sealed_terminal/step_5000.ckpt")
    require(controller["sha256"] == "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
            and old["sha256"] == "63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac"
            and new["sha256"] == "b56346fbb0c23dc9656a28fa054b45178ef5922a0e52c66b18531b94e296da90",
            "Fixed pretrained controller and old/new DP identities required")
    train = ROOT / "docs/style_improvement_20261003/orca_maps_diffusion_actual_runs_v1"
    owner = read(train / "training_actual_launch.json")
    source_epoch = read(SOURCE)
    source = verify_epoch(record(SOURCE), record(ROOT / source_epoch["archive"]["path"]), 185, current=True)
    math = {"fixture_raw": record(ROOT / "docs/style_improvement_20261002/audit_ground_support_capture.py")}
    inputs = {"default_controller": controller, "old_DP": old, "new_DP": new, "default_profile": baseline["profile"],
              "manifest_v7": record(ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"),
              "shared_timed_scene": record(ROOT / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml"),
              "DP_closed_training": record(train / "training_actual_launch.json"),
              "DP_terminal_seal": record(train / "terminal5000_root_seal.json"),
              "DP_training_source_manifest": owner["selected_source_after"]["manifest"],
              "historical_fixture_raw_audit": record(ROOT / "docs/style_improvement_20261002/ground_support_200s_actual_verification.json"),
              "historical_fixture_result": record(ROOT / "output/diffusion_amp/shared_amass2848_backward_penalty_ground_support_200s_v1/result.json"),
              "historical_fixture_source_freeze": record(ROOT / "docs/style_improvement_20261002/ground_support_capture_source_freeze/source_manifest.json"),
              "control_source_inventory": record(ROOT / "docs/style_improvement_20261003/orca_DP_physical_source_bridge_v1/source_inventory.json"),
              "specific_control_source_derivation": record(ROOT / "docs/style_improvement_20261003/native185_physical_source_derivation_v1/source_derivation.json"),
              **math}
    historical = {}
    for actors in (4, 15):
        old_case = baseline["historical_baseline_native_stability"][str(actors)]
        historical[str(actors)] = {key: old_case[key] for key in ("audit", "result", "deployment_entry", "counts")}
        historical[str(actors)]["source_epoch"] = old_case["historical_source_epoch"]
        for key in ("audit", "result", "deployment_entry"):
            inputs[f"historical_native{actors}_{key}"] = old_case[key]
    cases = {}
    for label, planner in (("old", old), ("new", new)):
        phase = "fixture_" + label
        output = ROOT / ("output/diffusion_amp/default_" + ("oldDP" if label == "old" else "newDP") + "_fixture200s_v1")
        argv = ["--amp-checkpoint", controller["path"], "--config", baseline["profile"]["path"],
                "--diffusion-checkpoint", planner["path"], "--manifest", inputs["manifest_v7"]["path"],
                "--scene-config", inputs["shared_timed_scene"]["path"], "--actors", "4", "--duration-seconds", "200",
                "--seed", "7", "--inference-only", "--arm-style-alpha", "0", "--output", str(output)]
        preflight = json.loads(json.dumps(preflight_deployment(parse_deployment_args(argv)), default=str, allow_nan=False))
        require(preflight["SDK_started"] is False and preflight["Torch_imported_by_entry"] is False
                and preflight["control_steps"] == 6000 and preflight["evaluation_actor_request"]["style_checkpoint"] is None,
                "Actual SDK-free fixed fixture loader admission failed")
        audit = RUNS / (phase + "_audit.json")
        command = [PYTHON, str(ROOT / "scripts/simulate_paper_pedestrians.py"), *argv]
        cases[phase] = {"label": label, "actors": 4, "planner": planner, "output": str(output), "command": command,
                        "preflight": preflight, "audit": str(audit), "GPU_receipt": str(RUNS / (phase + "_actual_launch.json")),
                        "CPU_receipt": str(RUNS / (phase + "_CPU_actual_launch.json")),
                        "CPU_command": [PYTHON, str(HERE / "audit_physical.py"), "--phase", phase, "--output", str(audit)],
                        "prior_physical_gate": None if label == "old" else "Actual closed complete OLD fixture first; OLD failures do not exclude NEW"}
    before, after = (cases[key]["preflight"]["parsed_driver_arguments"].copy() for key in ("fixture_old", "fixture_new"))
    delta = {key: {"old": before[key], "new": after[key]} for key in before if before[key] != after[key]}
    require(set(delta) == {"diffusion_checkpoint", "output"}, "Only DP artifact and output identity may differ")
    native = {}
    for actors in (4, 15):
        config = ROOT / f"configs/diffusion_amp/native_warehouse_pedestrians_{'15_' if actors == 15 else ''}clearance08_v1.yaml"
        inputs[f"native{actors}_scene"] = record(config)
        native[str(actors)] = {"actors": actors, "duration_seconds": 200.0, "controls": 6000,
                              "scene_configuration": inputs[f"native{actors}_scene"], "source_epoch": source,
                              "ordered_labels": ["old", "new"], "planners": {"old": old, "new": new},
                              "future_outputs": {label: str(ROOT / f"output/diffusion_amp/native_warehouse_default_{'oldDP' if label == 'old' else 'newDP'}_{'15_' if actors == 15 else ''}clearance08_200s_v1") for label in ("old", "new")},
                              "gate_to_start_pair": "NEW full fixture zeroF/static/peer" if actors == 4 else "NEW full native4 zeroF/static/peer",
                              "OLD_failure_does_not_skip_NEW": True, "command_arrays_generated": False, "future_effect": False,
                              "same_registered_guide20_alpha0_seed7_causal_reference_and_controls": True}
    plan = {"schema": "default_newDP.fixed_physical_paired_registration.v1", "status": "prepared_only",
            "created_utc": datetime.now(timezone.utc).isoformat(), "controller": controller,
            "controller_original_mode_preserved": oct(Path(controller["path"]).stat().st_mode & 0o777),
            "controller_training_provenance": "Historical state-only default1000/8192000; not new935/09e",
            "planners": {"old": old, "new": new}, "production_source": source,
            "DP_training_source_is_distinct935": True, "inputs": inputs, "math": math,
            "historical_native": historical, "cases": cases, "strict_current185_pair_recipe_delta": delta,
            "root_actual_order": ["fixture_old", "fixture_new", "native4_old", "native4_new", "native15_old", "native15_new"],
            "native_future_path_specifications": native, "same_current185_runtime_for_each_OLD_NEW_pair": True,
            "historical_outputs_are_reference_not_strict_current_source_pairs": True,
            "no_whole168_174_185_source_equivalence_claim": True,
            "AMP_control_PD_contact_physics_and_DP_math_edited": False, "all_failed_contact_idle_reset_samples_retained": True,
            "physical_gate": "NEW complete6000/200s controls with falls==0/static_actor_controls==0/peer_pair_controls==0",
            "physical_gate_uses_MMD": False, "future_effect": False, "GPU_used": False, "SDK_started": False,
            "models_deserialized": False, "sealed34_outputs_read": False, "default_promotion": False}
    source_binding(plan)
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "Preparation imported GPU/model/SDK")
    with (HERE / "registration.json").open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    commands = {"schema": "default_newDP.prepared_physical_commands.v1", "cwd": str(ROOT), "CPU_env": CPU_ENV, "GPU_env": GPU_ENV,
                "root_only_GPU": [PYTHON, str(HERE / "launch_registered.py")],
                "ordered_fixture_commands": [{"phase": phase, "root_GPU_owner": [PYTHON, str(HERE / "launch_registered.py"), "--phase", phase],
                                               "CPU_raw_owner_after_root_closed": [PYTHON, str(HERE / "launch_registered.py"), "--phase", phase, "--cpu-audit"]}
                                              for phase in ("fixture_old", "fixture_new")],
                "native_future_arrays": None, "future_effect": False, "default_promotion": False}
    with (HERE / "commands.json").open("x") as stream:
        json.dump(commands, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "prepared_actual_SDK_free_source_and_recipe_passed", "fixture_arrays": 2, "new_native_arrays": 0,
                      "parsed_driver_delta": delta, "production_source": source, "future_effect": False, "GPU_used": False,
                      "SDK_started": False, "models_deserialized": False, "owned_imports": {name: module.__file__ for name, module in sys.modules.items() if name.startswith("CrowdSim") and getattr(module, "__file__", None)}}))


if __name__ == "__main__":
    main()
