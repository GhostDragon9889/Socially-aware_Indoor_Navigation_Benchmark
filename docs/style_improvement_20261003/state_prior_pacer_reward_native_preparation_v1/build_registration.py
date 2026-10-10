"""Register future native commands on CPU; no model/capture/SDK is loaded."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SDK = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular bound source required")
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-reader-main", type=Path, required=True)
    args = parser.parse_args()
    if any((HERE / name).exists() for name in ("registration.json", "commands.json")):
        raise ValueError("Exclusive new registration required")
    helper = load(HERE / "validate_native_admission.py", "new_native_source_binding")
    sources = helper.prepare_sources()
    evaluation_dir = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_evaluation_v1"
    evaluation = json.loads((evaluation_dir / "registration.json").read_text())
    metrics_dir = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_CPU_metrics_preparation_v1"
    old_launcher = ROOT / "docs/style_improvement_20261003/state_prior_native_actual_runs_v1/launch_native_v2.py"
    tree = ast.parse(old_launcher.read_text())
    command_node = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                        and isinstance(node.value, ast.List) and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets))
    expression = compile(ast.Expression(command_node), str(old_launcher), "eval")
    manifest = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
    diffusion = ROOT / "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt"
    checkpoint = Path(evaluation["future_checkpoint"]["path"])
    cases = {}
    for actors in (4, 15):
        suffix = "" if actors == 4 else "_15"
        scene = ROOT / f"configs/diffusion_amp/native_warehouse_pedestrians{suffix}_clearance08_v1.yaml"
        output = ROOT / f"output/diffusion_amp/native_warehouse_pacer_reward{suffix}_clearance08_200s_v1"
        if output.exists() or output.is_symlink():
            raise ValueError("Exclusive future native output required")
        scope = {"SDK": SDK, "ROOT": ROOT, "CKPT": checkpoint, "config": helper.PROFILE,
                 "diffusion": diffusion, "manifest": manifest, "scene": scene, "output": output,
                 "args": argparse.Namespace(actors=actors), "str": str}
        command = eval(expression, {"__builtins__": {}}, scope)
        cases[str(actors)] = {"actors": actors, "scene": record(scene), "duration_s": 200.0,
                             "controls": 6000, "seed": 7, "arm_style_alpha": 0.0, "output": str(output), "command": command,
                             "actual_launch_record": str(ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_native_actual_runs_v1" / f"native{actors}_actual_launch.json"),
                             "actual_native_executed": False}
    reader = record(args.native_reader_main)
    if Path(reader["path"]).stat().st_mode & 0o222:
        raise ValueError("Actual readonly independent native reader required")
    bindings = {"evaluation_registration": record(evaluation_dir / "registration.json"),
                "evaluation_executor": record(evaluation_dir / "run_registered_evaluation.py"),
                "evaluation_source_freeze": record(evaluation_dir / "source_freeze_v1.json"),
                "native_reader_main": reader, "original_native_launcher": record(old_launcher),
                "manifest": record(manifest), "diffusion": record(diffusion),
                "fixture_metric_runner": record(metrics_dir / "run_CPU_metrics.py"),
                "fixture_metric_catalog": record(metrics_dir / "commands_CPU_metrics_v1.json"),
                "fixture_metric_freeze": record(metrics_dir / "source_freeze_v1.json")}
    raw_dir = ROOT / "docs/style_improvement_20261003/pacer_reward_metrics_v1/navigation_raw"
    registration = {"schema": "pacer_reward.registered_native_protocol.v1", "status": "prepared_not_executed",
                    "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
                    "future_effect": False, "current_source_binding": sources, "sources": bindings,
                    "controller_source_manifest": evaluation["sources"]["isolated_source_manifest"],
                    "future_checkpoint": {"path": str(checkpoint), "sha256": None, "bytes": None,
                                          "actual_terminal_admitted": False,
                                          "terminal_admission_path": evaluation["future_checkpoint"]["terminal_admission_path"]},
                    "fixture_inputs": {"raw_report_path": str(raw_dir / "audit.json"),
                                       "source_sidecar_path": str(raw_dir / "actual_source_admission.json"),
                                       "actual_CPU_command": [SDK, str(metrics_dir / "run_CPU_metrics.py"), "--metric", "navigation_raw"],
                                       "actual_closed_process_required": True},
                    "native4_audit_path": str(ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_native_actual_review_v1/native4_audit.json"),
                    "cases": cases,
                    "required_physical_progression": {"native4": "Actual terminal + full6000 fixture zero falls/static/peer + actual CPU closure",
                                                      "native15": "Actual complete native4 zero falls/static/peer + source-bound closure and independent raw reader"},
                    "style_score_is_launch_gate": False, "failed_completed_results_retained": True,
                    "author_eight_scene_equivalence": False, "original_V5_scene_configs_changed": False,
                    "original_command_list_AST_sha256": hashlib.sha256(ast.dump(command_node, include_attributes=False).encode()).hexdigest(),
                    "complete_runtime_import_ledger_claimed": False}
    path = HERE / "registration.json"
    path.write_text(json.dumps(registration, indent=2, allow_nan=False) + "\n")
    code = (HERE / "validate_native_admission.py").read_text()
    if code.count("UNFROZEN_REGISTRATION_SHA256") != 1:
        raise ValueError("Fresh unbound validator required")
    (HERE / "validate_native_admission.py").write_text(code.replace("UNFROZEN_REGISTRATION_SHA256", record(path)["sha256"]))
    commands = {"schema": "pacer_reward.future_native_commands.v1", "status": "prepared_not_executed", "future_effect": False,
                "registration": record(path), "validator": record(HERE / "validate_native_admission.py"),
                "GPU_root_serial_only": True, "commands": {key: value["command"] for key, value in cases.items()},
                "GPU_environment": {"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                                    "OPENBLAS_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                "CPU_prepare_command": [SDK, str(HERE / "validate_native_admission.py"), "--prepare-only"],
                "actual_checkpoint_SHA_terminal_and_physical_admissions_required": True,
                "checkpoint_path_is_future_not_admitted": str(checkpoint)}
    (HERE / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "registration": record(path), "GPU_used": False,
                      "future_checkpoint_read": False, "future_capture_read": False}))


if __name__ == "__main__":
    main()
