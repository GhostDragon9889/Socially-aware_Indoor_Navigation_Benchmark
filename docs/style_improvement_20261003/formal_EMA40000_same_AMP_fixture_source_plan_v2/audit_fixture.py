"""Metadata admission only; reuse the original f851 raw physical mathematics.

Future actual inputs are required. Importing this file does not open a model,
capture, training owner or terminal report, and does not import NumPy/SDK/Torch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = HERE / "commands_and_source_plan.json"
PLAN_SHA = "8124d8b033549fb4b81bbc325bca23e237939ab6220ae182814cf807a0da6b9a"


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def sealed(path, sha):
    row = record(path)
    require(row["sha256"] == sha and not Path(path).stat().st_mode & 0o222,
            "Actual readonly document/hash required")
    return read(path), row


def production_rows(plan):
    epoch_row = plan["production_source_epoch_inherited_from_existing_pair"]["manifest"]
    require(record(epoch_row["path"]) == epoch_row, "Deployed185 epoch changed")
    epoch = read(epoch_row["path"])
    require(epoch["source_count"] == len(epoch["files"]) == 185, "Exact185 deployed source rows required")
    rows = [record(row["path"]) for row in epoch["files"]]
    require(rows == epoch["files"], "Current deployed185 bytes differ from frozen epoch")
    return rows


def admit_training(plan, terminal_path, terminal_sha):
    terminal, terminal_row = sealed(terminal_path, terminal_sha)
    selected = plan["future_candidate"]
    formal_row = selected["final_root_formal_plan"]
    require(record(formal_row["path"]) == formal_row, "Final selected2beed formal plan changed")
    formal = read(formal_row["path"])
    require(terminal["schema"] == "official_orca.all_temporal_actual_formal40000_CPU_checkpoint_audit.v1"
            and terminal["status"] == "passed" and terminal["CPU_only"] is True
            and terminal["reader"] == formal["terminal_auditor"] and terminal["plan"] == formal_row
            and all(terminal[key] is True for key in ("original_validate_resume_passed",
                    "full_raw_EMA_Adam_strict_load_passed", "production_deployment_model_key_strict_load_equal_to_EMA"))
            and terminal["production_deployment_forward_executed"] is False,
            "Real77025 CPU terminal strict-load admission required")
    require(record(terminal["reader"]["path"]) == terminal["reader"]
            and record(formal["source_registration"]["path"]) == formal["source_registration"]
            == selected["final_root_formal_source_registration"], "Actual terminal/source reader identity differs")
    owner_row = terminal["owner"]
    require(owner_row["path"] == selected["actual_training_owner"]["path"], "Wrong training owner path")
    owner, actual_owner = sealed(owner_row["path"], owner_row["sha256"])
    require(actual_owner == owner_row and owner["schema"] == "official_orca.all_temporal_actual_formal40000_root_owner.v1"
            and owner["status"] == "closed" and type(owner["actual_exit_code"]) is int and owner["actual_exit_code"] == 0
            and owner["child_exit_observed"] is True and owner["command"] == formal["formal_GPU_argv"]
            and owner["env"] == formal["GPU_environment_root_serial_only"]
            and owner["prepared_plan"] == formal_row and owner["source_registration"] == formal["source_registration"],
            "Actual canonical closed40k owner required; running or old keys are rejected")
    result_row = terminal["training_result"]
    result, actual_result = sealed(result_row["path"], result_row["sha256"])
    require(actual_result == result_row and result_row["path"] == str(Path(formal["formal_output"]) / "training_result.json")
            and result["status"] == "completed" and result["completed_optimizer_updates"] == 40000
            and result["start_optimizer_update"] == 0 and result["microsteps"] == 80000
            and result["training_windows"] == 16000000 and result["default_inference_weight_source"] == "EMA",
            "Exact fresh40k/16M EMA training result required")
    checkpoint = terminal["checkpoint"]
    path = Path(checkpoint["path"])
    require(checkpoint["path"] == selected["path"] == formal["future_checkpoint"]["path"]
            and record(path) == checkpoint and not path.stat().st_mode & 0o222 and path.stat().st_nlink == 1
            and path.stat().st_ino != (path.parent / "last.ckpt").stat().st_ino
            and result["snapshot"] == checkpoint["path"] and result["snapshot_sha256"] == checkpoint["sha256"],
            "Independent actual readonly terminal EMA identity differs")
    return {"terminal_CPU_audit": terminal_row, "training_owner": owner_row, "training_result": result_row,
            "planner": checkpoint, "formal_plan": formal_row, "formal_source_registration": formal["source_registration"]}


def admit_capture(plan, terminal_path, terminal_sha, owner_path, owner_sha):
    training = admit_training(plan, terminal_path, terminal_sha)
    owner, owner_row = sealed(owner_path, owner_sha)
    template = plan["templates_pending_actual_terminal_admission"]
    rows = production_rows(plan)
    epoch = plan["production_source_epoch_inherited_from_existing_pair"]["manifest"]
    controller = plan["controller_identity_inherited_not_rehashed"]
    require(owner["schema"] == plan["future_fixture_root_owner_contract"]["schema"] and owner["status"] == "closed"
            and type(owner["actual_exit_code"]) is int and owner["actual_exit_code"] == 0
            and owner["child_exit_observed"] is True and type(owner["actual_root_tool_session_id"]) is int
            and bool(owner["actual_root_launch_chunk"]) and bool(owner["actual_root_completion_chunk"])
            and owner["command"] == template["GPU_child_argv"] and owner["env"] == template["GPU_environment_root_serial_only"]
            and owner["production_source_before"] == owner["production_source_after"] == rows
            and owner["production_source_epoch"] == epoch
            and owner["actual_terminal_admission"] == training["terminal_CPU_audit"]
            and owner["checkpoint"] == controller and owner["planner"] == training["planner"]
            and owner["requested_profile"] == plan["unchanged_inputs_inherited_not_rehashed"]["default_profile"]
            and owner["MMD_launch_filter"] is False and owner["default_promoted"] is False
            and owner["sealed34_outputs_read"] is False, "Actual closed fixed AMP62 physical root-tool ABI differs")
    run = Path(template["parsed_driver_arguments_expected"]["output"])
    capture_rows = owner["capture_files_sealed"]
    require(len({row["path"] for row in capture_rows}) == len(capture_rows), "Duplicate capture identities")
    for row in capture_rows:
        path = Path(row["path"])
        require(path.parent == run and record(path) == row and not path.stat().st_mode & 0o222,
                "Actual whole closed capture must be readonly and exact")
    require({str(run / name) for name in ("result.json", "deployment_entry.json", "physical_states.npz",
            "ground_support.npz", "reference_plans.npz")} <= {row["path"] for row in capture_rows},
            "All original physical captures required")
    result, entry = read(run / "result.json"), read(run / "deployment_entry.json")
    require(entry["status"] == "SDK_free_preflight_passed" and entry["driver_argv"] == template["forwarded_driver_argv_expected"]
            and entry["parsed_driver_arguments"] == result["arguments"] == template["parsed_driver_arguments_expected"]
            and result["status"] == "passed" and type(result["actual_control_steps"]) is int
            and result["actual_control_steps"] == 6000, "Actual full200s entry/recipe differs")
    old_audit_row = plan["OLD_baseline"]["raw_audit"]
    require(record(old_audit_row["path"]) == old_audit_row, "Sealed matched OLD raw identity changed")
    historical_result = read(old_audit_row["path"])["actual_source_admission"]["result"]
    require(record(historical_result["path"]) == historical_result, "Matched OLD provenance result changed")
    historical = read(historical_result["path"])["evaluation_actor"]["full_training_provenance"]
    cp = training["planner"]
    require(result["provenance"] == historical and result["evaluation_actor"]["full_training_provenance"] == historical
            and result["evaluation_planner_change"] == {"training_sha256": historical["diffusion_sha256"],
                "evaluation_sha256": cp["sha256"], "checkpoint": cp["path"]},
            "Historical whole AMP provenance and actual EMA planner must remain distinct")
    request, actor = entry["evaluation_actor_request"], result["evaluation_actor"]
    require(request["base_checkpoint"] == {"path": controller["path"], "sha256": controller["sha256"]}
            and request["style_checkpoint"] is None and request["alpha"] == 0 and request["experimental_arm_style"] is False
            and actor["facade"] == "EvaluationActor" and actor["checkpoint"] == controller["path"]
            and actor["checkpoint_sha256"] == controller["sha256"] and actor["checkpoint_bytes"] == controller["bytes"]
            and actor["update_count"] == 1000 and actor["physical_transition_count"] == 8192000
            and actor["AMP_reward_available"] is False and actor["critic_value_available"] is False
            and actor["saved_training_rng_restored"] is False and "experimental_arm_style" not in actor,
            "Same actual AMP62 alpha0 actor-only identity required")
    rng, runtime = result["evaluation_rng_preparation"], result["runtime_execution"]
    require(rng["schema"] == "legacy_evaluation_initialization_rng.v1" and rng["seed"] == 7
            and all(rng[key] is False for key in ("saved_checkpoint_rng_restored", "trainer_created", "optimizer_created", "replay_created"))
            and result["trajectory_source"] == "diffusion" and runtime["diffusion_used_for_rollouts"] is True
            and runtime["diffusion_steps"] == 100 and runtime["diffusion_samples_per_actor"] == 1
            and runtime["precision"] == "FP32" and runtime["compiled_forward"] is False,
            "Original seed7 initialization/DP runtime boundary differs")
    headers = [row for row in result["resource_preflight"]["records"] if row["name"] == "diffusion_checkpoint"]
    require(len(headers) == 1 and headers[0]["path"] == cp["path"] and headers[0]["bytes"] == cp["bytes"]
            and headers[0]["mtime_ns"] == Path(cp["path"]).stat().st_mtime_ns,
            "Actual EMA resource header must match sealed identity; header is not a SHA proof")
    return {"schema": "formal_EMA40000.actual_fixture_source_admission.v1", "status": "passed", **training,
            "fixture_owner": owner_row, "result": record(run / "result.json"), "entry": record(run / "deployment_entry.json"),
            "production_epoch": epoch, "production185_records": rows, "controller": controller,
            "all_failed_contact_idle_reset_samples_retained": True, "MMD_launch_filter": False,
            "checkpoint_deserialized_by_reader": False, "complete_runtime_import_ledger": False}


def audit(plan, args):
    admission = admit_capture(plan, args.terminal_audit, args.terminal_sha256,
                              args.fixture_owner, args.fixture_owner_sha256)
    math = plan["future_reader_minimal_I_O_hooks"]["original_raw_math"]
    require(record(math["path"]) == math, "Original f851 numerical source changed")
    spec = importlib.util.spec_from_file_location("formal_EMA40000_original_f851_raw", math["path"])
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)  # NumPy and original math only, after actual metadata admission.
    original = raw.verify_sources
    try:
        raw.verify_sources = lambda *_args, **_kwargs: admission
        checks = raw.Audit()
        report = raw.audit_capture(Path(plan["templates_pending_actual_terminal_admission"]["parsed_driver_arguments_expected"]["output"]), checks)
    finally:
        raw.verify_sources = original
    counts, contacts = report["counts"], report["original_obstacle_peer_channels"]
    require(counts["controls"] == 6000 and counts["actors"] == 4 and counts["duration_s"] == 200.,
            "Original complete6000/4people/200s denominator required")
    require(production_rows(plan) == admission["production185_records"], "Production sources changed during raw read")
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules),
            "Raw reader imported a model or SDK")
    gate = {"controls": 6000, "actors": 4, "duration_s": 200., "falls": counts["falls"],
            "static_actor_controls": contacts["static_contact_actor_frames"],
            "peer_pair_controls": contacts["peer_contact_unordered_pair_frames"]}
    report.update(schema="formal_EMA40000.independent_physical_CPU_audit.v1", status="passed", checks=checks.checks,
                  actual_source_admission=admission, physical_gate_counts=gate,
                  physical_zero_gate_passed=all(gate[key] == 0 for key in ("falls", "static_actor_controls", "peer_pair_controls")),
                  mathematics_adapter_proof={"original_fixture_math_file": math, "verify_sources_only_replaced": True,
                    "arrays_force_COM_contacts_falls_counters_math_changed": False},
                  execution_reader=record(__file__), source_plan=record(PLAN), CPU_only_reader=True,
                  actual_capture_independently_audited=True, GPU_used_by_reader=False, SDK_started_by_reader=False,
                  all_failed_contact_idle_reset_samples_retained=True, style_score_used_for_admission=False,
                  no_default_promotion_claimed=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-audit", type=Path, required=True)
    parser.add_argument("--terminal-sha256", required=True)
    parser.add_argument("--fixture-owner", type=Path, required=True)
    parser.add_argument("--fixture-owner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Actual raw reader must hide CUDA")
    require(not args.output.exists(), "Preserve earlier actual report, including failures")
    require(record(PLAN)["sha256"] == PLAN_SHA and not PLAN.stat().st_mode & 0o222, "Frozen evaluation source plan changed")
    try:
        report = audit(read(PLAN), args)
    except BaseException as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump({"schema": "formal_EMA40000.independent_physical_CPU_audit.v1", "status": "failed",
                       "failure": repr(error), "actual_capture_independently_audited": False}, stream, indent=2)
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "physical_gate_counts": report["physical_gate_counts"]}))


if __name__ == "__main__":
    main()
