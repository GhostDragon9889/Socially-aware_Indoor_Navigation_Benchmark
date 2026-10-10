"""State1950/current185 fixture I/O; original f851 physical functions unchanged.

Preparation admits the already completed original SDK-free resource preflight.
Actual reading requires a genuine closed root tool owner and sealed whole capture first.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN_SHA256 = "22b9eac0cea44d88cd6a58b3409259c587435a579b063177964c00f73f114d1d"
RUNS = ROOT / "docs/style_improvement_20261003/state_only_matched_q_physical_actual_runs_v1"
PREFLIGHT = RUNS / "SDK_free_preflight.stdout.json"
PREFLIGHT_RECEIPT = RUNS / "SDK_free_preflight_actual_receipt.json"
BASE_PATH = ROOT / "docs/style_improvement_20261003/default_newDP_physical_preparation_v1/physical_evidence.py"
BASE_SHA256 = "30202b34a13b5756ffa5c9b41110103a913c3883ed1d88d8389546322a8fc05a"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load("state1950_existing_stdlib_source_binding", BASE_PATH)
read, record, require = BASE.read, BASE.record, BASE.require


def plan():
    require(record(HERE / "commands.json")["sha256"] == PLAN_SHA256,
            "Exact prepared state1950 fixture command required")
    require(record(BASE_PATH)["sha256"] == BASE_SHA256,
            "Original stdlib source binding helper changed")
    return read(HERE / "commands.json")


def terminal_binding(registered):
    for key in ("terminal_admission", "terminal_root_seal", "requested_profile", "training_source_manifest"):
        row = registered[key]
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222,
                "Sealed state1950 input changed: " + key)
    document = read(registered["terminal_admission"]["path"])
    require(document["schema"] == "state_only_matched_q.actual_training_checkpoint_audit.v1"
            and document["status"] == "passed" and document["phase"] == "terminal2000"
            and document["actual_terminal_fresh2000_confirmed"] is True
            and document["milestone_only_admission"] is False
            and document["copied_files_hashed"] == 936
            and document["source_manifest"] == registered["training_source_manifest"]
            and document["check_count"] == len(document["checks"])
            and all(row["passed"] is True for row in document["checks"]),
            "Actual complete passed terminal48ec payload admission required")
    require(len(document["checkpoints"]) == 1, "One actual state1950 terminal required")
    row = document["checkpoints"][0]
    require(row["checkpoint"] == registered["checkpoint"]
            and all(row[key] == value for key, value in registered["terminal_facts"].items())
            and row["full_payload_resume_validated"] is True
            and row["state_prior"]["strategy"] == "public_pacer_state_only"
            and row["state_prior"]["D_action_columns"] == 0,
            "Exact state1950 counters/prior/history moments differ")
    sampling = row["actual_expert_sampling_saved_state"]
    require(sampling["private_CPU_RNG_saved_and_actual_loader_exact"] is True
            and sampling["actual_bank_registry_rebuilt"] is True
            and sampling["sampled_windows"] == 8192000 and sampling["sampled_batches"] == 2000
            and sampling["importance_correction_applied"] is False,
            "Actual saved q sampler admission differs")
    seal = read(registered["terminal_root_seal"]["path"])
    require(seal["schema"] == "state_only_matched_q.actual_terminal_root_seal.v1"
            and seal["status"] == "sealed_after_actual_GPU_and_watch_exit0"
            and seal["actual_updates"] == 2000 and seal["actual_physical_transitions"] == 16384000
            and {key: seal["checkpoint"][key] for key in ("path", "bytes", "sha256")} == registered["checkpoint"]
            and Path(registered["checkpoint"]["path"]).stat().st_nlink == 1
            and not Path(registered["checkpoint"]["path"]).stat().st_mode & 0o222,
            "Actual independent readonly terminal checkpoint/root closure differs")
    training = document["terminal_training_result"]
    require(record(training["path"]) == training and not Path(training["path"]).stat().st_mode & 0o222,
            "Actual sealed training result changed")
    config = read(training["path"])["training_config"]
    require(config["discriminator_input"] == "state" and config["warm_start_discriminator"] is True
            and config["warm_start_action_discriminator_from_state_prior"] is False
            and config["amp_reward_scale"] == config["amp_reward_weight"] == 2.
            and config["expert_sampling"] == sampling["configuration"],
            "State-only reward2/saved q configuration differs")
    return {"terminal_admission": registered["terminal_admission"],
            "terminal_root_seal": registered["terminal_root_seal"],
            "checkpoint": registered["checkpoint"], "training_result": training}


def source_binding(registered):
    terminal = terminal_binding(registered)
    source = registered["production_source"]
    observed = BASE.verify_epoch(source["manifest"], source["archive"], 185, current=True)
    require(observed == source, "Actual production185 source differs")
    for key in ("checkpoint", "planner", "requested_profile", "fixture_raw_math"):
        row = registered[key]
        require(record(row["path"]) == row, "Opaque input/source changed: " + key)
    require(PREFLIGHT.is_file() and not PREFLIGHT.stat().st_mode & 0o222,
            "Actual SDK-free original preflight must be sealed before launch")
    require(PREFLIGHT_RECEIPT.is_file() and not PREFLIGHT_RECEIPT.stat().st_mode & 0o222,
            "Actual resource preflight receipt must be sealed")
    receipt = read(PREFLIGHT_RECEIPT)
    require(receipt["schema"] == "state_only_matched_q.actual_SDK_free_resource_preflight.v1"
            and receipt["actual_child_return_code"] == 0 and receipt["child_exit_observed"] is True
            and receipt["preparation_plan_sha256"] == PLAN_SHA256
            and receipt["command"] == registered["SDK_free_preflight_command"],
            "Actual completed original resource preflight/command differs")
    return {"production_source": source, "terminal": terminal,
            "command_registration": record(HERE / "commands.json"),
            "reader": record(__file__), "actual_preflight": record(PREFLIGHT),
            "actual_preflight_receipt": record(PREFLIGHT_RECEIPT),
            "planner": registered["planner"], "requested_profile": registered["requested_profile"],
            "complete_runtime_import_ledger": False}


def prepare(registered):
    return {"status": "source_only_existing_actual_preflight_admitted", "preflight": record(PREFLIGHT),
            "source_binding": source_binding(registered), "physical_execution_verified": False}


def admit_capture(registered, owner_path, owner_sha256):
    binding = source_binding(registered)
    require(str(Path(owner_path).absolute()) == registered["future_GPU_owner"],
            "Registered root owner path required")
    require(record(owner_path)["sha256"] == owner_sha256 and not Path(owner_path).stat().st_mode & 0o222,
            "Actual root owner must be sealed and SHA-bound")
    owner = read(owner_path)
    require(owner["schema"] == "state_only_matched_q.actual_fixture_root_owner.v1"
            and owner["status"] == "closed" and type(owner["actual_exit_code"]) is int
            and owner["actual_exit_code"] == 0 and owner["child_exit_observed"] is True
            and type(owner["actual_root_tool_session_id"]) is int
            and all(isinstance(owner[key], str) and owner[key] for key in
                    ("actual_root_launch_chunk", "actual_root_completion_chunk"))
            and owner["command"] == registered["command"]
            and owner["production_source_epoch"] == registered["production_source"]["manifest"]
            and owner["production_source_before"] == owner["production_source_after"]
                == read(registered["production_source"]["manifest"]["path"])["files"]
            and owner["actual_terminal_admission"] == registered["terminal_admission"]
            and owner["checkpoint"] == registered["checkpoint"] and owner["planner"] == registered["planner"]
            and owner["requested_profile"] == registered["requested_profile"]
            and isinstance(owner["capture_files_sealed"], list) and owner["capture_files_sealed"],
            "Real complete root tool process/exact command/source closure required")
    output = Path(registered["output"])
    files = {str(x) for x in output.rglob("*") if x.is_file()}
    artifacts = {row["path"]: row for row in owner["capture_files_sealed"]}
    require(files == set(artifacts) and len(artifacts) == len(owner["capture_files_sealed"]),
            "Whole sealed capture ledger required")
    require({str(output / name) for name in ("result.json", "deployment_entry.json", "physical_states.npz",
            "ground_support.npz", "reference_plans.npz")} <= files, "Complete actual captures required")
    for path, row in artifacts.items():
        require(record(path) == row and not Path(path).stat().st_mode & 0o222,
                "Readonly actual capture changed")
    result, entry = read(output / "result.json"), read(output / "deployment_entry.json")
    require(entry == read(PREFLIGHT) and result["arguments"] == entry["parsed_driver_arguments"]
            and result["status"] == "passed" and result.get("error") is None
            and type(result["actual_control_steps"]) is int and result["actual_control_steps"] == 6000,
            "Actual original preflight/complete6000 result differs")
    request = entry["evaluation_actor_request"]
    checkpoint = registered["checkpoint"]
    require(request["base_checkpoint"] == {key: checkpoint[key] for key in ("path", "sha256")}
            and request["style_checkpoint"] is None and request["alpha"] == 0.
            and request["experimental_arm_style"] is False, "Actual actor-only alpha0 producer ABI differs")
    training = read(binding["terminal"]["training_result"]["path"])
    require(result["training_config"] == training["training_config"]
            and result["provenance"] == training["provenance"]
            and result["provenance"]["diffusion_checkpoint"] == registered["planner"]["path"]
            and result["provenance"]["diffusion_sha256"] == registered["planner"]["sha256"]
            and result.get("evaluation_planner_change") is None, "Whole historical state1950/OLD63e provenance differs")
    actor = result["evaluation_actor"]
    require(actor["facade"] == "EvaluationActor" and actor["checkpoint"] == checkpoint["path"]
            and actor["checkpoint_sha256"] == checkpoint["sha256"] and actor["checkpoint_bytes"] == checkpoint["bytes"]
            and actor["update_count"] == 2000 and actor["physical_transition_count"] == 16384000
            and actor["full_training_provenance"] == training["provenance"]
            and all(actor[key] is False for key in ("AMP_reward_available", "critic_value_available", "saved_training_rng_restored")),
            "Actually loaded state1950 actor/counters/provenance differs")
    rng = result["evaluation_rng_preparation"]
    require(rng["schema"] == "legacy_evaluation_initialization_rng.v1" and rng["seed"] == 7
            and all(rng[key] is False for key in ("saved_checkpoint_rng_restored", "trainer_created",
                                                "optimizer_created", "replay_created")), "Original actor-only seed7 boundary differs")
    runtime = result["runtime_execution"]
    require(result["trajectory_source"] == "diffusion" and runtime["diffusion_used_for_rollouts"] is True
            and runtime["diffusion_steps"] == 100 and runtime["diffusion_samples_per_actor"] == 1
            and runtime["precision"] == "FP32" and runtime["compiled_forward"] is False,
            "Actual fixed OLD63e/S1/FP32 planner runtime differs")
    return {"schema": "state_only_matched_q.actual_fixture_source_admission.v1", "status": "passed", **binding,
            "actual_root_owner": record(owner_path), "result": record(output / "result.json"),
            "entry": record(output / "deployment_entry.json"), "full_historical_AMP_provenance_exact": True,
            "checkpoint_deserialized_by_reader": False, "MMD_launch_filter": False}


def audit(registered, owner_path, owner_sha256):
    admission = admit_capture(registered, owner_path, owner_sha256)
    raw = load("state1950_original_f851_raw_math", registered["fixture_raw_math"]["path"])
    original = raw.verify_sources
    try:
        raw.verify_sources = lambda *_args, **_kwargs: admission
        checks = raw.Audit()
        report = raw.audit_capture(Path(registered["output"]), checks)
    finally:
        raw.verify_sources = original
    report["checks"] = checks.checks
    counts, contacts = report["counts"], report["original_obstacle_peer_channels"]
    require(counts["controls"] == 6000 and counts["actors"] == 4 and counts["duration_s"] == 200.,
            "Whole4x6000/200s raw denominators required")
    zero = counts["falls"] == contacts["static_contact_actor_frames"] == contacts["peer_contact_unordered_pair_frames"] == 0
    report.update(schema="state_only_matched_q.actual_fixture_raw_audit.v1", status="passed",
                  actual_capture_independently_audited=True, physical_zero_gate_passed=zero,
                  actual_source_admission=admission, original_math=registered["fixture_raw_math"],
                  only_verify_sources_hook_replaced=True, all_failed_contact_idle_reset_samples_retained=True,
                  source_post_observation=source_binding(registered), CPU_only=True,
                  physical_quality_or_default_promotion_claimed=False, MMD_launch_filter=False)
    require(report["source_post_observation"] == {key: admission[key] for key in report["source_post_observation"]},
            "Source changed during original raw read")
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}
                    for name in sys.modules), "CPU raw reader imported model/SDK")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--owner", type=Path)
    parser.add_argument("--owner-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Reader/preparation requires hidden CUDA")
    registered = plan()
    if args.prepare_only:
        require(not any((args.owner, args.owner_sha256, args.output)), "Preparation cannot consume future captures")
        print(json.dumps(prepare(registered), indent=2, allow_nan=False))
        return 0
    require(args.owner is not None and args.owner_sha256 is not None and args.output is not None,
            "Actual root owner/SHA and new raw report output required")
    require(str(args.output.absolute()) == registered["future_raw_report"] and not args.output.exists(),
            "New registered raw output required")
    try:
        report = audit(registered, args.owner, args.owner_sha256)
    except BaseException as error:
        report = {"schema": "state_only_matched_q.actual_fixture_raw_audit.v1", "status": "failed",
                  "failure": repr(error), "actual_capture_independently_audited": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "physical_zero_gate_passed": report["physical_zero_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
