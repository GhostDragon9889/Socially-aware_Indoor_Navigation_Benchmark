"""Thin CPU I/O for completed state1950 full22 tracking; original math objects."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PLAN = HERE / "commands_and_sources.json"
sys.path.insert(0, str(ROOT))


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def require(value, message):
    if not value:
        raise ValueError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_objects(plan):
    for row in plan["sources"].values():
        require(record(row["path"]) == row, "Registered source changed: " + row["path"])
    from scripts import evaluate_pedestrian_style as style
    decomposition = load("original_tracking_decomposition", plan["sources"]["decomposition"]["path"])
    comparison = load("original_tracking_summary", plan["sources"]["comparison"]["path"])
    objects = {"style_main": style.main, "decompose": decomposition.decompose,
               "summarize": comparison.summarize}
    identities = {}
    for name, function in objects.items():
        identity = {"path": str(Path(inspect.getsourcefile(function)).resolve()),
                    "name": function.__name__,
                    "function_source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest()}
        require(identity == plan["original_function_objects"][name], "Original math function identity differs")
        identities[name] = identity
    return objects, identities


def sealed_document(path, digest):
    row = record(path)
    require(row["sha256"] == digest and not Path(path).stat().st_mode & 0o222,
            "Actual document must have the supplied sealed SHA")
    return json.loads(Path(path).read_text()), row


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--terminal-sha256")
    parser.add_argument("--result-sha256")
    parser.add_argument("--owner-receipt", type=Path)
    parser.add_argument("--owner-sha256")
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU metric entry requires hidden CUDA")
    plan = json.loads(PLAN.read_text())
    require(record(__file__) == plan["runner"], "Registered thin entry changed")
    objects, identities = source_objects(plan)
    if args.prepare_only:
        require(not any((args.terminal_sha256, args.result_sha256, args.owner_receipt, args.owner_sha256)),
                "Preparation does not admit or read future capture")
        require("torch" not in sys.modules and not any(name.startswith(("omni.", "isaaclab.")) for name in sys.modules),
                "Source preparation imported Torch or SDK")
        print(json.dumps({"status": "prepared_source_objects_passed", "original_function_objects": identities,
                          "future_capture_or_terminal_read": False, "metrics_computed": False}))
        return 0
    require(all((args.terminal_sha256, args.result_sha256, args.owner_receipt, args.owner_sha256)),
            "Actual closed owner, terminal and result SHAs are required")
    terminal, terminal_record = sealed_document(plan["terminal_audit"], args.terminal_sha256)
    require(terminal.get("schema") == "state_only_matched_q.actual_training_checkpoint_audit.v1"
            and terminal.get("status") == "passed"
            and terminal.get("actual_terminal_fresh2000_confirmed") is True
            and terminal.get("milestone_only_admission") is False
            and terminal.get("copied_files_hashed") == 936
            and terminal.get("source_manifest") == plan["training_source_manifest"], "Full state936 terminal admission differs")
    rows = terminal.get("checkpoints", [])
    require(len(rows) == 1, "Exactly one sealed terminal checkpoint expected")
    checkpoint = rows[0]
    require(checkpoint["updates"] == 2000 and checkpoint["transitions"] == 16384000
            and checkpoint["D_input"] == 1950 and checkpoint["full_payload_resume_validated"] is True
            and checkpoint["state_prior_initialization_validated"] is True
            and checkpoint["public_D_or_AMP_RMS_transferred"] is True
            and checkpoint["state_prior"]["D_action_columns"] == 0
            and checkpoint["checkpoint"]["path"] == plan["checkpoint"], "Terminal state1950 payload differs")
    owner, owner_record = sealed_document(args.owner_receipt, args.owner_sha256)
    require(type(owner.get("actual_exit_code")) is int and owner["actual_exit_code"] == 0
            and owner.get("child_exit_observed") is True, "Actual root GPU owner has not observed clean closure")
    expected_command = plan["GPU_command_before_actual_SHA_append"] + ["--checkpoint-sha256", checkpoint["checkpoint"]["sha256"]]
    require(owner.get("command") == expected_command, "Root owner did not run registered complete22 argv")
    capture = Path(plan["capture"])
    result, result_record = sealed_document(capture / "result.json", args.result_sha256)
    expected_args = dict(plan["benchmark_arguments"])
    expected_args["checkpoint_sha256"] = checkpoint["checkpoint"]["sha256"]
    require(result.get("status") == "passed" and result["arguments"] == expected_args
            and result["training_config"] == plan["state1950_training_config"], "Actual tracking recipe differs")
    expected = plan["complete22"]
    observed = [{"motion_id": row["motion_id"], "complete_metric_frames": row["observed_metric_steps"]}
                for row in result["sequences"]]
    require(observed == expected and result["aggregate"]["observed_metric_steps"] == 11982
            and all(row["complete_clip"] is True for row in result["sequences"]), "Complete22 or frame denominator differs")
    epoch = json.loads(Path(plan["sources"]["production_epoch"]["path"]).read_text())
    epoch_hashes = {str(Path(row["path"]).relative_to(ROOT)): row["sha256"] for row in epoch["files"]}
    require(all(epoch_hashes.get(path) == digest for path, digest in result["selected_source_sha256"].items())
            and result["selected_source_sha256"] == plan["benchmark_selected_source_sha256"], "Actual production185 capture source differs")
    for row in result["sequences"]:
        path = Path(row["capture"])
        require(path.parent.resolve() == (capture / "captures").resolve()
                and not path.stat().st_mode & 0o222 and not path.is_symlink(), "Capture must be sealed in registered directory")
    output = Path(plan["output"])
    require(not output.exists() and not output.is_symlink(), "Metric output must be fresh")
    objects["style_main"](plan["style_argv"])
    style = json.loads((output / "tracking90/evaluation.json").read_text())
    item = style["runs"][plan["run_name"]]
    require(item["sources"]["result"] == result_record, "Original90 source result differs")
    decomposition = objects["decompose"](capture)
    require(decomposition["primary_aggregate_unchanged"] == result["aggregate"], "Original decomposition changed aggregate")
    decomposition_path = output / "decomposition.json"
    write(decomposition_path, decomposition)
    summary = objects["summarize"](item, result, decomposition, decomposition_path)
    write(output / "comparison.json", {"schema": "state_only_matched_q.actual_tracking_style_summary.v1",
          "status": "passed", "summary": summary, "historical_metrics_recomputed": False,
          "all_failed_and_idle_frames_retained": True, "metric_pass_is_quality_pass": False})
    import torch
    require(not torch.cuda.is_initialized() and not any(name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules),
            "CPU metric entry initialized CUDA or SDK")
    write(output / "actual_source_admission.json", {"schema": "state_only_matched_q.actual_tracking_metric_source_admission.v1",
          "status": "passed", "runner": record(__file__), "plan": record(PLAN), "sources": plan["sources"],
          "original_function_objects": identities, "actual_terminal_audit": terminal_record,
          "actual_root_closed_owner": owner_record, "actual_capture_result": result_record,
          "checkpoint": checkpoint["checkpoint"], "reader_deserialized_checkpoint": False,
          "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False})
    print(json.dumps({"status": "passed", "output": str(output), "metric_frames": 11982, "clips": 22}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
