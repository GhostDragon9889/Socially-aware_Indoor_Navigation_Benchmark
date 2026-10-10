"""Append a genuine full22 state-action capture to frozen13 validation reports.

Uses the unchanged d19c9 evaluator/decomposition invocation tool. Only the
run-label/checkpoint-admission bridge differs; no metric/ref/kernel retuning.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STYLE = HERE / "mixed_curriculum_style_v1"
PRIOR = HERE / "state_action_style_v1/full22_step2000_comparison_v1"
OUT_ROOT = HERE / "terrain_protected_style_v1"
TOOL_SHA = "d19c9cea62839792c85fccc92cdefd5bf9a3f1d27c3c39c7b0b7f6d0d4783612"
REFERENCE_SHA = "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_completed_tracking(result, baseline):
    if (
        result.get("status") != "passed"
        or result.get("error") is not None
        or result.get("arguments", {}).get("split") != "validation"
        or any(
            result["arguments"].get(key) is not None
            for key in ("motion_ids", "max_clips", "max_steps")
        )
        or len(result.get("sequences", [])) != 22
        or result.get("aggregate", {}).get("observed_metric_steps") != 11982
    ):
        raise ValueError(
            "Complete failed-inclusive22/11982 validation required, no test or partial capture"
        )
    for key in ("protocol", "physics", "layout", "reference_velocity_contract"):
        if result.get(key) != baseline.get(key):
            raise ValueError(
                "Frozen physical/reference tracking contract differs:" + key
            )
    if {row["motion_id"] for row in result["sequences"]} != {
        row["motion_id"] for row in baseline["sequences"]
    }:
        raise ValueError("Full22 source identities changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-admission", type=Path, required=True)
    parser.add_argument("--import-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or output.exists()
        or OUT_ROOT not in output.parents
    ):
        raise ValueError("CPU-only fresh descendant of state_action_style_v1 required")
    tool_path = STYLE / "compare.py"
    if record(tool_path)["sha256"] != TOOL_SHA:
        raise ValueError("Frozen actual mixed comparison/decomposition tool changed")
    prior_eval = json.loads((PRIOR / "evaluation.json").read_text())
    prior_summary = json.loads((PRIOR / "comparison.json").read_text())
    if (
        len(prior_eval["runs"]) != 13
        or set(prior_eval["runs"]) != set(prior_summary["runs"])
        or record(PRIOR / "heldout_reference.json")["sha256"] != REFERENCE_SHA
    ):
        raise ValueError("Frozen13-run predecessor/reference changed")
    baseline = json.loads(
        Path(prior_summary["runs"]["mixed2000"]["result"]["path"]).read_text()
    )
    candidate = json.loads((args.input / "result.json").read_text())
    validate_completed_tracking(candidate, baseline)
    admission = json.loads(args.checkpoint_admission.read_text())
    executor = load_module(
        "state_action_style_registered_executor",
        HERE / "run_isolated_terrain_protected_evaluation_v1.py",
    )
    evidence = load_module(
        "state_action_style_actual_evidence",
        HERE / "terrain_protected_evaluation_evidence_v1.py",
    )
    actual_checkpoint = record(candidate["arguments"]["checkpoint"])
    executor.validate_admission(
        admission,
        actual_checkpoint,
        json.loads(
            (
                HERE / "terrain_protected_evaluation_registration_v1/registration.json"
            ).read_text()
        ),
    )
    if admission.get("actual_terminal_fresh2000_confirmed") is not True:
        raise ValueError(
            "Actual terminal full2000 admission, not a partial milestone, required"
        )
    actual_source = evidence.admit_actual_evaluation(
        candidate, "tracking", args.checkpoint_admission, args.import_report
    )
    ledger_bytes, events_bytes = (
        actual_source.pop("ledger_bytes"),
        actual_source.pop("event_prefix_bytes"),
    )
    # Invoke the frozen original function bodies with only an explicit new
    # label/output root. Original nine dictionaries are preserved by its own
    # checks; mixed500/1000/2000 are added below without recomputation.
    tool = load_module("state_action_reused_frozen_mixed_comparison", tool_path)
    tool.HERE = OUT_ROOT
    tool.EXPECTED = {"protected2000": 2000}
    original_checkpoint_contract = tool.checkpoint_contract

    def checkpoint_contract(result, name):
        value = original_checkpoint_contract(result, name)
        executor.validate_admission(
            admission,
            value["checkpoint"],
            json.loads(
                (
                    HERE
                    / "terrain_protected_evaluation_registration_v1/registration.json"
                ).read_text()
            ),
        )
        return {
            **value,
            "discriminator_input": "state_action",
            "discriminator_dimensions": 2019,
            "public_D_or_AMP_RMS_transferred": False,
            "actual_terminal_admission": record(args.checkpoint_admission),
            "actual_isolated_source_manifest": admission["source_manifest"],
            "actual_observed_owned_imports": admission["observed_imports"],
        }

    tool.checkpoint_contract = checkpoint_contract
    saved_argv = sys.argv
    try:
        sys.argv = [
            str(tool_path),
            "--run",
            "protected2000=" + str(args.input.resolve()),
            "--output",
            str(output / "base_nine_comparison"),
        ]
        tool.main()
    finally:
        sys.argv = saved_argv
    child = output / "base_nine_comparison"
    (output / "GPU_import_report_as_observed.json").write_bytes(ledger_bytes)
    (output / "GPU_import_events_as_observed.jsonl").write_bytes(events_bytes)
    fresh_eval = json.loads((child / "evaluation.json").read_text())
    fresh_summary = json.loads((child / "comparison.json").read_text())
    merged_eval, merged_summary = deepcopy(prior_eval), deepcopy(prior_summary)
    merged_eval["runs"]["protected2000"] = fresh_eval["runs"]["protected2000"]
    merged_summary["runs"]["protected2000"] = fresh_summary["runs"]["protected2000"]
    exact_eval = {
        name: merged_eval["runs"][name] == row
        for name, row in prior_eval["runs"].items()
    }
    exact_summary = {
        name: merged_summary["runs"][name] == row
        for name, row in prior_summary["runs"].items()
    }
    if not all(exact_eval.values()) or not all(exact_summary.values()):
        raise ValueError("Frozen historical13 dictionaries changed")
    now = datetime.now(timezone.utc).isoformat()
    context = {
        "wrapper": record(__file__),
        "original_frozen_tool": record(tool_path),
        "actual_terminal_admission": record(args.checkpoint_admission),
        "candidate_result": record(args.input / "result.json"),
        "child_exact_score_output": record(child / "evaluation.json"),
        "prior13_evaluation": record(PRIOR / "evaluation.json"),
        "prior13_summary": record(PRIOR / "comparison.json"),
        "historical13_recomputed": False,
        "actual_evaluation_source_admission": actual_source,
        "source_boundary": "Original comparison source135 binds the old9 baseline only; the new state-action source is isolated931/old926 plus terrain-statistics delta, independently admitted. No same-source claim.",
        "sealed34_outputs_read": False,
        "kernel_feature_bins_threshold_reference_changed": False,
    }
    merged_eval.update(generated_at_utc=now, append_provenance=context)
    merged_summary.update(
        schema="state_action_frozen14_validation_style.v1",
        generated_at_utc=now,
        CPU_only=True,
        GPU_used=False,
        SDK_started=False,
        prior13_evaluation_dicts_exact=exact_eval,
        prior13_summary_dicts_exact=exact_summary,
        append_provenance=context,
        candidate_initial_reference_pairs=fresh_summary[
            "same_initial_reference_pairs_vs_negative_baseline"
        ],
        quality_or_author_effect_claim=False,
    )
    write(output / "evaluation.json", merged_eval)
    write(output / "comparison.json", merged_summary)
    shutil.copyfile(child / "heldout_reference.json", output / "heldout_reference.json")
    if record(output / "heldout_reference.json")["sha256"] != REFERENCE_SHA:
        raise ValueError("Frozen reference bytes changed")
    print(
        json.dumps(
            {
                "status": "passed",
                "historical13_exact": True,
                "new_run": "protected2000",
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
