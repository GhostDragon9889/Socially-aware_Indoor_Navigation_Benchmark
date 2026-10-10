#!/usr/bin/env python3
"""Compare completed S3-R measurements to a completed shared AMP baseline.

Uses the original comparison kernel and its full population/reference checks.
Only the owner-completion admission and historical single-run display key are
bridged. No simulator, reference fitting, failure filtering or metric changes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import recovery_evaluation_admission as admission  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts import compare_shared_social_evidence as original  # noqa: E402
from scripts import evaluate_shared_social_recovery as evaluation  # noqa: E402
from scripts import resume_shared_social_recovery_evaluation as resume  # noqa: E402
from scripts import verify_shared_social_evidence_admission as strict  # noqa: E402

KERNEL_SHA = "76e589b05101480ac21ae229e7c8ecdefde049073670deaddb88a4306ed77a35"
STRICT_SHA = "f2cf03483f9bab988d9fb293093c644ef97b7cfeda32458e9ba0b2a93c40600a"
MEASUREMENTS = ("tracking", "style90", "pose45", "navigation", "navigation_CPU_audit")


def single_run_label_bridge(bundle, expected_label):
    """Alias a display key in a private copy; retain every value and input dict."""
    result = deepcopy(bundle)
    for name in ("style90", "pose45"):
        runs = result[name]["runs"]
        admission.require(list(runs) == [expected_label], "Require one explicitly named style population")
        result[name]["runs"] = {"shared2000": runs[expected_label]}
    inputs = result["style90"]["shared_social_reader_bridge"]["inputs"]
    admission.require(list(inputs) == [expected_label], "Require the matching explicit style input label")
    result["style90"]["shared_social_reader_bridge"]["inputs"] = {"shared2000": inputs[expected_label]}
    return result


def admit_recovery_evaluation(completion_path):
    identity = admission.readonly_identity(completion_path)
    completed = original.read(completion_path)
    if completed.get("schema") == resume.COMPLETION_SCHEMA:
        return resume.admit_completion(completion_path)
    admission.require(completed.get("schema") == "shared_social.durable_recovery_evaluation_owner_state.v1"
                      and completed.get("status") == "completed_evaluation_quality_unclassified"
                      and completed.get("actual_pipeline_exit_observed") is True,
                      "Require the complete actual OS-owned recovery evaluation")
    steps = completed.get("steps", [])
    admission.require(len(steps) == 1 and steps[0].get("name") == "post_training_evaluation"
                      and admission.exact_integer(steps[0].get("actual_exit_code"), 0)
                      and steps[0].get("exit_observation") == "subprocess.Popen.wait",
                      "Require the real pipeline exit from OS wait")
    plan = original.bound(completed["plan"])
    for row in plan["sources"] + plan["inputs"]:
        admission.require(record(row["path"]) == row, "Recovered evaluation source/input changed")
    pipeline = original.bound(completed["execution"])
    module, _ = evaluation.prepare("post")
    _, _, expected_admission = module.admit_training(Path(plan["training_receipt"]))
    expected_names = ["actor_CPU_inference", "tracking_style", "navigation_smoke", "navigation200s",
                      "navigation_CPU_array_audit"]
    admission.require(pipeline.get("status") == "passed_execution"
                      and pipeline.get("training_execution_admission") == expected_admission
                      and [row.get("name") for row in pipeline.get("steps", [])] == expected_names
                      and all(row.get("status") == "passed" and admission.exact_integer(row.get("actual_exit_code"), 0)
                              for row in pipeline["steps"])
                      and completed["measurements"] == pipeline["measurements"],
                      "Recovered pipeline stages, training admission or measurement identities disagree")
    tracking_pipeline = original.bound(pipeline["steps"][1]["result"])
    admission.require(tracking_pipeline.get("status") == "passed_execution"
                      and [row.get("name") for row in tracking_pipeline.get("steps", [])] ==
                      ["lifecycle_smoke", "full22", "style"]
                      and all(row.get("status") == "passed" and admission.exact_integer(row.get("actual_exit_code"), 0)
                              for row in tracking_pipeline["steps"]),
                      "Recovered tracking/style children did not all actually complete")
    result = {key: original.bound(completed["measurements"][key]) for key in MEASUREMENTS}
    admission.require(tracking_pipeline["steps"][1]["result"] == completed["measurements"]["tracking"]
                      and tracking_pipeline["steps"][2]["result"] == completed["measurements"]["style90"]
                      and result["style90"]["frozen_pose45_result"] == completed["measurements"]["pose45"]
                      and pipeline["steps"][3]["result"] == completed["measurements"]["navigation"]
                      and pipeline["steps"][4]["result"] == completed["measurements"]["navigation_CPU_audit"],
                      "Recovered child exits bind different measurement artifacts")
    actor = original.bound(pipeline["steps"][0]["result"])
    for stage, document in (("actor_cpu", actor), ("tracking_style", tracking_pipeline), ("tracking", result["tracking"])):
        module, _ = evaluation.prepare(stage)
        _, _, expected = module.admit_training(Path(plan["training_receipt"]))
        admission.require(document.get("training_execution_admission") == expected,
                          "Recovered child has a different admitted checkpoint/source: " + stage)
    admission.require(tracking_pipeline["training_receipt"] == expected_admission["receipt"],
                      "Tracking/style pipeline used another training receipt")
    admission.require(result["navigation_CPU_audit"]["result"] == completed["measurements"]["navigation"],
                      "Recovered raw audit does not bind the actual navigation result")
    admission.require(record(completion_path) == identity, "Recovered evaluation completion changed")
    return result, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline-post-receipt", required=True, type=Path)
    parser.add_argument("--recovery-completion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    admission.require(not args.output.exists(), "Preserve previous comparisons; use a new output")
    kernel, entry, strict_source = record(original.__file__), record(__file__), record(strict.__file__)
    admission.require(kernel["sha256"] == KERNEL_SHA, "Original independent comparison kernel changed")
    admission.require(strict_source["sha256"] == STRICT_SHA, "Original source/population/sampling admission changed")
    baseline_receipt = record(args.baseline_post_receipt)
    baseline, before_ids = strict.admit_post(args.baseline_post_receipt)
    candidate, completion = admit_recovery_evaluation(args.recovery_completion)
    before = single_run_label_bridge(baseline, "shared2000")
    after = single_run_label_bridge(candidate, "S3_R12500")
    after_ids = original.read(args.recovery_completion)["measurements"]
    artifacts = strict.unique_artifacts([*before_ids.values(), *after_ids.values(),
                                        *strict.bundle_sources(before, before_ids),
                                        *strict.bundle_sources(after, after_ids)])
    for item in artifacts:
        admission.require(record(item["path"]) == item, "Bound source/capture/array changed before comparison")
    comparison = strict.supplemental_comparison(before, after)
    for item in artifacts:
        admission.require(record(item["path"]) == item, "Bound source/capture/array changed during comparison")
    admission.require(record(args.baseline_post_receipt) == baseline_receipt and record(original.__file__) == kernel
                      and record(__file__) == entry and record(args.recovery_completion) == completion
                      and record(strict.__file__) == strict_source,
                      "Comparison input or implementation changed")
    admission.require("torch" not in sys.modules, "Independent JSON comparison must remain simulator-free")
    report = {"schema": "shared_social.completed_recovery_independent_comparison.v1", "status": "passed_comparison",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "reader": entry, "metric_kernel": kernel,
              "strict_source_population_sampling_admission": strict_source, "verified_artifacts": artifacts,
              "baseline_post_receipt": baseline_receipt, "recovery_evaluation_completion": completion,
              "comparison": comparison, "display_key_bridge": {"baseline": "shared2000", "candidate": "S3_R12500",
                  "input_documents_modified": False, "metric_values_changed": False},
              "exposure_qualification": "S2 fresh2000 versus S3-R effective12500 with a physical restart; not a single-variable causal experiment",
              "evaluation_execution_qualification": {
                  "candidate_completion_schema": original.read(args.recovery_completion)["schema"],
                  "successful_prefix_reused_with_separate_navigation_retry":
                      original.read(args.recovery_completion)["schema"] == resume.COMPLETION_SCHEMA,
                  "original_failed_post_pipeline_success_claimed": False},
              "Torch_imported": False, "SDK_started": False,
              "paper_effect_achieved": False, "default_model_promoted": False}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({"status": report["status"], "output": record(args.output),
                      "paper_effect_achieved": False, "default_model_promoted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
