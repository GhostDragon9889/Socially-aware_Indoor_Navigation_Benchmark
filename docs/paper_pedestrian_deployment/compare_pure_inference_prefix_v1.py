"""Compare actual pure inference to the frozen baseline; preserve raw differences."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
READER_SHA = "334bdc58ad1f455c6184763a64adcded72bf5deba1b1990c6180aebd7a57c826"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("CPU-only and a fresh evidence directory are required")
    reader_path = HERE / "compare_physical_prefix.py"
    if hashlib.sha256(reader_path.read_bytes()).hexdigest() != READER_SHA:
        raise ValueError("Original prefix/measurement reader changed")
    spec = importlib.util.spec_from_file_location("frozen_original_prefix_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    helper = reader.load_helper()
    old_sources = reader.prepare_sources()
    baseline, original, oracle = reader.baseline_oracle(old_sources, helper)
    freeze = HERE / "pure_inference_deployment_source_v1.json"
    frozen = reader.read(freeze)
    source, source_bytes = reader.verify_archive(freeze, reader.record(freeze)["sha256"], 146)
    reader.require(frozen["source_count"] == len(source_bytes) == 146, "New source count differs")
    # New control-loop source is explicit: no substitution of the old loop hash.
    source["control_loop_SHA256"] = reader.control_loop_sha(source_bytes["scripts/train_paper_amp.py"])
    source["same_loop_source_as_old"] = False
    candidate = args.candidate_dir.resolve()
    result = reader.read(candidate / "result.json")
    reader.require(result["status"] == "passed" and result["actual_control_steps"] == 900,
                   "Candidate did not complete900 actual controls")
    reader.assert_argument_contract(baseline["arguments"], result["arguments"])
    entry = reader.read(candidate / "deployment_entry.json")
    reader.require(entry["parsed_driver_arguments"] == result["arguments"], "Forwarded argv differs")
    request = entry["evaluation_actor_request"]
    reader.require(type(request) is dict and request["alpha"] == 0., "Only alpha0 parity is admitted here")
    execution = result["evaluation_training_components"]
    for name in ("optimizer_constructed", "replay_constructed", "critic_evaluated",
                 "discriminator_evaluated", "training_transitions_collected"):
        reader.require(execution[name] is False, "Evaluation training component enabled:"+name)
    reader.require(result["evaluation_actor"]["outputs"] == ["actions", "mean_actions", "applied_actions"],
                   "Pure actor output contract differs")
    for name in ("runtime_numerics", "physics", "evaluation_dataset", "evaluation_scene", "training_config"):
        left = baseline[name]
        if name == "training_config":
            left = dict(left)
            left.setdefault("expert_sampling", None)
        reader.require(left == result[name], "Physical/evaluation contract differs:"+name)
    resources = reader.resource_identity(result)
    reader.require(resources == oracle["resources"], "Baseline weights/resources differ")
    reader.require(request["base_checkpoint"]["sha256"] == resources["controller_checkpoint"]["sha256"],
                   "Inference selected checkpoint SHA differs")
    actual = {"physical": reader.prefix_arrays(candidate/"physical_states.npz", 900),
              "ground": reader.prefix_arrays(candidate/"ground_support.npz", 900),
              "plans": reader.prefix_arrays(candidate/"reference_plans.npz", 900, plans=True)}
    reader.validate_prefix(actual["physical"], actual["ground"], actual["plans"], helper)
    original_source_gate = helper.verify_sources
    try:
        def source_gate(audit, measured, **kwargs):
            return {"new_source_archive": source, "resources_full_sha256_recomputed": resources,
                    "source_admission_callback_only": True, "actual_runtime_per_file_attestation": False}
        helper.verify_sources = source_gate
        audit = helper.Audit()
        native = helper.audit_capture(candidate, audit)
        native.update(status="passed", checks=audit.checks,
                      actual_native_auditor_sha256=reader.HELPER_SHA)
    finally:
        helper.verify_sources = original_source_gate
    comparisons = {name: reader.compare_fields(original[name], actual[name]) for name in actual}
    equal = all(row["bitwise_equal"] for block in comparisons.values() for row in block.values())
    output = {"schema": "pure_inference.actual900_prefix_comparison.v1", "CPU_only": True,
              "SDK_started": False, "GPU_used": False, "new34_outputs_read": False,
              "status": "measured_exact_prefix" if equal else "measured_differences",
              "paired_bitwise_equal": equal, "reader": reader.record(__file__),
              "original_prefix_reader": reader.record(reader_path), "baseline_oracle": oracle,
              "candidate_source": source, "candidate_artifacts": reader.artifact_identity(candidate, result),
              "candidate_entry": reader.record(candidate/"deployment_entry.json"),
              "inference_request": request, "evaluation_training_components": execution,
              "evaluation_rng_preparation": result["evaluation_rng_preparation"],
              "all_raw_field_comparisons": comparisons,
              "candidate_actual_native_measurement_audit": native,
              "candidate_prefix_statistics": reader.prefix_statistics(**actual),
              "limits": ["Actual30s comparison does not replace full200s stability evidence.",
                         "Source146 changes the evaluator branch; source hashes are not asserted equal to old source.",
                         "Initial captures are post-control, not a captured pre-step GT/history oracle.",
                         "All raw failed frames are retained; no tolerance replaces dtype/byte comparison.",
                         "Feet origins provide velocity proxies, not native contact-patch slip."]}
    args.output.mkdir(parents=True)
    (args.output/"comparison.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    (args.output/"report.md").write_text("# 纯推理实际30秒配对\n\n"+output["status"]+
        "; paired_bitwise_equal="+str(equal)+"。\n\n"+"\n\n".join(output["limits"])+"\n")
    print({"status": output["status"], "paired_bitwise_equal": equal})
    return 0 if equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
