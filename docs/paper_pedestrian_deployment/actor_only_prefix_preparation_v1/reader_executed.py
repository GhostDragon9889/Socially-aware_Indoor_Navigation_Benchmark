"""Reuse the frozen native/prefix reader for actual actor-only900 comparisons.

Preparation validates only existing old artifacts and source168. No future
candidate file is fabricated, and no tolerance substitutes for byte equality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
READER_SHA = "334bdc58ad1f455c6184763a64adcded72bf5deba1b1990c6180aebd7a57c826"
SOURCE_MANIFEST = ROOT / "docs/style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v4.json"
SOURCE_SHA = "07a34630e052a356aefa3755781bd4c5a7d1406d6a6dd83a0f0f3ec0de80b54f"
AMP_SHA = "570542a195372a36cf1874deed444495e7da11e763be2fdf8d6cc447d3fceef3"
PURE_BASELINE = ROOT / "output/diffusion_amp/pure_inference_baseline_30s_v1"
OLD_COMPARISON = HERE / "pure_inference_actual30s_prefix_comparison_v1/comparison.json"
DEFAULT_CANDIDATE = ROOT / "output/diffusion_amp/pure_actor_only_baseline_30s_v1"


def load_reader():
    path = HERE / "compare_physical_prefix.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != READER_SHA:
        raise ValueError("Frozen prefix reader changed")
    spec = importlib.util.spec_from_file_location("actor_only_frozen_prefix_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arrays(reader, directory):
    return {"physical": reader.prefix_arrays(directory / "physical_states.npz", 900),
            "ground": reader.prefix_arrays(directory / "ground_support.npz", 900),
            "plans": reader.prefix_arrays(directory / "reference_plans.npz", 900, plans=True)}


def comparisons(reader, left, right):
    return {name: reader.compare_fields(left[name], right[name]) for name in left}


def all_equal(compared):
    return all(row["bitwise_equal"] for block in compared.values() for row in block.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--preparation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("CPU-only environment and fresh output evidence directory required")
    reader = load_reader()
    helper = reader.load_helper()
    old_sources = reader.prepare_sources()
    old200, old200_arrays, old200_oracle = reader.baseline_oracle(old_sources, helper)
    pure = reader.read(PURE_BASELINE / "result.json")
    pure_artifacts = reader.artifact_identity(PURE_BASELINE, pure)
    previous = reader.read(OLD_COMPARISON)
    reader.require(previous["status"] == "measured_exact_prefix" and previous["paired_bitwise_equal"] is True,
                   "Prior pure30-vs-selected200 protocol was not admitted")
    reader.require(previous["candidate_artifacts"] == pure_artifacts, "Pure30 artifact identities changed")
    pure_arrays = arrays(reader, PURE_BASELINE)
    reader.validate_prefix(**pure_arrays, helper=helper)
    old_pair = comparisons(reader, old200_arrays, pure_arrays)
    reader.require(old_pair == previous["all_raw_field_comparisons"] and all_equal(old_pair),
                   "Old30/200 prefix equality no longer matches its historical record")
    source, source_bytes = reader.verify_archive(SOURCE_MANIFEST, SOURCE_SHA, 168)
    reader.require(hashlib.sha256(source_bytes["CrowdSim/humanoid/runtime/amp_inference.py"]).hexdigest() == AMP_SHA,
                   "Source168 lacks the tested actor-only implementation")
    denominator = {name: {"fields": len(block), "elements": sum(array.size for array in block.values()),
                          "field_elements": {field: array.size for field, array in block.items()}}
                   for name, block in pure_arrays.items()}
    report = {"schema": "actor_only.actual900_prefix_comparison.v1", "CPU_only": True, "SDK_started": False,
              "GPU_used": False, "sealed34_read": False, "reader": reader.record(__file__),
              "frozen_original_reader": reader.record(HERE / "compare_physical_prefix.py"),
              "source168_archive": source, "original200_baseline_oracle": old200_oracle,
              "old_pure30_artifacts": pure_artifacts, "historical_comparison": reader.record(OLD_COMPARISON),
              "old_pure30_equals_original200_prefix": True, "planned_candidate_dir": str(args.candidate_dir.resolve()),
              "comparison_denominators_per_pair": denominator, "candidate_comparison_executed": False,
              "candidate_capture_available": None, "paired_bitwise_equal": None,
              "status": "prepared_waiting_for_actual900_capture"}
    if not args.preparation_only:
        candidate = args.candidate_dir.resolve()
        result = reader.read(candidate / "result.json")
        reader.require(result["status"] == "passed" and result["actual_control_steps"] == 900,
                       "Candidate did not complete900 actual controls")
        reader.assert_argument_contract(old200["arguments"], result["arguments"])
        entry = reader.read(candidate / "deployment_entry.json")
        reader.require(entry["parsed_driver_arguments"] == result["arguments"], "Entry/actual argv differ")
        request = entry["evaluation_actor_request"]
        reader.require(type(request) is dict and request["alpha"] == 0., "Only unchanged alpha0 baseline admitted")
        for key in ("runtime_numerics", "physics", "evaluation_dataset", "evaluation_scene", "training_config"):
            reader.require(pure[key] == result[key], "Old pure30/actor-only evaluation contract differs:" + key)
        resources = reader.resource_identity(result)
        reader.require(resources == old200_oracle["resources"], "Weights/data/profile/fixture differ")
        reader.require(request["base_checkpoint"]["sha256"] == resources["controller_checkpoint"]["sha256"],
                       "Actor request selected a different checkpoint")
        networks = result["evaluation_actor"]["networks"]
        reader.require(networks["retained"] == ["task_encoder", "actor"]
                       and networks["validated_only"] == ["critic", "discriminator", "log_std"]
                       and networks["critic_forward_performed"] is False
                       and networks["Gaussian_distribution_created"] is False,
                       "Runtime does not declare the actor-only model")
        execution = result["evaluation_training_components"]
        for key in ("optimizer_constructed", "replay_constructed", "critic_evaluated",
                    "discriminator_evaluated", "training_transitions_collected"):
            reader.require(execution[key] is False, "Training-only component executed:" + key)
        reader.require(result["evaluation_rng_preparation"] == pure["evaluation_rng_preparation"],
                       "Legacy seed/CPU initialization contract changed")
        actual = arrays(reader, candidate)
        reader.validate_prefix(**actual, helper=helper)
        compare_pure = comparisons(reader, pure_arrays, actual)
        compare200 = comparisons(reader, old200_arrays, actual)
        equal = all_equal(compare_pure) and all_equal(compare200)
        saved_source_gate = helper.verify_sources
        try:
            def source_gate(audit, measured, **kwargs):
                return {"actual_prelaunch_source168_archive": source, "resources_full_sha256_recomputed": resources,
                        "source_admission_callback_only": True, "actual_runtime_per_file_attestation": False}
            helper.verify_sources = source_gate
            audit = helper.Audit()
            native = helper.audit_capture(candidate, audit)
            native.update(status="passed", checks=audit.checks, actual_native_auditor_sha256=reader.HELPER_SHA)
        finally:
            helper.verify_sources = saved_source_gate
        report.update({"candidate_comparison_executed": True, "candidate_capture_available": True,
                       "status": "measured_exact_prefix" if equal else "measured_differences", "paired_bitwise_equal": equal,
                       "candidate_artifacts": reader.artifact_identity(candidate, result), "candidate_entry": reader.record(candidate / "deployment_entry.json"),
                       "candidate_evaluation_rng_preparation": result["evaluation_rng_preparation"], "candidate_networks": networks,
                       "candidate_actual_native_measurement_audit": native, "candidate_prefix_statistics": reader.prefix_statistics(**actual),
                       "all_raw_vs_old_pure30": compare_pure, "all_raw_vs_original200_prefix": compare200,
                       "fields_compared_total": sum(len(block) for block in compare_pure.values()) * 2})
    report["limits"] = ["No candidate outputs read or created in preparation mode.",
                        "All shape/dtype/bytes comparisons are exact; numerical max differences accompany unequal bytes.",
                        "A900-control result cannot replace full200s quality evidence or prove universal determinism.",
                        "Source168 is a verified prelaunch archive, not a fabricated full runtime import ledger.",
                        "Original facade source did compute unused critic values despite historical metadata; only the new source is actor-only.",
                        "Original captures contain post-control states, not pre-step saved observations or initial GT/history.",
                        "Feet body-origin velocity remains a proxy rather than measured contact-patch slip."]
    reader.require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni"} for name in sys.modules),
                   "CPU prefix reader unexpectedly imported Torch/SDK")
    args.output.mkdir(parents=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "report.md").write_text("# Actor-only 30 秒配对\n\n" + report["status"] + "\n\n"
        + "candidate_comparison_executed=" + str(report["candidate_comparison_executed"])
        + "; paired_bitwise_equal=" + str(report["paired_bitwise_equal"]) + "。\n\n"
        + "\n\n".join(report["limits"]) + "\n")
    print({"status": report["status"], "candidate_comparison_executed": report["candidate_comparison_executed"],
           "paired_bitwise_equal": report["paired_bitwise_equal"]})
    return 2 if report["paired_bitwise_equal"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
