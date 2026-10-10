#!/usr/bin/env python3
"""Compare the completed fresh episode2000 candidate with the fixed S2 baseline.

Cold JSON/hash admission only. Require all actual owner/child exits, complete
populations and the original frozen metric kernels before comparing outcomes.
The post owner carries two explicit metadata extensions while unchanged child
evaluators carry the original admission; both complete views are verified.
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
from scripts import compare_shared_social_evidence as kernel  # noqa: E402
from scripts import verify_shared_social_evidence_admission as strict  # noqa: E402
from scripts import own_shared_social_episode_pipeline as owner  # noqa: E402
from scripts import evaluate_shared_social_tracking as child_admission  # noqa: E402

HERE = ROOT / "docs/style_improvement_20261008/shared_social_episode_comparison_preparation_v1"
REGISTRATION = HERE / "registration.json"
OUTPUT = HERE / "result.json"
BASELINE = ROOT / "docs/style_improvement_20261005/shared_social_full_epochs_v1/post_training_execution.json"
BASELINE_SHA = "640b0cf7d6152f2be7ecf523eaa20bc5d135642e4ad75f851dfa5a92e213f19f"
PLAN = ROOT / "docs/style_improvement_20261008/shared_social_episode_after_recovery_v4/owner_plan.json"
PLAN_SHA = "8e8314cb35d40eee4b095ac6a05fd7601e92d57027ba56a0ba20e155e3f79c4d"
PINS = {
    Path(kernel.__file__): "76e589b05101480ac21ae229e7c8ecdefde049073670deaddb88a4306ed77a35",
    Path(strict.__file__): "f2cf03483f9bab988d9fb293093c644ef97b7cfeda32458e9ba0b2a93c40600a",
}
SCHEMA = "shared_social.fresh_episode2000_comparison_registration.v1"
COMPLETION = PLAN.parent / "completion.json"
RECEIPT = PLAN.parent / "training_execution.json"
STEPS = [*owner.evaluation.STEPS, "independent_evaluation"]
EXTENSIONS = {"episode_reference_experiment", "episode_after_recovery_execution"}


def require(condition, message):
    owner.release.recovery.require(condition, message)


def readonly(path):
    return owner.release.recovery.readonly_identity(path)


def registration_document():
    sources = {}
    for path, sha in PINS.items():
        sources[path.name] = kernel.record(path)
        require(sources[path.name]["sha256"] == sha, "Original metric/admission source changed")
    baseline = readonly(BASELINE)
    require(baseline["sha256"] == BASELINE_SHA, "Fixed same-budget S2 owner receipt changed")
    bundle, measurements = strict.admit_post(BASELINE)
    training = kernel.bound(bundle["tracking"]["training_execution_admission"]["result"])
    require(training["completed_updates"] == 2000 and training["actual_training_transitions"] == 16384000,
            "Retain the full same-budget fresh2000 baseline")
    plan = owner.admit_plan(PLAN)
    plan_identity = readonly(PLAN)
    require(plan_identity["sha256"] == PLAN_SHA, "Select the actual registered episode candidate owner plan")
    selected = plan["stages"][-1]
    require(selected["stage"] == "pilot2000" and selected["runtime"]["updates"] == 2000,
            "Retain the preselected full candidate budget")
    return {"schema": SCHEMA, "reader": kernel.record(__file__), "metric_sources": sources,
            "baseline_post_receipt": baseline, "baseline_measurements": measurements,
            "baseline_checkpoint": bundle["tracking"]["training_execution_admission"]["checkpoint"],
            "candidate_owner_plan": plan_identity, "candidate_completion_path": str(COMPLETION),
            "candidate_training_receipt_path": str(RECEIPT),
            "preselected_candidate_checkpoint": str(Path(selected["output_directory"]) / "step_002000.pt"),
            "comparison_output": str(OUTPUT), "training_updates_each": 2000,
            "physical_training_transitions_each": 16384000, "fixed_real_AMASS_training_clips": 2848,
            "validation_clips": 22, "complete_validation_control_steps_each": 11982,
            "reference_fitting_changed": False, "failures_filtered": False, "Test34_used": False,
            "unsupported_or_unmatched_speed_bins": "unknown; excluded from directional aggregate only",
            "coupled_experiment_changes": ["episode procedural reference ownership and buffer/RNG exposure",
                                           "explicit CPU/CUDA 30Hz clock rounding correction"],
            "single_variable_causal_experiment_claimed": False,
            "author_protocol_equivalence_claimed": False, "paper_effect_achieved": False,
            "default_model_promoted": False, "candidate_evaluation_completed_at_registration": False}


def admit_registration(path):
    path = Path(path).resolve()
    identity = readonly(path)
    registered = kernel.read(path)
    require(registered == registration_document() and kernel.record(path) == identity,
            "Comparison registration or fixed sources changed")
    return registered, identity


def validate_owner(document):
    require(document.get("schema") == "shared_social.episode_after_recovery_pipeline_owner_state.v1"
            and document.get("status") == "completed_training_and_independent_evaluation_quality_unclassified"
            and [row.get("name") for row in document.get("steps", [])] == STEPS
            and all(row.get("status") == "passed" and type(row.get("actual_exit_code")) is int
                    and row["actual_exit_code"] == 0 and row.get("exit_observation") == "subprocess.Popen.wait"
                    for row in document["steps"]),
            "Require all actual fresh episode training, CPU audit and independent evaluation owner exits")


def child_lineage_view(pipeline, base, full):
    """Verify both admissions, then expose the original child view privately."""
    require(set(full) - set(base) == EXTENSIONS
            and {key: value for key, value in full.items() if key not in EXTENSIONS} == base
            and pipeline.get("training_execution_admission") == full,
            "Parent/child admissions may differ only by the two exact verified candidate metadata extensions")
    view = deepcopy(pipeline)
    view["training_execution_admission"] = deepcopy(base)
    return view


def admit_candidate(registered):
    identity = readonly(COMPLETION)
    completed = kernel.read(COMPLETION)
    validate_owner(completed)
    require(completed.get("plan") == registered["candidate_owner_plan"], "Completed candidate owner selects another plan")
    plan = owner.admit_plan(PLAN)
    full_module = owner.evaluation.prepare_pipeline(plan["serial_release"]["path"])
    checkpoint, _, full = full_module.admit_training(RECEIPT)
    base_checkpoint, _, base = child_admission.admit_training(RECEIPT)
    require(checkpoint == base_checkpoint and str(checkpoint) == registered["preselected_candidate_checkpoint"],
            "Select only the actual preselected episode2000 terminal checkpoint")
    pipeline_identity = completed["steps"][-1]["result"]
    require(completed.get("evaluation_pipeline") == pipeline_identity, "Owner selects another evaluation pipeline")
    pipeline = kernel.bound(pipeline_identity)
    require(completed.get("measurements") == pipeline.get("measurements"), "Owner/pipeline measurements differ")
    bundle = {name: kernel.bound(completed["measurements"][name]) for name in strict.MEASUREMENTS}
    view = child_lineage_view(pipeline, base, full)
    strict.post_lineage(completed, view, bundle)
    for name, row in zip(STEPS, completed["steps"], strict=True):
        for key in ("result", "log"):
            require(kernel.record(row[key]["path"]) == row[key], "Completed actual child artifact changed: " + name)
    require(kernel.record(COMPLETION) == identity, "Episode completion changed during admission")
    return bundle, completed["measurements"], identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args(argv)
    registered, registration_identity = admit_registration(args.registration)
    if args.check_sources:
        print(json.dumps({"status": "episode_comparison_cold_preflight_passed", "registration": registration_identity,
                          "Torch_imported": "torch" in sys.modules, "SDK_started": False,
                          "candidate_evaluation_completed": False}))
        return 0
    require(not OUTPUT.exists(), "Use a fresh comparison output; preserve earlier comparisons")
    before, before_ids = strict.admit_post(BASELINE)
    after, after_ids, completion_identity = admit_candidate(registered)
    sources = strict.unique_artifacts([*before_ids.values(), *after_ids.values(),
                                      *strict.bundle_sources(before, before_ids),
                                      *strict.bundle_sources(after, after_ids)])
    for item in sources:
        require(kernel.record(item["path"]) == item, "Bound comparison source/capture/array changed")
    comparison = strict.supplemental_comparison(before, after)
    for item in sources:
        require(kernel.record(item["path"]) == item, "Comparison input changed during measurement")
    admit_registration(args.registration)
    require(kernel.record(COMPLETION) == completion_identity and "torch" not in sys.modules,
            "Candidate completion changed or cold comparison imported Torch")
    result = {"schema": "shared_social.completed_episode2000_independent_comparison.v1", "status": "passed_comparison",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "reader": registered["reader"],
              "registration": registration_identity, "candidate_completion": completion_identity,
              "baseline_post_receipt": registered["baseline_post_receipt"], "verified_artifacts": sources,
              "comparison": comparison, "metric_sources": registered["metric_sources"],
              "exposure_qualification": "Both fresh2000/16.384M; declared coupled reference/RNG/clock changes, no single-variable causal claim",
              "parent_child_admission_view": {"extensions_verified_separately": sorted(EXTENSIONS),
                  "original_input_files_modified": False, "metric_math_changed": False},
              "Torch_imported": False, "SDK_started": False, "paper_effect_achieved": False,
              "default_model_promoted": False}
    owner.io.write_json(OUTPUT, result, readonly=True)
    print(json.dumps({"status": result["status"], "output": kernel.record(OUTPUT), "paper_effect_achieved": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
