"""Correct only the trained-controller versus deployed-planner provenance ABI."""
from __future__ import annotations

import inspect
from pathlib import Path

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[2]
BASE_DIR = ROOT / "docs/style_improvement_20261003/default_newDP_physical_preparation_v1"
BASE_SHA = "30202b34a13b5756ffa5c9b41110103a913c3883ed1d88d8389546322a8fc05a"

# Import the frozen stdlib helper by explicit file, not a production/model module.
import importlib.util
spec = importlib.util.spec_from_file_location("default_DP_original_source_helper", BASE_DIR / "physical_evidence.py")
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
BASE.require(BASE.record(BASE_DIR / "physical_evidence.py")["sha256"] == BASE_SHA, "Frozen original source helper changed")

HERE, RUNS = BASE.HERE, BASE.RUNS
CPU_ENV, GPU_ENV = BASE.CPU_ENV, BASE.GPU_ENV
read, record, require, source_binding, registration = BASE.read, BASE.record, BASE.require, BASE.source_binding, BASE.registration

BEFORE = '''    require(result["provenance"]["diffusion_checkpoint"] == case["planner"]["path"]
            and result["provenance"]["diffusion_sha256"] == case["planner"]["sha256"],
            "Actual loaded Diffusion path/SHA differs")
'''
AFTER = '''    validate_runtime_planner(result, plan, phase)
'''


def validate_runtime_planner(result, plan, phase):
    """Producer preserves training provenance; evaluation_planner_change identifies the deployed DP."""
    selected = plan["cases"][phase]["planner"]
    historical = read(plan["historical_native"]["4"]["result"]["path"])["evaluation_actor"]["full_training_provenance"]
    require(result["provenance"] == historical and result["evaluation_actor"]["full_training_provenance"] == historical,
            "Entire legacy AMP training provenance must remain historically exact")
    require(result["arguments"]["diffusion_checkpoint"] == selected["path"] and record(selected["path"]) == selected,
            "Actual selected deployment DP path/full opaque bytes/SHA differs")
    change = result.get("evaluation_planner_change")
    if historical["diffusion_sha256"] != selected["sha256"]:
        require(result["arguments"]["allow_evaluation_planner_change"] is True and change == {
            "training_sha256": historical["diffusion_sha256"], "evaluation_sha256": selected["sha256"], "checkpoint": selected["path"]},
            "Actual producer evaluation_planner_change exact training/evaluation/path triple required")
    else:
        require(change is None, "Unchanged OLD planner must not invent a planner-change record")
    rows = [row for row in result["resource_preflight"]["records"] if row["name"] == "diffusion_checkpoint"]
    require(len(rows) == 1 and rows[0]["path"] == selected["path"] and rows[0]["bytes"] == selected["bytes"]
            and rows[0]["mtime_ns"] == Path(selected["path"]).stat().st_mtime_ns,
            "Actual resource header/stat must refer to the deployed DP, not controller training DP")
    runtime = result["runtime_execution"]
    require(result["trajectory_source"] == "diffusion" and runtime["diffusion_used_for_rollouts"] is True
            and runtime["diffusion_steps"] == 100 and runtime["diffusion_samples_per_actor"] == 1
            and runtime["precision"] == "FP32" and runtime["compiled_forward"] is False,
            "Actual original Diffusion rollout runtime boundary differs")
    return {"training_diffusion_checkpoint": historical["diffusion_checkpoint"], "training_diffusion_sha256": historical["diffusion_sha256"],
            "deployed_diffusion_checkpoint": selected, "evaluation_planner_change": change,
            "resource_preflight_is_header_stat_only_not_full_SHA_proof": True,
            "full_SHA_proof": "registered opaque model rehash plus actual root owner equal before/after binding",
            "full_legacy_training_provenance_exact": True, "checkpoint_deserialized_by_reader": False}


def corrected_admission_function():
    source = inspect.getsource(BASE.admit_capture)
    require(source.count(BEFORE) == 1, "Exactly one erroneous metadata guard expected")
    revised = source.replace(BEFORE, AFTER)
    require(revised.replace(AFTER, BEFORE) == source, "Only deployment/training metadata guard may change")
    namespace = {**BASE.__dict__, "validate_runtime_planner": validate_runtime_planner}
    exec(compile(revised, str(PREP / "physical_evidence_v2.py"), "exec"), namespace)
    return namespace["admit_capture"]


def admit_capture(plan, phase):
    admission = corrected_admission_function()(plan, phase)
    result = read(admission["result"]["path"])
    admission.update(schema="default_newDP.actual_capture_source_admission.v2",
                     runtime_planner_identity=validate_runtime_planner(result, plan, phase),
                     metadata_revision_helper=record(__file__))
    return admission


def revision():
    frozen = read(PREP / "source_freeze_v2.json")
    require(frozen["status"] == "frozen_CPU_only_revision", "Exact frozen metadata revision required")
    for row in frozen["files"]:
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222, "Metadata revision source changed")
    return read(PREP / "revision_registration.json")
