"""Pre-register local native paired seeds without inventing paper task IDs.

All original resource cases, including the four spacing rejections, remain
visible. Completed SDK/CPU/style receipts are still required after training.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import evaluate_shared_social_native_seeded as entry

ROOT = entry.ROOT
SEEDS = (7, 19, 42)
SCENES = ("Office", "Warehouse", "Full_Warehouse", "Hospital")
BASE = ROOT / "docs/style_improvement_20261005"
PINS = (
    (BASE / "shared_social_native_SFM_preparation_v1/resource_cohort_CPU.json",
     "3385701a74fe120d53ba540402e5257e6748653b6112b102c4ac872b982b72ec"),
    (BASE / "shared_social_native_SFM_preparation_v1/resource_cohort_registration.json",
     "252a206db19febea8d965022390c09620b090f7db19ec6c2f6b8ac5eed88e570"),
    (BASE / "shared_social_budget12500_v1/registration.json",
     "6cc950072a30f0f0ff58951afeb1e59f51ef74a0516dd127ddd26062627d9daa"),
)


def load_registered_inputs():
    records, documents = [], []
    for path, sha in PINS:
        row = entry.record(path)
        if row["sha256"] != sha or path.is_symlink() or path.stat().st_nlink != 1 or path.stat().st_mode & 0o222:
            raise ValueError("Require the exact readonly original native resource/S3 registration")
        records.append(row)
        documents.append(json.loads(path.read_text()))
    return documents, records


def validate_cohort(cohort, registration):
    expected = [[scene, actors] for scene in SCENES for actors in range(3, 16)]
    if (registration["case_order"] != expected or cohort["status"] != "CPU_resource_case_review_complete"
            or [[row["scene"], row["actors"]] for row in cohort["cases"]] != expected
            or cohort["admitted_cases"] != 48 or cohort["rejected_cases"] != 4
            or registration["minimum_spawn_separation_m"] != 1.6
            or registration["route_clearance_m"] != .8 or registration["minimum_spawn_clearance_m"] != 3):
        raise ValueError("Preserve all52 original native cases/order and spacing/route constraints")
    for row in cohort["cases"]:
        rejected = row["scene"] == "Hospital" and row["actors"] >= 12
        if row["status"] != ("rejected" if rejected else "CPU_resource_admitted"):
            raise ValueError("Native resource case status changed")
        if rejected and (row["error"]["type"] != "ValueError"
                or "below the declared minimum1.6m; no fallback/resampling" not in row["error"]["message"]
                or row["clearance_or_spacing_reduced"] is not False or row["replacement_scene_or_resampling"] is not False):
            raise ValueError("Retain original Hospital resource refusal without replacement or resampling")


def build_registration(output_root):
    from CrowdSim.humanoid.application.native_deployment import preflight_native_scene
    entry.source_bindings()
    documents, inputs = load_registered_inputs()
    cohort, original, training = documents
    validate_cohort(cohort, original)
    checkpoint = Path(training["checkpoint_preselected"])
    if (training["updates"] != 12500 or training["seed"] != 7 or training["fresh_initialization"] is not True
            or checkpoint != ROOT / "output/diffusion_amp/shared_social_budget12500_v1/step_012500.pt"
            or training["training_manifest"]["sha256"] != "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"):
        raise ValueError("Native matrix must use the preselected fixed2848 S3 terminal candidate")
    output_root = Path(output_root).resolve()
    cases, resources, supported_actors = [], [], 0
    for row in cohort["cases"]:
        resource = None
        if row["status"] == "CPU_resource_admitted":
            for key in ("configuration", "proof"):
                identity = row[key]
                if entry.record(identity["path"]) != identity:
                    raise ValueError("Original admitted native configuration/proof changed")
                inputs.append(identity)
            resource = preflight_native_scene(row["configuration"]["path"], actors=row["actors"])
            if (resource["configuration_file"] != row["configuration"]
                    or resource["configuration"]["scene"] != row["scene"]):
                raise ValueError("Re-admitted native scene differs from its original resource cohort")
            resources.append({"scene": row["scene"], "actors": row["actors"], "actual_CPU_request": resource})
        for seed in SEEDS:
            entry.validate_seed(seed)
            case = {"case_id": f"{row['scene']}_N{row['actors']:02}_seed{seed}",
                    "scene": row["scene"], "actors": row["actors"], "seed": seed, "control_steps": 6000,
                    "duration_s": 200, "source_CPU_case_status": row["status"],
                    "status": "pending_actual_training_and_native_SDK" if resource else "resource_rejected",
                    "same_AMP_checkpoint_for_both_arms": str(checkpoint), "arms": [],
                    "actual_SDK_exit_observed": False, "actual_initial_pairing_verified": False,
                    "actual_raw_style_or_quality_evaluated": False}
            if resource:
                case["scene_configuration"] = row["configuration"]
                case["paired_exposure_actor_controls"] = 2 * 6000 * row["actors"]
                case["arms"] = [{"planner": planner, "seed": seed,
                    "output": str(output_root / case["case_id"] / planner), "control_steps": 6000,
                    "actual_process_exit_code": None, "actual_result_identity": None}
                    for planner in ("diffusion", "sfm")]
                supported_actors += row["actors"]
            else:
                case["original_resource_refusal"] = row["error"]
                case["clearance_or_spacing_reduced"] = False
                case["replacement_scene_or_resampling"] = False
            cases.append(case)
    if any(entry.record(row["path"]) != row for row in inputs):
        raise ValueError("Native matrix registration inputs changed")
    return {"schema": "shared_social.local_native_multiseed_registration.v1",
        "status": "registered_CPU_resources_actual_execution_pending", "sources": entry.source_bindings(),
        "protocol_source": entry.record(__file__), "inputs": inputs, "seeds": list(SEEDS), "scenes": list(SCENES),
        "case_order": "Original scene-major N3..15 order, then seed7/19/42; both DP/SFM arms per admitted cohort",
        "cases": cases, "CPU_resources": resources, "registered_cohorts": len(cases),
        "resource_admitted_cohorts": sum(bool(row["arms"]) for row in cases),
        "resource_rejected_cohorts": sum(not row["arms"] for row in cases),
        "planned_full200s_arm_runs_after_all_gates": sum(len(row["arms"]) for row in cases),
        "resource_rejected_arm_slots": 2 * sum(not row["arms"] for row in cases),
        "planned_paired_actor_controls": 2 * 6000 * supported_actors,
        "preselected_S3_terminal_checkpoint": str(checkpoint), "actual_checkpoint_identity": None,
        "actual_training_exit_admitted": False, "same_initial_body_map_RNG_required": True,
        "native_seeded_raw_and_style_admission_still_required": True,
        "no_best_or_intermediate_checkpoint_selection": True, "GPU_or_SDK_started": False,
        "author_eight_scene_tasks_or_seeds_recovered": False, "paper_effect_achieved": False,
        "default_model_promoted": False}
