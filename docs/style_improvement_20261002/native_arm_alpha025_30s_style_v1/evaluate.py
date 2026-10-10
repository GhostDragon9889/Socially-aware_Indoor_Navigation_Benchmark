"""CPU same900 comparison, frozen90D features and frozen45D AMASS centers."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
REF = ROOT / "docs/style_improvement_20261002/backward_penalty_style_v1/heldout_reference.json"
REF_SHA = "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"
FROZEN = {
    "scripts/evaluate_pedestrian_style.py": "82ee083fd16d25ddc39a4c7541d884dfef13a34bb68abc99ba3d45195665f26f",
    "CrowdSim/humanoid/style_evaluation.py": "b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2",
    "docs/style_improvement_20261002/pose_diversity_candidate_v1/evaluate_pose_diversity.py": "208ff775d765b7d046e7e7e8c331dbb2b8d3bfdc4c1567c1df499a584af30bfa",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU only")
    destination = HERE / "evaluation.json"
    require(not destination.exists(), "Preserve frozen prior evidence")
    from scripts.evaluate_pedestrian_style import record, load_reference, assert_heldout_disjoint, subset
    from scripts.evaluate_paper_pedestrians import read_capture
    from CrowdSim.humanoid.style_evaluation import StyleSpec, BODY_NAMES, sequence_style_samples, compare_speed_conditioned
    import torch
    torch.set_num_threads(4)
    for path, sha in FROZEN.items():
        require(record(ROOT / path)["sha256"] == sha, "Frozen source changed: " + path)
    require(record(REF)["sha256"] == REF_SHA, "Frozen reference changed")
    artifact = json.loads(REF.read_text())
    require(artifact["selection"]["split"] == "validation", "No test captures allowed")
    spec = StyleSpec()
    require(artifact["metric_spec"] == json.loads(json.dumps(asdict(spec))), "Frozen specification changed")
    manifest = Path(artifact["manifest"]["path"])
    require(record(manifest) == artifact["manifest"], "Reference manifest changed")
    reference_document = json.loads(manifest.read_text())
    body, rotation, hashes, provenance = load_reference(manifest, artifact["selection"])
    require(hashes == artifact["actual_selected_array_sha256"], "Reference actual arrays changed")
    episodes = np.ones(body.shape[:2], dtype=np.int64)
    resets = np.zeros(body.shape[:2], dtype=bool)
    resets[0] = True
    reference_pose = sequence_style_samples(body, rotation, episode_ids=episodes, reset_mask=resets, spec=spec)
    pose_module = load("frozen_pose45_candidate", ROOT / "docs/style_improvement_20261002/pose_diversity_candidate_v1/evaluate_pose_diversity.py")
    pose_report_path = ROOT / "docs/style_improvement_20261002/pose_diversity_candidate_v1/actual_evaluation_v1/evaluation.json"
    pose_report = json.loads(pose_report_path.read_text())
    require(pose_report["source"]["reference_artifact"]["sha256"] == REF_SHA, "Pose45 reference changed")
    basis_path = pose_report_path.with_name("reference_basis.npz")
    with np.load(basis_path, allow_pickle=False) as archive:
        centers = {k: archive[f"centers_{k}"] for k in (50, 100)}
    for k, value in centers.items():
        require(value.shape == (k, 45) and value.dtype == np.float64 and np.isfinite(value).all() and
                pose_module.array_sha(value) == pose_report["method"]["fit"]["centers"][str(k)]["float64_sha256"], "Frozen AMASS centers changed")
    fit = pose_report["method"]["fit"]
    reference_indices = np.asarray(fit["fit_indices"], dtype=np.int64)
    require(pose_module.array_sha(reference_pose["local_pose"][reference_indices].astype(np.float64)) == fit["fit_samples_sha256"], "Frozen reference fit samples changed")
    audit_path = ROOT / "docs/style_improvement_20261002/native_arm_alpha025_30s_actual_review_v1/actual_audit.json"
    summary_path = audit_path.with_name("actual_summary.json")
    audit, summary = json.loads(audit_path.read_text()), json.loads(summary_path.read_text())
    require(summary["audit"] == record(audit_path) and audit["checks_passed"] == 891, "Native actual admission changed")
    require(audit["arm_blend"]["alpha"] == .25, "Actual alpha differs")
    runs = {
        "native_baseline_source168_prefix900": ROOT / "output/diffusion_amp/native_warehouse_clearance08_200s_v1",
        "native_arm_alpha025_source171_900": ROOT / "output/diffusion_amp/native_warehouse_arm_alpha025_30s_v1",
    }
    report = {"schema": "native_arm_style.same900_frozen90D_and_pose45.v1", "status": "passed", "CPU_only": True,
        "scope": "actual30s arm candidate vs actual200s native baseline first900; not cuboid, not whole-code single-variable",
        "controls_per_run": 900, "actors_per_run": 4, "seconds_per_run": 30, "no_test34_captures_read": True,
        "source": {"analysis_script": record(__file__), "reference": record(REF), "frozen_sources": {p: record(ROOT/p) for p in FROZEN},
            "native_actual_audit": record(audit_path), "native_actual_summary": record(summary_path), "pose45_basis": record(basis_path), "pose45_prior_report": record(pose_report_path)},
        "reference": artifact, "reference_load_provenance": provenance, "pose45_fit": fit,
        "models_already_fully_hashed_by_native_actual_audit": audit["arm_weights"], "source_epoch_comparison": audit["source_epoch_comparison"],
        "all_retained900_raw_frames_read": True, "failure_or_outlier_filter": False,
        "runs": {}}
    for index, (name, directory) in enumerate(runs.items()):
        arrays, runner, validation = read_capture(directory / "physical_states.npz", directory / "result.json")
        require(runner.get("error") is None and runner["actual_control_steps"] == (6000 if index == 0 else 900) and arrays["body_pos"].shape[:2] == ((6000 if index == 0 else 900), 4), "Completed actual capture size changed")
        declared = Path(runner["state_capture"]["path"])
        require((ROOT/declared if not declared.is_absolute() else declared).resolve() == directory/"physical_states.npz", "Capture path declaration changed")
        # Only temporal restriction; no failure/upright/speed/outlier removal.
        prefix = {k: v[:900] for k, v in arrays.items()}
        masses, names = runner["physics"]["live_body_masses_kg"], runner["physics"]["body_names"]
        require(len(masses) == 24 and set(names) == set(BODY_NAMES), "Canonical live mass mapping changed")
        reference_masses = [masses[names.index(n)] for n in BODY_NAMES]
        generated_masses = [masses[names.index(n)] for n in validation["body_names"]]
        generated = sequence_style_samples(prefix["body_pos"], prefix["root_rot"], body_names=validation["body_names"],
            episode_ids=prefix["episode_ids"], reset_mask=prefix["reset_mask"], spec=spec, body_masses_kg=generated_masses)
        reference = sequence_style_samples(body, rotation, episode_ids=episodes, reset_mask=resets, spec=spec, body_masses_kg=reference_masses)
        train_path = Path(runner["arguments"]["manifest"])
        train_path = ROOT/train_path if not train_path.is_absolute() else train_path
        disjoint = assert_heldout_disjoint(artifact["selection"], json.loads(train_path.read_text()), reference_document)
        inclusive = compare_speed_conditioned(generated, reference, spec=spec, minimum_bin_samples=30, maximum_motion_samples=256)
        gm = (generated["metrics"]["root_up_z"] >= .8) & (generated["metrics"]["pelvis_height_m"] >= .7)
        rm = (reference["metrics"]["root_up_z"] >= .8) & (reference["metrics"]["pelvis_height_m"] >= .7)
        native_outcomes = summary["baseline_prefix"] if index == 0 else summary["counts"]
        row = {"sources": {"states": record(directory/"physical_states.npz"), "result": record(directory/"result.json"), "training_manifest": record(train_path)},
            "raw_temporal_slice": "first900 control frames, half-open [0,900)", "prefix_actual_array_sha256": {k: pose_module.array_sha(v) for k, v in prefix.items()},
            "capture_validation_full_source": validation, "generated": generated["summary"], "reference": reference["summary"],
            "reference_parent_disjointness": disjoint, "inclusive_speed_conditioned90D": inclusive,
            "upright_secondary": {"retained_fraction": float(gm.mean()), "comparison": compare_speed_conditioned(subset(generated, gm), subset(reference, rm), spec=spec, minimum_bin_samples=30, maximum_motion_samples=256)},
            "pose45": pose_module.compare_pose_only(generated["local_pose"], reference_pose["local_pose"], centers),
            "raw_actor_frames": 3600, "included_same_episode_pose_endpoints": len(generated["local_pose"]),
            "excluded_first_frame_actor_samples": 4, "excluded_cross_reset_endpoints": generated["summary"]["excluded_reset_transitions"],
            "native_outcomes_from_independent_prefix_audit": native_outcomes,
            "native_descriptive_statistics": summary["baseline_prefix_statistics" if index == 0 else "candidate_statistics"]}
        report["runs"][name] = row
    report["limits"] = ["Inclusive90D uses unchanged physical-speed bins/min30/sigma1/cap256; legal same-episode endpoints only. No fall or upright exclusion for primary results.",
        "Pose45 reuses unchanged frozen local_pose and prior AMASS-only centers; no generated fit. Gaussian MMD2 sigma1/cap2048; not an author TableIII score.",
        "Prefix900 vs full900 is the comparison window; no direct30s-versus200s MMD comparison.",
        "Source168 and171 differ; oldV5 branch compatibility CPU evidence is not whole-code equality or a sole-factor causal proof.",
        "Both raw histories remain intact; this analysis creates no synthetic rollout or trimmed replacement result files.",
        "Arm blend has24 rawmean coordinates at equal observations; other coordinates need not match across differing trajectories/observations.",
        "Native foot/COM/contact counts are taken from the existing891-check independent audit; body-origin support velocities are proxies, not contact-point slip.",
        "No200s candidate acceptance/default promotion/paper reproduction claim."]
    require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({n: {"inclusive90D_MMD2": r["inclusive_speed_conditioned90D"]["equal_supported_bin_mean_pose_velocity_mmd2"],
        "speed_W1": r["inclusive_speed_conditioned90D"]["physical_speed_distribution_wasserstein_m_s"], "pose45_MMD2": r["pose45"]["pose_only_biased_gaussian_MMD2"]["value"],
        "included_pose_endpoints": r["included_same_episode_pose_endpoints"]} for n, r in report["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
