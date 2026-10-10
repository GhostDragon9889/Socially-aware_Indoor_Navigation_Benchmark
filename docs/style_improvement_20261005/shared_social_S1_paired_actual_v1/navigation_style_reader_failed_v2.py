#!/usr/bin/env python3
"""Evaluate complete paired navigation poses against the frozen AMASS reference.

This is a CPU diagnostic of actual closed-loop motion, including failures.
It reuses the existing pose/velocity metric and cluster basis. It does not
refit a basis, select successful episodes, or reproduce the author's scores.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis.paired_initial import compare_initial_conditions  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from CrowdSim.humanoid.style_evaluation import (  # noqa: E402
    BODY_NAMES, StyleSpec, assert_parent_split_disjoint, compare_speed_conditioned,
    gaussian_motion_mmd2, sequence_style_samples,
)
from scripts import evaluate_shared_social_style as frozen  # noqa: E402

MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
METRIC_SHA = "b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(identity):
    require(record(identity["path"]) == identity, "Recorded input changed: " + identity["path"])
    return json.loads(Path(identity["path"]).read_text())


def read_completed_run(owner_path):
    """Require external CPU and SDK exits, full native audit and sealed arrays."""
    identities = [record(owner_path)]
    owner = bound(identities[0])
    require(owner.get("schema") == "shared_social.actual_high_level_CPU_audit_execution.v1"
            and type(owner.get("actual_exit_code")) is int and owner["actual_exit_code"] == 0
            and isinstance(owner.get("terminal_tool_chunk"), str) and owner["terminal_tool_chunk"],
            "Require externally observed successful native CPU audit")
    audit = bound(owner["audit"])
    require(audit.get("schema") == "shared_social.all_registered_controls_native_CPU_audit.v1"
            and audit.get("status") == "passed" and audit["actual_controls"] == 6000
            and audit["actor_controls"] == 24000 and audit["native_actor_substep_samples"] == 96000
            and audit["all_failure_contact_and_reset_frames_retained"] is True
            and audit["recorded_body_pose"]["all24_same_control_native_body_positions_equal"] is True
            and audit["initial_capture_link"]["status"] == "live_initial_capture_linked",
            "Require complete200s native body audit with failures and live initial capture")
    receipt = bound(audit["execution_receipt"])
    require(receipt.get("schema") == "shared_social.actual_navigation200s_execution.v1"
            and type(receipt.get("actual_exit_code")) is int and receipt["actual_exit_code"] == 0
            and isinstance(receipt.get("terminal_tool_chunk"), str) and receipt["terminal_tool_chunk"]
            and receipt["result"] == audit["result"], "Require this run's actual successful SDK exit")
    result, base = bound(audit["result"]), bound(audit["base_audit"])
    require(base["status"] == "passed" and base["result"] == audit["result"]
            and base["execution_receipt"] == audit["execution_receipt"]
            and result["status"] == "passed" and result["cleanup_succeeded"] is True
            and result["cleanup_errors"] == [] and result["actual_control_steps"] == 6000
            and result["physics"]["control_hz"] == 30 and result["physics"]["physics_hz"] == 120
            and Path(result["arguments"]["manifest"]).resolve() == MANIFEST,
            "Complete native result, CPU audit and fixed training manifest must agree")
    arrays = base["arrays"]
    run = Path(audit["result"]["path"]).parent
    require([Path(item["path"]) for item in arrays] ==
            [run / name for name in ("physical_states.npz", "ground_support.npz", "reference_plans.npz")],
            "CPU audit arrays belong to another run")
    for identity in arrays:
        path = Path(identity["path"])
        require(record(path) == identity and not path.is_symlink() and path.stat().st_nlink == 1
                and not path.stat().st_mode & 0o222, "Completed raw arrays must be sealed and unchanged")
    initial = result["paired_initial_condition"]
    require(audit["initial_capture_link"]["document"] == initial["document"]
            and audit["initial_capture_link"]["arrays"] == initial["arrays"],
            "Native audit's initial capture belongs to another run")
    identities.extend([owner["audit"], audit["execution_receipt"], audit["result"], audit["base_audit"],
                       audit["reader"], *audit["original_sources"], initial["document"], initial["arrays"], *arrays])
    with np.load(arrays[0]["path"], allow_pickle=False) as physical, np.load(arrays[1]["path"], allow_pickle=False) as ground:
        data = {name: physical[name] for name in ("body_pos", "root_rot", "episode_ids", "reset_mask", "terminated")}
        names = ground["common_body_names"].tolist()
        require(names == result["state_capture"]["body_names"] and set(names) == set(BODY_NAMES)
                and np.array_equal(data["body_pos"], ground["body_link_pos_w"][:, -1])
                and np.array_equal(data["root_rot"], ground["body_link_quat_w"][:, -1, :, names.index("Pelvis")])
                and np.array_equal(data["episode_ids"], ground["episode_ids"])
                and np.array_equal(data["reset_mask"], ground["reset_mask"]),
                "Style body/root quaternion/episode capture differs from native last substep")
    samples = sequence_style_samples(data["body_pos"], data["root_rot"], body_names=names,
        episode_ids=data["episode_ids"], reset_mask=data["reset_mask"], spec=StyleSpec())
    require(samples["summary"]["same_episode_transition_count"] == base["same_episode_position_intervals"],
            "Style transition denominator differs from full physical audit")
    terminal_kept = int(data["terminated"][samples["end_frame_indices"], samples["actor_indices"]].sum())
    return samples, result, base, {"owner": identities[0], "audit": owner["audit"], "arrays": arrays,
        "all_capture_frames": 24000, "fall_termination_frames": int(data["terminated"].sum()),
        "fall_termination_interval_endpoints_kept": terminal_kept,
        "all_failure_frames_retained": True, "filter_applied": False}, identities


def matched_motion_bins(runs, reference, *, spec=StyleSpec(), minimum=30, cap=256):
    """Use equal sample counts and identical reference indices for both arms."""
    require(len(runs) == 2 and type(minimum) is int and 1 <= minimum <= cap,
            "Matched style comparison requires two arms and valid sample limits")
    rows = []
    edges = (*spec.speed_bin_lower_edges_m_s, np.inf)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        indices = {name: np.flatnonzero((value["speed_m_s"] >= lower) & (value["speed_m_s"] < upper))
                   for name, value in runs.items()}
        reference_pool = np.flatnonzero((reference["speed_m_s"] >= lower) & (reference["speed_m_s"] < upper))
        n = min(cap, len(reference_pool), *(len(value) for value in indices.values()))
        row = {"lower_m_s": float(lower), "upper_m_s": float(upper) if np.isfinite(upper) else None,
               "generated_counts": {name: len(value) for name, value in indices.items()},
               "reference_count": len(reference_pool), "sample_count_each": n,
               "status": "supported" if n >= minimum else "unknown_insufficient_samples", "scores": None}
        if n >= minimum:
            ri = reference_pool[np.linspace(0, len(reference_pool) - 1, n, dtype=np.int64)]
            rf = np.concatenate((reference["local_pose"][ri],
                                 spec.motion_velocity_time_scale_s * reference["local_pose_velocity"][ri]), -1)
            row["reference_sample_indices"] = ri.tolist()
            row["scores"] = {}
            for name, pool in indices.items():
                gi = pool[np.linspace(0, len(pool) - 1, n, dtype=np.int64)]
                value = runs[name]
                gf = np.concatenate((value["local_pose"][gi],
                                     spec.motion_velocity_time_scale_s * value["local_pose_velocity"][gi]), -1)
                row["scores"][name] = {"MMD2": gaussian_motion_mmd2(gf, rf, sigma=spec.motion_kernel_sigma),
                                       "generated_sample_indices": gi.tolist()}
        rows.append(row)
    supported = [row for row in rows if row["scores"] is not None]
    return {"bins": rows, "common_supported_bin_count": len(supported),
            "equal_common_bin_mean_MMD2": {name: float(np.mean([row["scores"][name]["MMD2"] for row in supported]))
                if supported else None for name in runs}, "minimum_samples_each": minimum, "cap_each": cap,
            "reference_indices_identical_for_both_arms": True, "metric_spec": asdict(spec),
            "unknown_bins_omitted_from_mean": True}


def load_frozen_reference():
    sources = [(frozen.TOOL, frozen.TOOL_SHA), (frozen.REFERENCE, frozen.REFERENCE_SHA),
               (frozen.POSE_TOOL, frozen.POSE_TOOL_SHA), (frozen.BASIS, frozen.BASIS_SHA),
               (frozen.BASIS_EVALUATION, frozen.BASIS_EVALUATION_SHA),
               (MANIFEST, MANIFEST_SHA), (ROOT / "CrowdSim/humanoid/style_evaluation.py", METRIC_SHA)]
    identities = [record(path) for path, _ in sources]
    require(all(identity["sha256"] == sha for identity, (_, sha) in zip(identities, sources, strict=True)),
            "Frozen reference, metric, basis or fixed2848 manifest changed")
    modules = []
    for name, path in (("navigation_reference", frozen.TOOL), ("navigation_pose45", frozen.POSE_TOOL)):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    tool, pose = modules
    artifact = json.loads(frozen.REFERENCE.read_text())
    require(artifact["schema"] == "navisaaclab.heldout_style_reference.v1"
            and artifact["selection"]["split"] == "validation"
            and artifact["metric_spec"] == json.loads(json.dumps(asdict(StyleSpec()))),
            "Only frozen validation selection with the original metric spec is admitted")
    document = bound(artifact["manifest"])
    training = json.loads(MANIFEST.read_text())
    require(len(training["splits"]["train"]["clips"]) == 2848, "Require fixed2848 training")
    assert_parent_split_disjoint(document)
    disjoint = tool.assert_heldout_disjoint(artifact["selection"], training, document)
    body, rotation, hashes, provenance = tool.load_reference(Path(artifact["manifest"]["path"]), artifact["selection"])
    require(hashes == artifact["actual_selected_array_sha256"], "Frozen held-out reference arrays changed")
    resets = np.zeros(body.shape[:2], dtype=bool)
    resets[0] = True
    samples = sequence_style_samples(body, rotation, episode_ids=np.ones(body.shape[:2], dtype=np.int64),
                                     reset_mask=resets, spec=StyleSpec())
    fit = json.loads(frozen.BASIS_EVALUATION.read_text())["method"]["fit"]
    require(len(samples["local_pose"]) == fit["available_reference_poses"], "Frozen reference pose population changed")
    with np.load(frozen.BASIS, allow_pickle=False) as saved:
        require(set(saved.files) == {"centers_50", "centers_100"}, "Unexpected cluster basis")
        centers = {k: saved[f"centers_{k}"] for k in (50, 100)}
    for k, value in centers.items():
        require(value.shape == (k, 45) and value.dtype == np.float64 and np.isfinite(value).all()
                and pose.array_sha(value) == fit["centers"][str(k)]["float64_sha256"], "Saved cluster centers changed")
    return samples, pose, centers, {"selection_artifact": identities[1], "loaded_provenance": provenance,
        "actual_array_hashes": hashes, "parent_disjoint": disjoint, "Test34_read": False}, identities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--diffusion-audit-execution", type=Path, required=True)
    parser.add_argument("--sfm-audit-execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use CUDA_VISIBLE_DEVICES='' for CPU style analysis")
    require(not args.output.exists(), "Use a fresh output; preserve historical measurements")
    sources = [record(__file__), record(Path(frozen.__file__))]
    runs, results, base_audits, admission, identities = {}, {}, {}, {}, list(sources)
    for name, path in (("diffusion", args.diffusion_audit_execution), ("sfm", args.sfm_audit_execution)):
        runs[name], results[name], base_audits[name], admission[name], inputs = read_completed_run(path)
        require(results[name]["trajectory_source"] == name, "Audit belongs to the wrong planner arm")
        identities.extend(inputs)
    initial = compare_initial_conditions(*(Path(results[name]["paired_initial_condition"]["document"]["path"])
                                          for name in ("diffusion", "sfm")))
    reference, pose, centers, reference_admission, reference_inputs = load_frozen_reference()
    identities.extend(reference_inputs)
    report = {"schema": "shared_social.paired_navigation_inclusive_style.v1", "status": "passed_CPU_measurement",
        "sources": sources, "initial_pairing": initial, "reference": reference_admission,
        "runs": {name: {"admission": admission[name], "inclusive_summary": value["summary"],
            "pose45": pose.compare_pose_only(value["local_pose"], reference["local_pose"], centers),
            "style90_all_population": compare_speed_conditioned(value, reference),
            "physical_route_audit": base_audits[name]} for name, value in runs.items()},
        "style90_matched_pair": matched_motion_bins(runs, reference),
        "basis_refit": False, "metric_math_changed": False, "failure_filter_applied": False,
        "motion_capture_scope": "all post-control frames; reset jumps excluded from derivatives only; failed episodes retained",
        "numerical_comparison_to_paper_valid": False, "paper_effect_achieved": False}
    require(all(record(identity["path"]) == identity for identity in identities), "Inputs changed during style measurement")
    import torch
    require(not torch.cuda.is_initialized(), "CPU style measurement unexpectedly initialized CUDA")
    report["CUDA_initialized"] = False
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "style90_matched_pair": report["style90_matched_pair"]["equal_common_bin_mean_MMD2"],
                      "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
