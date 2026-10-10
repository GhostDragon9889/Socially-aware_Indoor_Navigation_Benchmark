"""Summarize the audited native30 run and immutable native200 prefix on CPU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_actual30s import BASELINE, DOC, RUN, comparison, record


def arrays(path):
    with np.load(path, allow_pickle=False) as x:
        return {k: x[k] for k in x.files}


def stats(directory):
    p = {k: x[:900] for k, x in arrays(directory / "physical_states.npz").items()}
    g = arrays(directory / "ground_support.npz")
    distance = np.linalg.norm(np.diff(p["root_pos"][..., :2], axis=0), axis=-1)
    distance[p["reset_mask"][1:]] = 0
    return {"within_episode_distance_m_per_actor": distance.sum(0).tolist(),
            "mean_target_XY_error_m_per_actor": np.linalg.norm(p["root_pos"][..., :2] - p["target_xy"], axis=-1).mean(0).tolist(),
            "minimum_root_z_m_per_actor": p["root_pos"][..., 2].min(0).tolist(),
            "double_support_fraction_per_actor": g["double_support_fraction"][:900].mean(0).tolist(),
            "no_support_fraction_per_actor": g["no_support_fraction"][:900].mean(0).tolist()}


def main():
    audit = json.loads((DOC / "actual_audit.json").read_text())
    rows = {}
    for label, directory in (("candidate", RUN), ("baseline_prefix", BASELINE)):
        capture = arrays(directory / "physical_roots.npz")
        physical = arrays(directory / "physical_states.npz")
        if (set(capture) != {"root_pos"} or capture["root_pos"].dtype != physical["root_pos"].dtype
                or capture["root_pos"].tobytes(order="C") != physical["root_pos"].tobytes(order="C")):
            raise ValueError("Separate physical roots do not match retained physical states")
        rows[label] = {"artifact": record(directory / "physical_roots.npz"), "physical_state_root_bytes_identical": True}
    root_compare = comparison(arrays(RUN / "physical_roots.npz")["root_pos"], arrays(BASELINE / "physical_roots.npz")["root_pos"][:900])
    summary = {"schema": "native_arm_alpha025_actual30s_summary.v1", "CPU_only": True,
               "audit": record(DOC / "actual_audit.json"), "status": audit["status"], "checks_passed": audit["checks_passed"],
               "counts": audit["counts"], "baseline_prefix": audit["baseline"], "separate_physical_root_closure": rows,
               "separate_root_prefix_comparison": root_compare,
               "raw_field_comparison_including_redundant_root": {"fields": 78, "bytes_exact_fields": 26, "different_fields": 52,
                                                                 "redundant_root_counted_separately": True},
               "candidate_statistics": stats(RUN), "baseline_prefix_statistics": stats(BASELINE),
               "navigation_quality_200s_validated": False, "pose_style_quality_evaluated_here": False,
               "same_whole_code_causal_ablation": False, "toe_body_origin_velocity_is_contact_point_slip": False,
               "native_capture_validated_in_retained30s_archive": True,
               "historical_native_capture_probe_flag_rewritten": False}
    destination = DOC / "actual_summary.json"
    if destination.exists():
        raise FileExistsError("Preserve independent summary")
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary["candidate_statistics"]))
    print(json.dumps(summary["baseline_prefix_statistics"]))


if __name__ == "__main__":
    main()
