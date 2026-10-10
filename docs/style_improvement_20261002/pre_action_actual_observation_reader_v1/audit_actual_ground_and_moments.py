"""Complete actual native force/COM audit, preserving its frozen math gates."""

from pathlib import Path
import hashlib
import importlib.util
import json
import os

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = ROOT / "output/diffusion_amp/pre_action_baseline_cuboid_30s_v1"


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only environment required")
    core = json.loads((HERE / "actual_cuboid30_v2/report.json").read_text())
    helper_path = (
        ROOT / "docs/style_improvement_20261002/audit_ground_support_capture.py"
    )
    helper_sha = hashlib.sha256(helper_path.read_bytes()).hexdigest()
    if helper_sha != "f851e7d5f90d48d830c564fdc2b2e446dfe35daa36ec636aba6f946ec7752c3f":
        raise ValueError("Frozen actual native measurement auditor changed")
    spec = importlib.util.spec_from_file_location(
        "frozen_ground_measurement_auditor", helper_path
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original_gate = helper.verify_sources
    try:

        def source_gate(audit, result, **kwargs):
            audit.require(
                core["status"] == "passed" and core["resources"]["equal"],
                "Independent source177/resource admission passed",
            )
            return {
                "source177_archive": core["new_source177"],
                "resources_full_sha256_recomputed": core["resources"]["candidate"],
                "source_admission_callback_only": True,
                "actual_runtime_per_file_attestation": False,
                "scope": "Adapt source admission only; frozen native measurement math/threshold/COM/counter gates unchanged",
            }

        helper.verify_sources = source_gate
        audit = helper.Audit()
        native = helper.audit_capture(CANDIDATE, audit)
        native.update(
            status="passed", checks=audit.checks, auditor_source_sha256=helper_sha
        )
    finally:
        helper.verify_sources = original_gate
    stats = []
    for path in sorted((CANDIDATE / "pre_action_observations").glob("call_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            std = np.sqrt(data["task_variance"])
            record = {
                "call_index": int(path.stem.rsplit("_", 1)[1]),
                "actual_count": float(data["task_count"]),
                "trajectory_prefix_std_quantiles_min50max": np.quantile(
                    std[:20], [0, 0.5, 1]
                ).tolist(),
                "terrain_suffix_std_quantiles_min50max": np.quantile(
                    std[20:], [0, 0.5, 1]
                ).tolist(),
                "prefix_clipped_per_actor": data["task_clip_mask"][:, :20]
                .sum(-1)
                .tolist(),
                "terrain_clipped_per_actor": data["task_clip_mask"][:, 20:]
                .sum(-1)
                .tolist(),
            }
            stats.append(record)
    result = json.loads((CANDIDATE / "result.json").read_text())
    supplement = {
        "schema": "actual_pre_action_native_moments_supplement.v1",
        "CPU_only": True,
        "SDK_started": False,
        "GPU_used": False,
        "core_report_sha256": hashlib.sha256(
            (HERE / "actual_cuboid30_v2/report.json").read_bytes()
        ).hexdigest(),
        "actual_result_outcomes": {
            k: result[k]
            for k in (
                "actual_control_steps",
                "fall_resets",
                "time_limit_resets",
                "route_vertices_reached",
                "elapsed_s",
            )
        },
        "actual_native_ground_audit": native,
        "moments_and_clips_per_recorded_call": stats,
        "terrain_input_source_semantics": "PACER task heights=(center_height-measured_absolute_worldZ).clamp(-3,3)*5; actual raw -10 is a -2m relative obstacle height feature, not -10m ground Z",
        "sampling_denominator": "30 stride30 pre-action calls ×4 actors=120 sampled actorframes; not all3600 control actorframes",
        "root_cause_claim": None,
        "native_engine_time_observed": False,
        "source177_vs168_pair_scope": "Actual full900 arrays exactly equal, despite selected source differences; no general bitwise engine guarantee",
    }
    destination = HERE / "actual_cuboid30_v2/native_and_moments.json"
    with destination.open("x") as stream:
        json.dump(supplement, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": "passed",
                "native_checks": len(audit.checks),
                "counts": native["counts"],
                "maximum_COM_error": native["COM_statistics"][
                    "maximum_independent_composition_error_m"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
