#!/usr/bin/env python3
"""Check spherical-joint position parameters in an admitted captured batch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis.joint_velocity_consistency import require  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402

VELOCITY = ROOT / "docs/style_improvement_20261008/shared_social_episode500_joint_velocity_CPU_v1/result.json"
VELOCITY_SHA = "6478fa7a0615e54029b9fc09a00efb41469ba135cd233316cd607eadfe314251"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    here = args.output.resolve()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not here.exists(), "Fresh CPU-only output required")
    velocity_record = identity(VELOCITY)
    require(velocity_record["sha256"] == VELOCITY_SHA and not VELOCITY.stat().st_mode & 0o222,
            "Require the preserved actual update500 batch-coordinate audit")
    velocity = json.loads(VELOCITY.read_text())
    require(velocity["status"] == "passed_sameinstant_batch_coordinate_audit"
            and velocity["selected_archive_update"] == 500 and velocity["physical_actor_controls"] == 8192
            and velocity["phase_verification"]["any_metrics_rows_removed"] is False,
            "Require the complete actual intermediate batch and original clock/reset checks")
    inputs = [velocity_record, *velocity["inputs"]]
    sources = [identity(path) for path in (Path(__file__), ROOT / "CrowdSim/humanoid/analysis/joint_velocity_consistency.py",
                                         ROOT / "scripts/diagnose_shared_social_training.py")]
    require(all(identity(row["path"]) == row for row in inputs + velocity["sources"]),
            "Admitted archive or velocity-audit sources changed")
    import torch

    packet = torch.load(inputs[1]["path"], weights_only=True, map_location="cpu", mmap=True)
    diagnostics = packet["collection"]["diagnostics"]
    metadata = packet["collection"]["metadata"]
    require(metadata["training_update_count"] == 499 and metadata["controls"] == 32 and metadata["actors"] == 256,
            "Actual packet counter differs")
    quaternion = diagnostics["common_rigid_body_rot"].numpy().astype(np.float64)
    joint_position = diagnostics["common_dof_pos"].numpy().astype(np.float64)
    require(quaternion.shape == (32, 256, 24, 4) and joint_position.shape == (32, 256, 69)
            and np.isfinite(quaternion).all() and np.isfinite(joint_position).all()
            and np.all(np.abs(np.linalg.norm(quaternion, axis=-1) - 1.) <= 1e-3), "Captured position/rotation arrays differ")
    parents = velocity["nonroot_parent_indices"]
    child_rotation = Rotation.from_quat(quaternion[..., 1:, :].reshape(-1, 4))
    parent_rotation = Rotation.from_quat(quaternion[..., parents, :].reshape(-1, 4))
    coordinate_rotation = Rotation.from_rotvec(joint_position.reshape(-1, 3))
    # SciPy p*q applies q then p: R_position^-1 R_parent^-1 R_child.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.__mul__.html
    errors = (coordinate_rotation.inv() * parent_rotation.inv() * child_rotation).magnitude().reshape(32, 256, 23)
    require(np.isfinite(errors).all() and errors.max() <= 1e-5, "Native position triples do not match local rotation vectors")
    require(all(identity(row["path"]) == row for row in sources + inputs + velocity["sources"]),
            "Archive or sources changed during pose check")
    require(not torch.cuda.is_initialized() and not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"}
                                                      for name in sys.modules), "CPU audit initialized CUDA or imported an SDK")
    report = {"schema": "shared_social.actual_intermediate_joint_pose_CPU_audit.v1", "status": "passed_sameinstant_rotation_vector_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "inputs": inputs,
        "selected_archive_update": 500, "physical_actor_controls": 8192, "nonroot_joints_per_state": 23,
        "phase_verification": velocity["phase_verification"],
        "rotation_vector_geodesic_mean_rad": float(errors.mean()),
        "rotation_vector_geodesic_RMS_rad": float(np.sqrt(np.mean(errors * errors))),
        "rotation_vector_geodesic_max_rad": float(errors.max()), "geodesic_tolerance_rad": 1e-5,
        "per_joint_max_rad": dict(zip(velocity["body_order"][1:], errors.max(axis=(0, 1)).tolist(), strict=True)),
        "method": "Geodesic magnitude of Exp(native_position)^-1 R_parent_world^-1 R_child_world, independent SciPy rotation composition; same instant, no time difference",
        "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
        "scope": "Actual256-person intermediate500 batch, not arbitrary assets or entire training exposure. Consistency is not motion style, route, physics reliability or completion evaluation.",
        "all_failure_reset_controls_retained": True, "NN_constructed": False, "SDK_started": False,
        "CUDA_initialized": False, "training_restarted": False, "training_completed_inferred": False,
        "heldout_quality_evaluation": False, "default_model_promoted": False, "paper_effect_achieved": False}
    here.mkdir(parents=True)
    np.savez_compressed(here / "rotation_errors.npz", geodesic_error_rad=errors)
    report["errors"] = identity(here / "rotation_errors.npz")
    result = here / "result.json"
    result.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for path in here.iterdir():
        path.chmod(0o444)
    print(json.dumps({"result": identity(result), "mean_rad": report["rotation_vector_geodesic_mean_rad"],
                      "max_rad": report["rotation_vector_geodesic_max_rad"], "SDK_started": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
