"""CPU-only actual-bank equivalence, segment integrity and independent FK/dvs."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.training.manifests import validate_training_manifest
from protomotions.components.pose_lib import extract_transforms_from_qpos, compute_forward_kinematics_from_transforms
from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.assets import resolve_asset_root


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    paths = {version: root/f"data/diffusion_amp/amass_style_2848/manifest_labeled_{version}.json" for version in ("v6", "v7")}
    documents = {version: json.loads(path.read_text()) for version, path in paths.items()}
    old, new = documents["v6"], documents["v7"]
    banks = {version: torch.load(document["splits"]["train"]["path"], weights_only=True, mmap=True, map_location="cpu") for version, document in documents.items()}
    old_bank, bank = banks["v6"], banks["v7"]
    equality = {key: bool(torch.equal(value, bank[key])) if isinstance(value, torch.Tensor) else value == bank[key]
                for key, value in old_bank.items() if key != "dvs"}
    assert all(equality.values())
    records = new["splits"]["train"]["clips"]
    unchanged_record_keys = ["motion_id", "original_motion_id", "source_path", "parent_source_path", "frames", "dt",
        "parent_control_frame_start", "parent_control_frame_stop", "parent_source_frame_start", "parent_source_frame_stop", "style_class"]
    assert all(all(left[key] == right[key] for key in unchanged_record_keys) for left, right in zip(old["splits"]["train"]["clips"], records))
    assert new["splits"]["validation"] == old["splits"]["validation"] and new["splits"]["test"] == old["splits"]["test"]
    assert len(records) == 2848 and len(old["splits"]["train"]["clips"]) == len(records)
    groups = {}
    samples, velocity_indices = [], []
    for row in records:
        groups.setdefault(row["parent_source_path"], []).append(row)
        index, count = row["motion_id"], row["frames"]
        start = int(bank["length_starts"][index])
        samples.extend([start, start + count//2, start + count - 1])
        velocity_indices.extend([start, start + count//2, start + count - 2])
        assert count >= 91 and row["source_path"] == f"{row['parent_source_path']}#frames={row['parent_control_frame_start']}:{row['parent_control_frame_stop']}"
        assert 0 <= row["parent_source_frame_start"] < row["parent_source_frame_stop"] <= row["parent_source_frames"]
        assert torch.equal(bank["dvs"][start + count - 1], bank["dvs"][start + count - 2])
    for rows in groups.values():
        assert len(rows) <= 4
        ordered = sorted(rows, key=lambda row: row["parent_control_frame_start"])
        assert all(left["parent_control_frame_stop"] <= right["parent_control_frame_start"] for left, right in zip(ordered, ordered[1:]))
        assert all(left["parent_source_frame_stop"] <= right["parent_source_frame_start"] for left, right in zip(ordered, ordered[1:]))
    cfg = SmplRobotConfig()
    info = cfg.kinematic_info
    indices = torch.tensor(samples)
    qpos = torch.cat((bank["gts"][indices, 0], bank["grs"][indices, 0][:, [3, 0, 1, 2]], bank["dps"][indices]), -1)
    root_pos, matrices = extract_transforms_from_qpos(info, qpos, qpos_is_exp_map_on_3dof_joints=True)
    positions, _ = compute_forward_kinematics_from_transforms(info, root_pos, matrices)
    fk_error = torch.linalg.vector_norm(positions-bank["gts"][indices], dim=-1)
    assert float(fk_error.max()) < .001
    velocity_indices = torch.tensor(velocity_indices)
    q0 = bank["lrs"][velocity_indices, 1:].double().numpy().reshape(-1, 4)
    q1 = bank["lrs"][velocity_indices + 1, 1:].double().numpy().reshape(-1, 4)
    independent = (Rotation.from_quat(q0).inv()*Rotation.from_quat(q1)).as_rotvec().reshape(-1, 69)*30
    velocity_error = independent-bank["dvs"][velocity_indices].double().numpy()
    assert float(np.max(np.abs(velocity_error))) < .001
    frozen_bodies = {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}
    active = [index for index, name in enumerate(info.dof_names) if name.rsplit("_", 1)[0] not in frozen_bodies]
    entry = new["splits"]["train"]
    actual_sha = sha(entry["path"])
    conversion = new["curation"]["joint_velocity_conversion"]
    assert actual_sha == entry["sha256"] == conversion["target_bank_sha256"]
    assert bank["dof_velocity_frame"] == entry["dof_velocity_frame"] == new["selection"]["dof_velocity_frame"] == "child_joint"
    corpus = AMASSCorpus(paths["v7"], dof_velocity_frame="child_joint")
    state, history, actions, ids, frame = corpus.sample_reference(64)
    diffusion = corpus.diffusion_batch(8)
    assert state.dof_vel.shape == (64, 69) and history.dof_vel.shape == (640, 69)
    assert torch.isfinite(state.dof_vel).all() and torch.isfinite(actions).all()
    assert diffusion["history"].shape == (8, 31, 5) and diffusion["future"].shape == (8, 52, 5)
    asset = resolve_asset_root(cfg.asset.asset_root)/cfg.asset.asset_file_name
    report = {"status": "passed", "cpu_only": True, "kit_started": False, "cuda_used": False,
        "manifest": str(paths["v7"]), "manifest_sha256": sha(paths["v7"]), "bank": entry["path"],
        "bank_bytes": Path(entry["path"]).stat().st_size, "bank_sha256_recomputed": actual_sha,
        "other_field_equality": equality, "other_field_count": len(equality),
        "clip_order_membership_parent_bounds_unchanged": True, "selected_clips": len(records), "parent_sources": len(groups),
        "max_segments_per_source": max(map(len, groups.values())), "control_and_original_dependency_frame_ranges_nonoverlapping": True,
        "heldout_entries_exactly_unchanged": True, "selected_diffusion_clips": entry["diffusion_clip_count"],
        "selected_pose_samples": len(indices), "neutral_asset": str(asset), "neutral_asset_sha256": sha(asset),
        "fk_body_position_max_error_m": float(fk_error.max()), "fk_body_position_mean_error_m": float(fk_error.mean()),
        "independent_scipy_child_velocity_samples": len(velocity_indices), "independent_velocity_rms_error_rad_s": float(np.sqrt(np.mean(velocity_error**2))),
        "independent_velocity_max_abs_error_rad_s": float(np.max(np.abs(velocity_error))),
        "dof_velocity_frame": "child_joint", "dof_velocity_dtype": str(bank["dvs"].dtype), "dt": 1/30,
        "all_clip_last_velocity_frames_exactly_carry": True,
        "dvs_abs_max_rad_s_69": float(bank["dvs"].abs().max()), "dvs_abs_max_rad_s_active57": float(bank["dvs"][:, active].abs().max()),
        "active57_components_over40_rad_s": int((bank["dvs"][:, active].abs() > 40).sum()),
        "tensor_manifest_gate": validate_training_manifest(paths["v7"]),
        "corpus_real_batches": {"amp": 64, "history": 640, "diffusion": 8},
        "source_hashes": {str(path): sha(root/path) for path in ("CrowdSim/humanoid/training/amass_curation.py", "scripts/curate_amass_pedestrian_data.py", "scripts/convert_amass_joint_velocities.py", "configs/diffusion_amp/amass_curation_labeled.yaml", "tests/test_amass_curation.py")},
        "limitations": ["CPU data integrity/FK/formula checks are not controller convergence or a paper benchmark replication.",
            "Native coordinate evidence was supplied by the separate actual PhysX probe; this script does not start a simulator.",
            "Quantile quality gates allow rare angular velocity outliers, reported explicitly."]}
    Path(__file__).with_name("verification_v7.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"tensor_manifest_gate", "source_hashes"}}, indent=2))


if __name__ == "__main__":
    main()
