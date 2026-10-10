"""CPU-only sampled audit of the actual AMASS manifest and neutral SMPL rig."""

from __future__ import annotations

from collections import Counter
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path

import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES, build_amp_observation
from data.smpl.smpl_joint_names import SMPL_MUJOCO_NAMES
from protomotions.assets import resolve_asset_root
from protomotions.components.pose_lib import (
    compute_forward_kinematics_from_transforms,
    extract_kinematic_info,
    extract_transforms_from_qpos,
)
from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.utils.rotations import (
    matrix_to_quaternion,
    quat_conjugate,
    quat_mul,
    quaternion_to_matrix,
)


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data/diffusion_amp/amass_manifest.json"
OUT = Path(__file__).parent / "amass_rig_consistency_review.json"
KEY_NAMES = ("R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    torch.set_num_threads(4)
    with contextlib.redirect_stdout(io.StringIO()):
        config = SmplRobotConfig()
    info = config.kinematic_info
    asset = resolve_asset_root(config.asset.asset_root) / config.asset.asset_file_name
    xml_info = extract_kinematic_info(str(asset))
    # Producer converts AMASS joint names to this explicit MJCF DFS order.
    # Embedded .pt tensors lack body names, so do not infer names from shape.
    names = list(info.body_names)
    assert names == list(PACER_BODY_NAMES) == list(SMPL_MUJOCO_NAMES) == list(xml_info.body_names)
    assert list(info.parent_indices) == list(xml_info.parent_indices)
    assert torch.equal(info.local_pos, xml_info.local_pos)
    parents = torch.as_tensor(info.parent_indices[1:], dtype=torch.long)
    expected_local = info.local_pos[1:].double()
    expected_lengths = expected_local.norm(dim=-1)
    key_indices = [names.index(name) for name in KEY_NAMES]
    asset_sha = sha256(asset)
    source_files = [
        MANIFEST, asset, Path(__file__),
        REPO / "protomotions/robot_configs/smpl.py",
        REPO / "protomotions/components/pose_lib.py",
        REPO / "protomotions/components/motion_lib.py",
        REPO / "data/scripts/prepare_amass_dataset.py",
        REPO / "data/scripts/convert_amass_to_proto.py",
        REPO / "data/smpl/smpl_joint_names.py",
        REPO / "CrowdSim/humanoid/amass_corpus.py",
        REPO / "CrowdSim/humanoid/amp/observations.py",
        REPO / "scripts/check_motion_asset_consistency.py",
        REPO / "docs/completion_20261001/isaaclab_smpl_fk.json",
    ]
    report = {
        "schema": "navisaaclab.amass_rig_consistency.v1",
        "status": "sampled_compatible",
        "cpu_only": True, "torch_threads": 4, "random_seed": 123,
        "kit_started": False, "source_resources_modified": False,
        "scope": "All selected train/validation clips at first/middle/last plus 512 random training-window frames per split; not every source frame.",
        "body_order": {
            "names": names, "parent_indices": list(info.parent_indices),
            "config_equals_actual_xml_equals_pacer_equals_converter_order": True,
            "embedded_names_in_pt": False,
            "evidence": [
                "prepare_amass_dataset.init_worker extracts smpl_humanoid.xml; convert_one passes SMPL_MUJOCO_NAMES",
                "convert_amass_to_motion permutes AMASS named joints into SMPL_MUJOCO_NAMES and generates gts through that rig FK",
                "MotionLib packs RobotState COMMON tensors unchanged; AMASSCorpus gathers gts/dps without a second body permutation",
                "Every selected motion ID/source path is checked against the actual packed motion_files and per-motion conversion sidecar",
            ],
        },
        "three_dof_representation": "exp_map",
        "stored_global_quaternion_order": "xyzw",
        "qpos_root_quaternion_order": "wxyz",
        "source_hashes": {str(path.relative_to(REPO)): sha256(path) for path in source_files},
        "large_pt_hash_policy": "Reuse manifest-declared SHA256 and recheck stat size; no repeated multi-GB hashing.",
        "splits": {},
        "interpretation": {
            "different_subject_bone_lengths_observed": False,
            "key_body_shape_shortcut_supported": False,
            "discriminator_unsolvable_from_rig_mismatch": False,
            "reason": "AMASS betas are not applied by this conversion: poses/trans are reprojected through the shared neutral MJCF. Sampled lengths and key positions agree to float32-scale numerical error.",
            "limits": [
                "No current physical rollout is read or altered; this verifies expert cache versus selected neutral rig, not contact/compliance in dynamic simulated policy states.",
                "The audit does not prove discriminator convergence or exclude other observation/action/velocity/timing distribution shortcuts.",
                "Original AMASS subject shape information is discarded by conversion; this is not preservation of individualized AMASS geometry.",
                "Current checkpoint training quality must be judged by its own held-out physical evaluation; an earlier 1000-update historical run had falls.",
            ],
        },
    }
    for split in ("train", "validation"):
        corpus = AMASSCorpus(MANIFEST, split, device="cpu", seed=123)
        ids = corpus.ids
        starts = corpus.data["length_starts"][ids]
        counts = corpus.data["motion_num_frames"][ids]
        representative = torch.stack((starts, starts + counts // 2, starts + counts - 1), dim=-1)
        random_rows, random_ids, _ = corpus._indices(512, past=9)
        indices = torch.cat((representative.flatten(), random_rows))
        state = corpus.state_at(indices)
        positions = state.rigid_body_pos.float()
        quats = torch.nn.functional.normalize(state.rigid_body_rot.double(), dim=-1)
        delta = positions[:, 1:].double() - positions[:, parents].double()
        lengths = delta.norm(dim=-1)
        length_error = (lengths - expected_lengths).abs()
        parent_rotation = quaternion_to_matrix(quats[:, parents], w_last=True)
        recovered_local = (parent_rotation.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)
        local_error = (recovered_local - expected_local).norm(dim=-1)
        qpos = torch.cat((positions[:, 0], quats[:, 0, [3, 0, 1, 2]].float(), state.dof_pos.float()), -1)
        root, relative = extract_transforms_from_qpos(info, qpos, qpos_is_exp_map_on_3dof_joints=True)
        fresh_pos, fresh_rot = compute_forward_kinematics_from_transforms(info, root, relative)
        pos_error = (positions - fresh_pos).norm(dim=-1)
        fresh_quats = torch.nn.functional.normalize(matrix_to_quaternion(fresh_rot, w_last=True).double(), dim=-1)
        quat_delta = quat_mul(quats, quat_conjugate(fresh_quats, w_last=True), w_last=True)
        rot_error = torch.rad2deg(2 * torch.atan2(quat_delta[..., :3].norm(dim=-1), quat_delta[..., 3].abs()))
        # Hold every other AMP field fixed to isolate the actual 12 key offsets.
        fresh_state = copy.copy(state)
        fresh_state.rigid_body_pos = fresh_pos
        fresh_state.root_pos = fresh_pos[:, 0]
        amp_error = (build_amp_observation(state) - build_amp_observation(fresh_state)).abs()
        assert amp_error.shape == (len(indices), 195)
        assert torch.count_nonzero(amp_error[:, :183]) == 0
        per_clip_mean = lengths[:len(ids) * 3].reshape(len(ids), 3, 23).mean(1)
        sidecar_hashes = {}
        historical_asset_hashes = Counter()
        missing_sidecars = []
        mismatched_source_paths = []
        for record in corpus.entry["clips"]:
            source = Path(record["source_path"])
            if str(source) != corpus.data["motion_files"][record["motion_id"]]:
                mismatched_source_paths.append(str(source))
            sidecar = source.with_suffix(".json")
            if not sidecar.is_file():
                missing_sidecars.append(str(sidecar))
                continue
            metadata = json.loads(sidecar.read_text())
            assert metadata["output"] == str(source)
            sidecar_hashes[str(sidecar)] = sha256(sidecar)
            historical_asset_hashes[metadata["signature"].get("protomotions/data/assets/mjcf/smpl_humanoid.xml")] += 1
        assert not missing_sidecars and not mismatched_source_paths
        assert historical_asset_hashes == {asset_sha: len(ids)}
        stat = Path(corpus.entry["path"]).stat()
        report["splits"][split] = {
            "source_path": corpus.entry["path"], "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns, "source_sha256_declared": corpus.entry["sha256"],
            "source_size_matches_manifest": stat.st_size == corpus.entry["bytes"],
            "manifest_selected_clips": len(ids), "clips_sampled_first_middle_last": len(ids),
            "random_frames": 512, "random_clips_covered": len(random_ids.unique()),
            "sampled_frames_including_duplicates": len(indices), "distinct_global_rows": len(indices.unique()),
            "sampled_global_rows": indices.tolist(),
            "finite": bool(all(torch.isfinite(t).all() for t in (lengths, local_error, pos_error, rot_error, amp_error))),
            "bone_abs_error_mean_m": float(length_error.mean()),
            "bone_abs_error_max_m": float(length_error.max()),
            "bone_population_variance_max_m2": float(lengths.var(0, correction=0).max()),
            "bone_per_clip_mean_std_max_m": float(per_clip_mean.std(0, correction=0).max()),
            "recovered_local_offset_error_mean_m": float(local_error.mean()),
            "recovered_local_offset_error_max_m": float(local_error.max()),
            "fk_body_position_error_mean_m": float(pos_error.mean()),
            "fk_body_position_error_max_m": float(pos_error.max()),
            "fk_body_rotation_error_mean_deg": float(rot_error.mean()),
            "fk_body_rotation_error_max_deg": float(rot_error.max()),
            "fk_key_body_position_error_max_m": float(pos_error[:, key_indices].max()),
            "amp_key_offset_coordinate_abs_error_max_m": float(amp_error[:, 183:].max()),
            "amp_other_183_fields_exactly_equal": True,
            "key_body_errors_max_m": {name: float(pos_error[:, i].max()) for name, i in zip(KEY_NAMES, key_indices)},
            "per_bone": {
                name: {
                    "parent": names[int(parents[j])], "rig_length_m": float(expected_lengths[j]),
                    "sample_mean_length_m": float(lengths[:, j].mean()),
                    "sample_min_length_m": float(lengths[:, j].min()),
                    "sample_max_length_m": float(lengths[:, j].max()),
                    "population_variance_m2": float(lengths[:, j].var(correction=0)),
                    "std_between_clip_mean_lengths_m": float(per_clip_mean[:, j].std(correction=0)),
                    "abs_error_mean_m": float(length_error[:, j].mean()),
                    "abs_error_max_m": float(length_error[:, j].max()),
                    "recovered_local_offset_error_max_m": float(local_error[:, j].max()),
                } for j, name in enumerate(names[1:])
            },
            "conversion_provenance": {
                "sidecar_count": len(sidecar_hashes), "sidecar_sha256": sidecar_hashes,
                "recorded_asset_sha256_counts": dict(historical_asset_hashes),
                "all_recorded_asset_hashes_match_current": True,
                "motion_id_source_paths_match_packed_motion_files": True,
            },
        }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    with torch.no_grad():
        result = audit()
    print(json.dumps({split: {key: entry[key] for key in (
        "sampled_frames_including_duplicates", "bone_abs_error_max_m",
        "bone_population_variance_max_m2", "fk_key_body_position_error_max_m",
        "recovered_local_offset_error_max_m", "amp_key_offset_coordinate_abs_error_max_m",
    )} for split, entry in result["splits"].items()}, indent=2))
