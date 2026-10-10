"""Read-only CPU audit of actual recorded native poses versus replay SMPL LBS."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from CrowdSim.tools.smpl_mesh_visualizer import (
    HumanMeshConfig,
    SMPL_JOINT_NAMES,
    SMPLMeshVisualizer,
    SMPLRobotPoseMapper,
    quat_apply_wxyz,
    smpl_to_robot_matrix,
)
from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.utils.rotations import exp_map_to_quat, quaternion_to_matrix


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).parent
REPLAY = ROOT / "output/diffusion_amp/native_warehouse_recorded_replay_indoor_30s_v1/result.json"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def summary(values):
    values = values.detach().double().reshape(-1)
    return {"count": values.numel(), "mean": values.mean().item(), "max": values.max().item(),
            "p50": values.quantile(.5).item(), "p95": values.quantile(.95).item()}


def audit():
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        raise RuntimeError("This audit must run with CUDA_VISIBLE_DEVICES='' and no SDK")
    replay = json.loads(REPLAY.read_text())
    if replay["status"] != "passed" or replay["mode"] != "recorded_pose_visualization":
        raise ValueError("Completed actual recorded-only replay is required")
    source_records = replay["sources"]
    for key in ("states", "result", "common_rig"):
        if record(source_records[key]["path"]) != source_records[key]:
            raise ValueError(f"Replay source {key} bytes no longer match")
    result = json.loads(Path(source_records["result"]["path"]).read_text())
    if result["status"] != "passed":
        raise ValueError("Completed actual physical source is required")
    ground_record = result["ground_support_measurement"]["capture"]
    ground = record(ground_record["path"])
    if any(ground[key] != ground_record[key] for key in ("path", "bytes", "sha256")):
        raise ValueError("Actual native ground capture bytes changed")
    with np.load(source_records["states"]["path"], allow_pickle=False) as archive:
        states = {name: archive[name] for name in ("root_pos", "root_rot", "dof_pos", "body_pos")}
    with np.load(ground["path"], allow_pickle=False) as archive:
        native_pos = archive["body_link_pos_w"][:, -1].copy()
        native_quat = archive["body_link_quat_w"][:, -1].copy()
        native_names = archive["common_body_names"].tolist()
    rig = extract_kinematic_info(source_records["common_rig"]["path"])
    names = list(rig.body_names)
    if names != native_names or names != result["state_capture"]["body_names"]:
        raise ValueError("Common body semantic mapping differs")
    if not np.array_equal(native_pos, states["body_pos"]):
        raise ValueError("Last native substep and recorded post-control body positions differ")
    frames, actors = states["root_pos"].shape[:2]
    if (frames, actors) != (900, 4):
        raise ValueError("This frozen audit expects all900 controls of the actual4-actor source")
    visual = SMPLMeshVisualizer.from_config(HumanMeshConfig(
        str(ROOT / "data/smpl"), actors, "cpu", shape_std=0.0,
        texture_paths=tuple(replay["visual_resources"]["actor_texture_paths"]),
        appearance_seed=result["arguments"]["seed"],
    ))
    if not visual.has_texture or torch.count_nonzero(visual.betas):
        raise ValueError("Exact neutral textured replay configuration was not reconstructed")
    model = visual.smpl_model
    mapper = SMPLRobotPoseMapper()
    order = [SMPL_JOINT_NAMES.index(name) for name in names]
    transform = smpl_to_robot_matrix()
    roots = torch.from_numpy(states["root_pos"]).reshape(-1, 3)
    root_quats = torch.from_numpy(states["root_rot"]).reshape(-1, 4)[:, [3, 0, 1, 2]]
    dofs = torch.from_numpy(states["dof_pos"]).reshape(-1, 69)
    actual_pos = torch.from_numpy(native_pos).reshape(-1, 24, 3)
    actual_quat = torch.from_numpy(native_quat).reshape(-1, 24, 4)
    pos_errors, rot_errors, skin_min_z, finite_batches = [], [], [], 0
    with torch.no_grad():
        for start in range(0, len(roots), 64):
            stop = min(start + 64, len(roots))
            count = stop - start
            pose = mapper(dofs[start:stop], rig.dof_names)
            zero = torch.zeros(count, 3)
            output = model(betas=torch.zeros(count, 10), global_orient=zero,
                           body_pose=pose, transl=zero, return_verts=True)
            joints = (output.joints[:, :24] - output.joints[:, :1]) @ transform.T
            joints = quat_apply_wxyz(root_quats[start:stop, None], joints) + roots[start:stop, None]
            vertices = (output.vertices - output.joints[:, :1]) @ transform.T
            vertices = quat_apply_wxyz(root_quats[start:stop, None], vertices) + roots[start:stop, None]
            local_rot = quaternion_to_matrix(exp_map_to_quat(pose.reshape(-1, 23, 3), w_last=False), w_last=False)
            world_rot = [quaternion_to_matrix(root_quats[start:stop], w_last=False)]
            for body in range(1, 24):
                world_rot.append(world_rot[int(model.parents[body])] @
                                 (transform @ local_rot[:, body - 1] @ transform.T))
            world_rot = torch.stack(world_rot, 1)[:, order]
            native_rot = quaternion_to_matrix(actual_quat[start:stop], w_last=True)
            relative = world_rot.transpose(-1, -2) @ native_rot
            sin_angle = .5 * torch.linalg.vector_norm(torch.stack((
                relative[..., 2, 1] - relative[..., 1, 2],
                relative[..., 0, 2] - relative[..., 2, 0],
                relative[..., 1, 0] - relative[..., 0, 1],
            ), -1), dim=-1)
            cos_angle = .5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
            pos_errors.append((joints[:, order] - actual_pos[start:stop]).norm(dim=-1))
            rot_errors.append(torch.atan2(sin_angle, cos_angle))
            skin_min_z.append(vertices[..., 2].amin(-1))
            if not all(torch.isfinite(value).all() for value in (vertices, joints, world_rot)):
                raise ValueError("Actual LBS/FK produced nonfinite geometry")
            finite_batches += 1
    pos_error, rot_error = torch.cat(pos_errors), torch.cat(rot_errors)
    skin_min = torch.cat(skin_min_z)
    # The actual replay uses beta0 so rest-floor corrections are identically0.
    if torch.count_nonzero(visual.shape_floor_offsets):
        raise ValueError("Neutral replay unexpectedly applied a shape/floor correction")
    uv_path = ROOT / "data/smpl/smpl_uv.npz"
    with np.load(uv_path, allow_pickle=False) as archive:
        uv_shapes = {name: list(archive[name].shape) for name in archive.files}
    import smplx.body_models
    import smplx.lbs
    report = {
        "schema": "navisaaclab.recorded_smpl_skin_cpu_audit.v1",
        "status": "completed", "CPU_only": True, "GPU_used": False, "SDK_started": False,
        "production_source_modified": False, "new_renderer_candidate_needed": False,
        "actual_source": {"replay_result": record(REPLAY), **source_records, "ground_support": ground},
        "scope": {"post_control_frames": frames, "actors": actors, "actor_frames": frames * actors,
                  "joint_comparisons": frames * actors * 24, "vertices_computed": frames * actors * 6890,
                  "finite_skin_batches": finite_batches, "sampled_subset": False},
        "skin": {"vertices_per_actor": len(model.v_template), "triangle_faces_per_actor": len(visual.faces),
                 "joints": 24, "blend_shape_betas": visual.betas.tolist(),
                 "shape_floor_offsets_m": visual.shape_floor_offsets.tolist(),
                 "LBS_not_physical_capsule_render": True, "actual_replay_mesh_mode": replay["mesh_mode"],
                 "uv_shapes": uv_shapes, "face_varying_uv": visual._valid_uv_data()},
        "last_native_substep_matches_recorded_body_positions_bitwise": True,
        "LBS_joint_position_vs_native_link_origin_m": summary(pos_error),
        "reconstructed_joint_rotation_vs_native_link_rotation_rad": summary(rot_error),
        "per_body": {name: {"position_m": summary(pos_error[:, i]), "rotation_rad": summary(rot_error[:, i])}
                     for i, name in enumerate(names)},
        "skin_min_vertex_world_z_m": summary(skin_min),
        "skin_min_vertex_below_declared_floor_zero_actorframes": int((skin_min < 0).sum()),
        "skin_floor_interpretation": "Visual LBS surface only; not a collision/contact/slip metric. Native floor isz0; rigid collision approximations differ from skin.",
        "actual_replay_execution": {key: replay.get(key) for key in (
            "status", "frames", "physics_steps", "policy_forward_calls", "physical_humanoids_spawned",
            "timeline_time_before_s", "timeline_time_after_s", "simulation_time_advanced_s", "video_file")},
        "licensed_resources": {name: record(ROOT / "data/smpl" / name) for name in (
            "SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl", "smpl_uv.npz")},
        "textures": [record(path) for path in dict.fromkeys(visual.person_texture_paths)],
        "source_code": {str(path): record(path) for path in (
            ROOT / "scripts/replay_paper_pedestrians.py", ROOT / "CrowdSim/tools/smpl_mesh_visualizer.py",
            ROOT / "CrowdSim/humanoid/runtime/native_visual_scene.py", ROOT / "protomotions/utils/rotations.py",
            Path(smplx.body_models.__file__), Path(smplx.lbs.__file__), Path(__file__))},
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                    "smplx": importlib.metadata.version("smplx"), "CPU_threads": torch.get_num_threads()},
        "limits": ["This is a CPU audit of already recorded physical state, not a new SDK/GPU/physics run.",
                   "Small joint FK differences remain; pose is reconstructed from native DOFs and neutral bones, not fitted or teleported to captured link positions.",
                   "No new pose smoothing, body-shape diversity, clothing deformation, contact simulation, or actor training is introduced.",
                   "Existing textured appearance uses2body atlases on one neutral rig, not gender-specific actor physics."]
    }
    destination = OUTPUT / "actual_cpu_audit.json"
    if destination.exists():
        raise FileExistsError("Existing audit evidence must not be overwritten")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("scope", "LBS_joint_position_vs_native_link_origin_m",
        "reconstructed_joint_rotation_vs_native_link_rotation_rad", "skin_min_vertex_world_z_m")}, indent=2))


if __name__ == "__main__":
    audit()
