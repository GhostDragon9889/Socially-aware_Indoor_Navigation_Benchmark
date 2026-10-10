#!/usr/bin/env python3
"""Read-only, CPU/mmap check of cached motion FK against the selected MJCF asset.

No simulator is started, and no source motion or checkpoint is rewritten. This
accepts tensor-only .motion / packaged .pt files through weights_only=True.
A mismatched cache must be reconverted from its original source and robot asset;
changing backend tolerances or silently replacing cached tracking poses is unsafe.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from protomotions.assets import resolve_asset_root
from protomotions.simulator.base_simulator.simulator_state import StateConversion
from protomotions.components.pose_lib import (
    extract_transforms_from_qpos, compute_forward_kinematics_from_transforms,
)
from protomotions.robot_configs.factory import robot_config
from protomotions.utils.rotations import matrix_to_quaternion, quat_mul, quat_conjugate


def check_motion_asset(motion_file, robot_name, *, position_tolerance_m=1e-3,
                       rotation_tolerance_deg=.1, three_dof_representation=None):
    path = Path(motion_file).resolve()
    if position_tolerance_m <= 0 or rotation_tolerance_deg <= 0:
        raise ValueError('Consistency tolerances must be positive')
    # mmap reads only selected tensors/frames; in particular it avoids loading
    # the multi-GB AMASS train package into RAM or rehashing the whole dataset.
    with torch.serialization.safe_globals([StateConversion]):
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    packaged = 'gts' in data
    positions = data['gts' if packaged else 'rigid_body_pos']
    quaternions = data['grs' if packaged else 'rigid_body_rot']
    dofs = data['dps' if packaged else 'dof_pos']
    cfg = robot_config(robot_name)
    info = cfg.kinematic_info
    if positions.ndim != 3 or tuple(positions.shape[1:]) != (info.num_bodies, 3):
        raise ValueError(f'Cached body layout {tuple(positions.shape)} does not match {robot_name}: {info.num_bodies} bodies')
    if tuple(quaternions.shape) != (len(positions), info.num_bodies, 4) or tuple(dofs.shape) != (len(positions), info.num_dofs):
        raise ValueError('Cached quaternion/DOF/frame layout does not match the selected asset')
    if len(positions) == 0:
        raise ValueError('Motion cache has no frames')
    frames = sorted({0, len(positions)//2, len(positions)-1})
    indices = torch.tensor(frames)
    cached_positions = positions[indices].float()
    cached_quats = quaternions[indices].double()
    quat_norms = torch.linalg.vector_norm(cached_quats, dim=-1, keepdim=True)
    if not torch.isfinite(cached_positions).all() or not torch.isfinite(cached_quats).all() or torch.any(quat_norms == 0):
        raise ValueError('Cached poses must be finite with nonzero quaternions')
    cached_quats = cached_quats / quat_norms
    qpos = torch.cat((cached_positions[:, 0], cached_quats[:, 0, [3,0,1,2]].float(), dofs[indices].float()), dim=-1)
    representation = three_dof_representation or ('exp_map' if robot_name in {'smpl','smplx','soma23','amp'} else 'hinge')
    if representation not in {'exp_map','hinge'}:
        raise ValueError('three_dof_representation must be exp_map or hinge')
    root, relative = extract_transforms_from_qpos(info, qpos, qpos_is_exp_map_on_3dof_joints=representation == 'exp_map')
    fresh_pos, fresh_rot = compute_forward_kinematics_from_transforms(info, root, relative)
    position_error = torch.linalg.vector_norm(cached_positions - fresh_pos, dim=-1)
    fresh_quats = torch.nn.functional.normalize(matrix_to_quaternion(fresh_rot, w_last=True).double(), dim=-1)
    delta = quat_mul(cached_quats, quat_conjugate(fresh_quats, w_last=True), w_last=True)
    rotation_error = torch.rad2deg(2*torch.atan2(torch.linalg.vector_norm(delta[..., :3],dim=-1),delta[...,3].abs()))
    finite = bool(torch.isfinite(position_error).all() and torch.isfinite(rotation_error).all())
    max_pos = float(position_error.max())
    max_rot = float(rotation_error.max())
    asset = resolve_asset_root(cfg.asset.asset_root) / cfg.asset.asset_file_name
    stat = path.stat()
    return dict(
        status='compatible' if finite and max_pos <= position_tolerance_m and max_rot <= rotation_tolerance_deg else 'mismatch',
        cached_motion_vs_current_asset=True, motion_file=str(path), motion_size_bytes=stat.st_size,
        motion_mtime_ns=stat.st_mtime_ns, asset_file=str(asset.resolve()),
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        robot=robot_name, three_dof_representation=representation,
        total_frames=len(positions), sampled_frames=frames, finite=finite,
        body_pos_error_mean_per_frame_m=position_error.mean(-1).tolist(),
        body_pos_error_max_per_frame_m=position_error.max(-1).values.tolist(),
        body_pos_error_mean_m=float(position_error.mean()), body_pos_error_max_m=max_pos,
        body_rot_error_max_deg=max_rot, position_tolerance_m=position_tolerance_m,
        rotation_tolerance_deg=rotation_tolerance_deg,
        body_pos_error_max_per_body_m=dict(zip(info.body_names, position_error.max(0).values.tolist())),
        source_unchanged=True, deserialization='weights_only=True, mmap=True',
        remediation='Reconvert mismatched cache from original motion source with this selected robot asset; no automatic cache rewriting is performed.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion_file', type=Path)
    parser.add_argument('--robot', required=True, choices=['smpl','smplx','soma23','amp','g1','h1_2'])
    parser.add_argument('--three-dof-representation', choices=['exp_map','hinge'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check_motion_asset(args.motion_file, args.robot, three_dof_representation=args.three_dof_representation)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered+'\n')
    print(rendered)
    return 0 if report['status']=='compatible' else 2


if __name__ == '__main__':
    sys.exit(main())
