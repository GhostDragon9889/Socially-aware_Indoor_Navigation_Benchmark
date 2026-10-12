#!/usr/bin/env python3
"""Bounded real Kit SMPL FK probe. Root executes one Kit instance at a time.

Uses the validated pedestrian builder, resets states without advancing physics,
then compares native/common body poses with current-asset exp-map and hinge FK.
Writes all evidence before close, including failures. No motion resources change.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='CrowdSim/config/pedestrian.yaml')
    parser.add_argument('--output', default='resources/runtime_sources/completion_20261001/isaaclab_smpl_fk.json')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    import_simulator_before_torch('isaaclab')
    import torch
    import warp as wp
    from protomotions.components.pose_lib import extract_transforms_from_qpos, compute_forward_kinematics_from_transforms
    from protomotions.utils.rotations import matrix_to_quaternion, quat_mul, quat_conjugate
    from protomotions.simulator.base_simulator.simulator_state import StateConversion
    from CrowdSim.utils.config_loader import cfg_path, load_config
    from CrowdSim.world.builder import build_env
    config = load_config(cfg_path(args.config, PROJECT_ROOT))
    config['navigation'].update(num_humanoids=2, num_robots=0)
    config['humanoid'].update(human_mesh=False, hide_humanoid=False, compile_components=False)
    config['car'].update(usd=None, rl_policy=False)
    config.setdefault('runtime', {})['enable_rendering'] = False
    config.setdefault('sensors', {}).setdefault('camera', {})['enabled'] = False
    config['sensors'].setdefault('lidar', {})['enabled'] = False
    env = None
    report = {'status': 'failed', 'physics_steps': 0, 'cases': [], 'config': config}
    try:
        env, agent, nav, runtime = build_env(config, headless=True)
        sim = env.simulator
        info = env.robot_config.kinematic_info
        lib = env.motion_lib
        total_frames = len(lib.gts)
        selected = [0, total_frames // 2]
        # Global rows mapped to per-motion frame indices explicitly.
        rows = torch.tensor(selected, device=env.device)
        motion_ids = torch.searchsorted(lib.length_starts, rows, right=True) - 1
        frame_ids = rows - lib.length_starts[motion_ids]
        reference = lib.get_motion_state_exact_frame(motion_ids, frame_ids)

        def oracle(state, exp_map):
            qpos = torch.cat((state.root_pos, state.root_rot[:, [3,0,1,2]], state.dof_pos), dim=-1)
            root, relative = extract_transforms_from_qpos(info, qpos, qpos_is_exp_map_on_3dof_joints=exp_map)
            return compute_forward_kinematics_from_transforms(info, root, relative)

        synthetic = reference.clone()
        synthetic.dof_pos.zero_()
        synthetic.dof_pos[:, :3] = torch.tensor([.8,-.6,.5], device=env.device)
        synthetic.dof_pos[:, 3:6] = torch.tensor([-.7,.3,.9], device=env.device)
        synthetic.root_rot[:] = torch.tensor([0.,0.,0.,1.], device=env.device)
        synthetic.rigid_body_pos, matrices = oracle(synthetic, True)
        synthetic.rigid_body_rot = matrix_to_quaternion(matrices, w_last=True)
        for name, state in [('amass_first_and_middle', reference), ('isolated_non_small_three_axis', synthetic)]:
            canonical = state.clone()
            canonical.state_conversion = StateConversion.COMMON
            canonical.dof_vel.zero_()
            canonical.rigid_body_vel.zero_()
            canonical.rigid_body_ang_vel.zero_()
            expected_pos, expected_rot = oracle(canonical, True)
            hinge_pos, hinge_rot = oracle(canonical, False)
            sim_time_before = float(sim._sim.physics_manager.get_simulation_time())
            sim.reset_envs(canonical.clone())
            sim._sim.physics_manager.forward()
            # This advances data cache timestamps only, not physical state/time.
            sim._scene.update(dt=sim._sim.get_physics_dt())
            actual = sim.get_robot_state()
            raw = sim._robot.data
            position_error = torch.linalg.vector_norm(actual.rigid_body_pos - expected_pos, dim=-1)
            hinge_error = torch.linalg.vector_norm(actual.rigid_body_pos - hinge_pos, dim=-1)
            expected_quat = matrix_to_quaternion(expected_rot, w_last=True).double()
            actual_quat = torch.nn.functional.normalize(actual.rigid_body_rot.double(), dim=-1)
            expected_quat = torch.nn.functional.normalize(expected_quat, dim=-1)
            delta = quat_mul(actual_quat, quat_conjugate(expected_quat, w_last=True), w_last=True)
            rotation_error = torch.rad2deg(2*torch.atan2(torch.linalg.vector_norm(delta[...,:3],dim=-1),delta[...,3].abs()))
            cached_error = torch.linalg.vector_norm(canonical.rigid_body_pos - expected_pos, dim=-1)
            case = {
                'name': name, 'global_motion_rows': selected if name.startswith('amass') else None,
                'exp_map_body_position_max_m': float(position_error.max()),
                'exp_map_body_position_mean_m': float(position_error.mean()),
                'exp_map_body_rotation_max_deg': float(rotation_error.max()),
                'hinge_body_position_max_m': float(hinge_error.max()),
                'cached_motion_vs_current_asset_max_m': float(cached_error.max()),
                'per_body_exp_map_error_m': dict(zip(info.body_names, position_error.max(0).values.cpu().tolist())),
                'common_dof_input': canonical.dof_pos.cpu().tolist(),
                'common_dof_output': actual.dof_pos.cpu().tolist(),
                'native_dof_positions': wp.to_torch(raw.joint_pos).cpu().tolist(),
                'native_root_quaternions_xyzw': wp.to_torch(raw.root_quat_w).cpu().tolist(),
                'native_body_quaternions_xyzw': wp.to_torch(raw.body_quat_w).cpu().tolist(),
                'actual_common_body_positions': actual.rigid_body_pos.cpu().tolist(),
                'expected_exp_map_body_positions': expected_pos.cpu().tolist(),
                'physical_time_before_s': sim_time_before,
                'physical_time_after_s': float(sim._sim.physics_manager.get_simulation_time()),
                'finite': bool(torch.isfinite(actual.rigid_body_pos).all() and torch.isfinite(actual.rigid_body_rot).all()),
            }
            case['status'] = 'compatible' if case['finite'] and case['exp_map_body_position_max_m'] < .001 and case['exp_map_body_rotation_max_deg'] < .1 and case['physical_time_before_s'] == case['physical_time_after_s'] else 'mismatch'
            report['cases'].append(case)
            output.write_text(json.dumps(report, indent=2)+'\n')
        report.update(native_joint_names=list(sim._robot.joint_names),
                      native_body_names=list(sim._robot.body_names),
                      common_body_names=info.body_names,
                      common_dof_names=info.dof_names,
                      hinge_axes={info.body_names[index]: axes.cpu().tolist() for index, axes in info.hinge_axes_map.items()},
                      motion_file=str(lib.motion_file),
                      status='compatible' if all(case['status']=='compatible' for case in report['cases']) else 'mismatch')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'status':report['status'], 'output':str(output), 'cases':[{key:case[key] for key in ('name','status','exp_map_body_position_max_m','hinge_body_position_max_m')} for case in report['cases']]},indent=2), flush=True)
        if env is not None:
            env.simulator._shutdown_exit_code = 0 if report['status']=='compatible' else 1
            env.close()
    return 0 if report['status']=='compatible' else 1


if __name__ == '__main__':
    sys.exit(main())
