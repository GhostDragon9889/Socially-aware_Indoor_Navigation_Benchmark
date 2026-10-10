"""Training-only, phase-aligned real-AMASS posture and velocity rewards.

V2 emphasizes named trunk/arm joints and their current velocities. It rewards
matching moving reference arms, rather than freezing them. No pose override or
held-out reference is provided to the deployed controller.
"""
import torch
from ..amp.observations import PACER_DOF_NAMES, PACER_BODY_NAMES


def _heading_vectors(state, vectors):
    q = state.root_rot
    yaw = torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]), 1-2*(q[:,1]**2+q[:,2]**2))
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    return torch.stack((c*vectors[...,0]+s*vectors[...,1],
                        -s*vectors[...,0]+c*vectors[...,1], vectors[...,2]), -1)


def locomotion_reward(state, reference, previous_action, action, active_mask, settings, *, mapping=None, target_velocity_xy=None):
    mask = active_mask.to(state.dof_pos)
    joint_error = ((state.dof_pos-reference.dof_pos).square()*mask).sum(-1)/mask.sum()
    target=reference.root_vel[:,:2] if target_velocity_xy is None else target_velocity_xy
    if target.shape!=state.root_vel[:,:2].shape or not torch.isfinite(target).all():raise ValueError('Finite explicit training command velocity required')
    velocity_error = (state.root_vel[:, :2]-target).square().sum(-1)
    change = ((action-previous_action).square()*mask).sum(-1)/mask.sum()
    pose = torch.exp(-4.*joint_error)
    velocity = torch.exp(-4.*velocity_error)
    reward = (settings['posture_weight']*pose + settings['velocity_weight']*velocity
              - settings['action_change_weight']*change.clamp(max=4.))
    metrics = {'posture_mse_rad2':joint_error,'velocity_error_m_s':velocity_error.sqrt(),
               'posture_reward':pose,'velocity_reward':velocity,'action_change':change}
    if settings.get('schema') in ('crowdsim.locomotion_finetune.v2', 'crowdsim.locomotion_finetune.v3', 'crowdsim.locomotion_finetune.v4', 'crowdsim.locomotion_finetune.v5'):
        dofs = PACER_DOF_NAMES if mapping is None else mapping.dof_names
        bodies = PACER_BODY_NAMES if mapping is None else mapping.body_names
        upper_names = [n for n in PACER_DOF_NAMES if n.split('_')[0] in ('Torso','Spine','Chest','Neck','Head')
                       or any(n.startswith(s+'_'+j+'_') for s in ('L','R') for j in ('Thorax','Shoulder','Elbow','Wrist'))]
        own_ids = [dofs.index(n) for n in upper_names]
        ref_ids = [PACER_DOF_NAMES.index(n) for n in upper_names]
        upper_mask = mask[own_ids]
        upper_error = ((state.dof_pos[:,own_ids]-reference.dof_pos[:,ref_ids]).square()*upper_mask).sum(-1)/upper_mask.sum()
        upper_velocity = ((state.dof_vel[:,own_ids]-reference.dof_vel[:,ref_ids]).square()*upper_mask).sum(-1)/upper_mask.sum()
        twist_names = [n+'_z' for n in ('Torso','Spine','Chest')]
        twist_error = (state.dof_pos[:,[dofs.index(n) for n in twist_names]]
                       - reference.dof_pos[:,[PACER_DOF_NAMES.index(n) for n in twist_names]]).square().mean(-1)
        own_body = [bodies.index(n) for n in ('Torso','Spine','Chest')]
        ref_body = [PACER_BODY_NAMES.index(n) for n in ('Torso','Spine','Chest')]
        relative_own = state.rigid_body_ang_vel[:,own_body] - state.root_ang_vel[:,None]
        relative_ref = reference.rigid_body_ang_vel[:,ref_body] - reference.root_ang_vel[:,None]
        trunk_rate_error = (_heading_vectors(state,relative_own)-_heading_vectors(reference,relative_ref)).square().mean((-1,-2))
        def pelvis_up_heading(sample):
            x,y,z,w = sample.root_rot.unbind(-1)
            up = torch.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)), -1)
            return _heading_vectors(sample, up[:,None])[:,0]
        pelvis_tilt_error = (pelvis_up_heading(state)-pelvis_up_heading(reference)).square().sum(-1)
        pelvis_rate_error = (_heading_vectors(state,state.root_ang_vel[:,None])[:,:,:2]
                            - _heading_vectors(reference,reference.root_ang_vel[:,None])[:,:,:2]).square().mean((-1,-2))
        reward = (reward + settings['upperbody_pose_weight']*torch.exp(-8*upper_error)
                  + settings['upperbody_velocity_weight']*torch.exp(-.5*upper_velocity)
                  + settings['trunk_twist_weight']*torch.exp(-16*twist_error)
                  + settings['trunk_rate_weight']*torch.exp(-2*trunk_rate_error)
                  + settings['pelvis_stability_weight']*torch.exp(-40*pelvis_tilt_error)
                  + settings['pelvis_rate_weight']*torch.exp(-pelvis_rate_error))
        metrics.update(upperbody_pose_mse_rad2=upper_error, upperbody_velocity_mse_rad2_s2=upper_velocity,
                       trunk_twist_mse_rad2=twist_error, trunk_relative_rate_mse_rad2_s2=trunk_rate_error,
                       pelvis_up_vector_mse=pelvis_tilt_error, pelvis_roll_pitch_rate_mse_rad2_s2=pelvis_rate_error)
    if settings.get('schema') in ('crowdsim.locomotion_finetune.v3','crowdsim.locomotion_finetune.v4', 'crowdsim.locomotion_finetune.v5'):
        # Training-only end-effector imitation, in each body's heading frame.
        # DeepMimic combines imitation with task rewards:
        # https://xbpeng.github.io/projects/DeepMimic/
        # Match the real clip phase; never prescribe a synthetic stride/swing.
        def offsets(sample, names, selected, anchor='Pelvis', velocity=False):
            values = sample.rigid_body_vel if velocity else sample.rigid_body_pos
            delta = values[:, [names.index(n) for n in selected]] - values[:, names.index(anchor), None]
            return _heading_vectors(sample, delta)
        legs = [s+'_'+j for s in ('L','R') for j in ('Knee','Ankle','Toe')]
        leg_error = (offsets(state,bodies,legs)-offsets(reference,PACER_BODY_NAMES,legs)).square().mean((-1,-2))
        leg_rate_error = (offsets(state,bodies,legs,velocity=True)-offsets(reference,PACER_BODY_NAMES,legs,velocity=True)).square().mean((-1,-2))
        arm_errors, arm_rates = [], []
        for side in ('L','R'):
            names = [side+'_Elbow',side+'_Wrist']; anchor = side+'_Shoulder'
            arm_errors.append((offsets(state,bodies,names,anchor)-offsets(reference,PACER_BODY_NAMES,names,anchor)).square().mean((-1,-2)))
            arm_rates.append((offsets(state,bodies,names,anchor,True)-offsets(reference,PACER_BODY_NAMES,names,anchor,True)).square().mean((-1,-2)))
        arm_error = torch.stack(arm_errors,-1).mean(-1)
        arm_rate_error = torch.stack(arm_rates,-1).mean(-1)
        reward = (reward + settings['leg_endpoint_weight']*torch.exp(-60*leg_error)
                  + settings['leg_endpoint_velocity_weight']*torch.exp(-2*leg_rate_error)
                  + settings['arm_endpoint_weight']*torch.exp(-60*arm_error)
                  + settings['arm_endpoint_velocity_weight']*torch.exp(-2*arm_rate_error))
        metrics.update(leg_endpoint_mse_m2=leg_error, leg_relative_velocity_mse_m2_s2=leg_rate_error,
                       arm_endpoint_mse_m2=arm_error, arm_relative_velocity_mse_m2_s2=arm_rate_error)
    return reward, metrics
