"""Optional finite clean-trajectory optimization of actual DDPM samples.

This is a declared local inference adaptation, not a trained replacement
planner or the author's private method. Every candidate starts from TRACE's
actual sampled actions. Only those intent actions are optimized; actual body
positions, policy weights and physical observations are never changed.
"""
import torch
from .trace_helpers import unicyle_forward_dynamics
from .guidance import motion_admission

def refine_samples(samples, states, guide, dynamics):
    cfg=guide.config
    if cfg.final_refinement_steps==0:return samples,None
    b,s,t,_=samples.shape
    bounds=samples.new_tensor([cfg.max_acceleration_m_s2,cfg.max_yaw_rate_rad_s])
    # Clipping actions preserves the existing bounded forward dynamics.
    original=samples[...,4:6].detach().clamp(-bounds,bounds)
    actions=original.clone().requires_grad_()
    initial=states.new_zeros(b,s,4);initial[...,2]=states[:,None,2]
    def integrate(value):
        x=unicyle_forward_dynamics(dynamics,initial.reshape(-1,4),value.reshape(-1,t,2),.1,
            pedestrian_speed_ceiling=cfg.max_speed_m_s,
            pedestrian_yaw_rate_ceiling=cfg.max_yaw_rate_rad_s,
            bounded_guidance_gradient=1.).reshape(b,s,t,4)
        return torch.cat((x,value),-1)
    optimizer=torch.optim.Adam([actions],lr=cfg.final_refinement_learning_rate)
    with torch.enable_grad():
        initial_trajectory=integrate(actions)
        initial_score,initial_parts=guide.score(initial_trajectory)
        best_score=initial_score.detach().clone()
        best_safe=motion_admission(initial_parts,cfg).detach().clone()
        best=initial_trajectory.detach().clone()
        for _ in range(cfg.final_refinement_steps):
            optimizer.zero_grad(set_to_none=True)
            trajectory=integrate(actions)
            score,_=guide.score(trajectory)
            anchor=((actions-original)/bounds).square().mean((-1,-2))
            loss=(score+cfg.final_refinement_anchor_weight*anchor).sum()
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite clean refinement loss')
            loss.backward()
            if not bool(torch.isfinite(actions.grad).all()):raise FloatingPointError('Nonfinite clean refinement gradient')
            optimizer.step()
            with torch.no_grad():
                actions.clamp_(-bounds,bounds)
                candidate=integrate(actions);cost,parts=guide.score(candidate)
                safe=motion_admission(parts,cfg)
                # Retain each sampled candidate's best feasible iteration.
                # A different sample's cost cannot erase an admitted route.
                keep=(safe & ~best_safe) | ((safe==best_safe) & (cost<best_score))
                best=torch.where(keep[...,None,None],candidate,best)
                best_score=torch.where(keep,cost,best_score)
                best_safe=torch.where(keep,safe,best_safe)
    meta={'kind':'finite bounded clean-trajectory optimization of actual TRACE DDPM samples',
          'steps':cfg.final_refinement_steps,'learning_rate':cfg.final_refinement_learning_rate,
          'anchor_weight':cfg.final_refinement_anchor_weight,
          'guidance_cost_sum_before':float(initial_score.sum().detach()),'guidance_cost_sum_after':float(best_score.sum()),
          'retention':'best feasible iteration per actor/candidate; final joint selection required',
          'motion_admitted_candidates':int(best_safe.sum()),
          'original_DDPM_raw_acceleration_max':float(samples[...,4].abs().max()),
          'original_DDPM_raw_yaw_rate_max':float(samples[...,5].abs().max()),
          'returned_action_channels':'refined bounded intents; original raw DDPM maxima recorded separately',
          'maximum_intent_action_change':float((best[...,4:6]-original).abs().max()),
          'body_pose_modified':False,'model_parameters_optimized':False,
          'same_joint_admission_required':True,'author_protocol':False}
    return best.detach(),meta
