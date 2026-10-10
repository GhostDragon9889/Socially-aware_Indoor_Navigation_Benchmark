#!/usr/bin/env python3
"""Actual isolated PhysX PPO+AMP fine tuning with a frozen public PACER teacher."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def train_physics(app, args, owner, report, lifecycle):
    import numpy as np
    import torch
    from dataclasses import asdict
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.amp.observations import build_amp_observation
    from CrowdSim.humanoid.amp_world import AMPWorld
    from CrowdSim.humanoid.application.settings import TrainingConfig
    from CrowdSim.humanoid.training.pacer_finetune import AnchoredPacerTrainer
    from CrowdSim.humanoid.training.rewards import paper_task_reward, paper_total_reward
    torch.set_num_threads(4)
    owner.physics_frame = inspect.currentframe()
    started = time.monotonic()
    optimization = None if args.optimization_config is None else json.loads(args.optimization_config.read_text())
    if optimization is not None:
        required = {'schema', 'policy_learning_rate', 'discriminator_learning_rate', 'anchor_coefficient',
                    'posture_weight', 'velocity_weight', 'action_change_weight', 'amp_weight', 'task_weight'}
        if optimization.get('schema') in ('crowdsim.locomotion_finetune.v2', 'crowdsim.locomotion_finetune.v3', 'crowdsim.locomotion_finetune.v4', 'crowdsim.locomotion_finetune.v5'):
            required |= {'upperbody_pose_weight', 'upperbody_velocity_weight', 'trunk_twist_weight', 'trunk_rate_weight', 'pelvis_stability_weight', 'pelvis_rate_weight'}
        if optimization.get('schema') in ('crowdsim.locomotion_finetune.v3','crowdsim.locomotion_finetune.v4', 'crowdsim.locomotion_finetune.v5'):
            required |= {'leg_endpoint_weight', 'leg_endpoint_velocity_weight', 'arm_endpoint_weight', 'arm_endpoint_velocity_weight', 'motion_symmetry_coefficient'}
        if optimization.get('schema') in ('crowdsim.locomotion_finetune.v4','crowdsim.locomotion_finetune.v5'):required |= {'kinematic_gait_weight'}
        if optimization.get('schema')=='crowdsim.locomotion_finetune.v5':required |= {'temporal_gait_weight'}
        if set(optimization) != required or optimization['schema'] not in ('crowdsim.locomotion_finetune.v1', 'crowdsim.locomotion_finetune.v2', 'crowdsim.locomotion_finetune.v3', 'crowdsim.locomotion_finetune.v4', 'crowdsim.locomotion_finetune.v5'):
            raise ValueError('Wrong explicit locomotion recipe')
        import math
        if any(type(v) not in (float,int) or not math.isfinite(v) or v < 0
               for k,v in optimization.items() if k != 'schema'):
            raise ValueError('Locomotion settings must be finite nonnegative numbers')
        from CrowdSim.humanoid.training.locomotion_objective import locomotion_reward
        report['locomotion_optimization'] = optimization
        report['optimization_config_sha256'] = digest(args.optimization_config)
    try:
        corpus = AMASSCorpus(args.manifest, split="train", device="cuda:0", seed=args.seed,
                             dof_velocity_frame="child_joint")
        if len(corpus.ids) != 2848:
            raise ValueError("Fixed training population must be exactly2848")
        report["corpus"] = corpus.provenance()
        walking_course=None
        if args.walking_turn_windows is not None:
            from CrowdSim.humanoid.training.walking_turn_curriculum import WalkingTurnCurriculum
            walking_course=WalkingTurnCurriculum(corpus,args.walking_turn_windows,args.num_envs)
            report['walking_turn_course']=walking_course.describe()
        world = AMPWorld(app, corpus, num_envs=args.num_envs, episode_steps=300,
            terrain_extent_m=256., spawn_margin_m=96., amass_root_training=True,training_walking_course=walking_course)
        owner.world = world
        lifecycle.verify_loaded()
        config = TrainingConfig(learning_rate=1e-6, discriminator_learning_rate=2e-6,
            update_epochs=2, minibatch_size=1024, discriminator_batch_size=512,
            critic_coefficient=1., bounds_coefficient=1., max_grad_norm=10.,
            discriminator_input="state", amp_normalization="per_history_coordinate",
            warm_start_discriminator=True, effective_action_likelihood=True,
            discriminator_updates_per_update=2, discriminator_replay_capacity=32768,
            direction_mode="backwards_penalty", expert_dof_velocity_frame="child_joint",
            target_kl=.02, amp_reward_weight=1., task_reward_weight=.5, seed=args.seed)
        provenance = {"finetune": {"anchor_coefficient": 1., "normalizers": "all_four_frozen",
            "expert_split": "train", "deployment": "public_TRACE_DDPM_plus_local_finetuned_PACER_AMP",
            "task_sample_offsets_s": [i*.5 for i in range(10)],
            "training_commands": "same_clip_source_root_xy_fixed_translation_at_reset",
            "budget_updates": args.updates, "num_envs": args.num_envs, "rollout_steps": args.rollout_steps,
            "validation_test_clips_used_for_gradients": False,
            "author_experiment_reproduction": False}, "amass": corpus.provenance()}
        if optimization is not None:
            config.motion_symmetry_coefficient = optimization.get('motion_symmetry_coefficient', 0.)
            config.learning_rate = optimization['policy_learning_rate']
            config.discriminator_learning_rate = optimization['discriminator_learning_rate']
            config.amp_reward_weight = optimization['amp_weight']
            config.task_reward_weight = optimization['task_weight']
            provenance['finetune'].update(anchor_coefficient=optimization['anchor_coefficient'],
                locomotion_optimization=optimization, posture_reference='current same real training clip phase',
                future_pose_provided_to_policy=False)
        trainer = AnchoredPacerTrainer(config, device="cuda:0", provenance=provenance)
        trainer.warm_start_pacer(args.parent_checkpoint, world.lower, world.upper, expected_sha256=digest(args.parent_checkpoint))
        teacher_checkpoint = ROOT/'data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth'
        trainer.bind_teacher(teacher_checkpoint, world.lower, world.upper)
        report['frozen_teacher_sha256'] = digest(teacher_checkpoint)
        initial = {name: value.detach().cpu().clone() for name, value in trainer.model.state_dict().items()}
        report["physics"] = world.physics_manifest()
        report["training_config"] = asdict(config)
        report["initialization"] = trainer.provenance["initialization"]
        trainer.provenance['finetune']['immediate_parent_sha256'] = digest(args.parent_checkpoint)
        gait_prior = None
        temporal_prior = None
        if optimization is not None and optimization['schema'] in ('crowdsim.locomotion_finetune.v4','crowdsim.locomotion_finetune.v5'):
            if args.gait_prior is None:raise ValueError('Explicit training-only AMASS gait prior required')
            from CrowdSim.humanoid.training.kinematic_gait_prior import KinematicGaitPrior
            prior_data=json.loads(args.gait_prior.read_text())
            if prior_data['manifest_sha256']!=digest(args.manifest) or prior_data['source_bank_sha256']!=corpus.entry['sha256']:
                raise ValueError('Gait prior must bind exactly this real training bank')
            gait_prior=KinematicGaitPrior(prior_data,world.device)
            report['training_gait_prior']={'path':str(args.gait_prior),'sha256':digest(args.gait_prior),'fit_split':'train','deployed_pose_override':False}
            trainer.provenance['finetune']['training_gait_prior']=report['training_gait_prior']
        elif args.gait_prior is not None:raise ValueError('Gait prior needs explicit v4 recipe')
        if optimization is not None and optimization['schema']=='crowdsim.locomotion_finetune.v5':
            if args.temporal_gait_prior is None:raise ValueError('Train-only temporal gait statistics required')
            from CrowdSim.humanoid.training.temporal_gait_prior import TemporalGaitPrior
            temporal_data=json.loads(args.temporal_gait_prior.read_text())
            if temporal_data['manifest_sha256']!=digest(args.manifest) or temporal_data['source_bank_sha256']!=corpus.entry['sha256']:
                raise ValueError('Temporal gait statistics must bind this exact real AMASS train bank')
            temporal_prior=TemporalGaitPrior(temporal_data,world.num_envs,world.device)
            report['training_temporal_gait_prior']={'path':str(args.temporal_gait_prior),'sha256':digest(args.temporal_gait_prior),'fit_split':'train','deployed_pose_override':False}
            trainer.provenance['finetune']['training_temporal_gait_prior']=report['training_temporal_gait_prior']
        elif args.temporal_gait_prior is not None:raise ValueError('Temporal statistics require explicit v5 recipe')
        from CrowdSim.humanoid.training.task_horizon import horizon_frame_offsets
        offset_array=horizon_frame_offsets(world.num_envs,args.task_horizons_s)
        task_offsets=torch.as_tensor(offset_array.copy(),device='cpu')
        horizon_recipe={'horizons_s':list(args.task_horizons_s),'actor_assignment':'actor_id modulo horizon_count','source_frame_offsets_by_group':offset_array[:len(args.task_horizons_s)].tolist(),'current_source_phase_unchanged':True,'future_padding':'repeat each actor own last source root target','reference_body_and_velocity_unmodified':True,'deployed_pose_override':False}
        report['training_task_horizons']=horizon_recipe
        trainer.provenance['finetune']['training_task_horizons']=horizon_recipe
        curriculum = world.amass_root_curriculum
        coverage = set()
        resets = falls = tail_knots = queried_knots = 0

        previous_action = torch.zeros(world.num_envs, 69, device=world.device)

        def targets():
            if walking_course is not None:
                xy,beyond,_=walking_course.commands(world,task_offsets)
                return xy,beyond
            actors = torch.arange(world.num_envs, device=world.device)
            current = curriculum._phase(world.age, actors, world.episode_ids)
            phases = (curriculum.initial_frames + current).cpu()
            xy, _, beyond = curriculum._sample_xy(curriculum.motion_ids.cpu(),
                phases[:, None] + task_offsets, curriculum.offset_xy, curriculum.end_velocity_xy)
            return xy, beyond

        for update in range(args.updates):
            collected = {}
            route_errors = []
            for _ in range(args.rollout_steps):
                trajectory, beyond = targets()
                tail_knots += int(beyond.sum())
                queried_knots += beyond.numel()
                self_obs, task_obs = world.observations(trajectory)
                episode = world.episode_ids.clone()
                with torch.no_grad():
                    action = trainer.act(self_obs, task_obs)
                    state, terminated, truncated = world.step(action["applied_actions"])
                    history = world.amp_history.clone()
                    amp_reward = trainer.amp_reward(history, action["applied_actions"])
                    next_trajectory, _ = targets()
                    delta = (state.root_pos[:, :2] - next_trajectory[:, 0]).square().sum(-1)
                    yaw = world.planar(state)[:, 4]
                    forward = state.root_vel[:, 0]*yaw.cos() + state.root_vel[:, 1]*yaw.sin()
                    components = paper_task_reward(delta, state.dof_forces, state.dof_vel, forward,
                        direction_mode=config.direction_mode, signed_power_w=world.measured_signed_power_w())
                    reward = paper_total_reward(amp_reward, components,
                        amp_weight=config.amp_reward_weight, task_weight=config.task_reward_weight)
                    if optimization is not None:
                        absolute = (corpus.data['length_starts'][curriculum.motion_ids.cpu()]
                                    + (curriculum.initial_frames+world.age).cpu())
                        reference = corpus.state_at(absolute)
                        extra_reward, posture_metrics = locomotion_reward(state, reference, previous_action,
                            action['applied_actions'], trainer.action_mask, optimization, mapping=world.mapping, target_velocity_xy=None if walking_course is None else walking_course.commands(world,task_offsets)[2])
                        reward = reward + extra_reward
                        if gait_prior is not None:
                            gait_reward,gait_metrics=gait_prior.reward(state,world.mapping.body_names,arm_weight=.05 if temporal_prior is not None else .7)
                            reward=reward+optimization['kinematic_gait_weight']*gait_reward
                            posture_metrics.update(gait_metrics)
                        if temporal_prior is not None:
                            temporal_reward,temporal_metrics=temporal_prior.reward(state,world.mapping.body_names)
                            reward=reward+optimization['temporal_gait_weight']*temporal_reward
                            posture_metrics.update(temporal_metrics)
                        previous_action.copy_(action['applied_actions'])
                    next_self, next_task = world.observations(next_trajectory, state)
                    transition = {"self_obs": self_obs, "task_obs": task_obs, "amp_history": history,
                        "actions": action["actions"], "applied_actions": action["applied_actions"],
                        "log_prob": action["log_prob"], "values": action["values"],
                        "next_values": trainer.value(next_self, next_task), "rewards": reward,
                        "terminated": terminated, "truncated": truncated, "episode_ids": episode,
                        "history_episode_ids": world.amp_history_episode_ids.clone(), "amp_rewards": amp_reward,
                        "trajectory_rewards": components["trajectory"], "forward_velocity": forward}
                    for key, value in transition.items():
                        collected.setdefault(key, []).append(value.detach().clone())
                    route_errors.append(float(delta.sqrt().mean()))
                    ids = (terminated | truncated).nonzero().flatten()
                    resets += len(ids)
                    falls += int(terminated.sum())
                    world.reset(ids)
                    previous_action[ids] = 0
                    if temporal_prior is not None:temporal_prior.reset(ids)
            _, expert_state, _, motion_ids, _ = corpus.sample_reference(4096, 10)
            coverage.update(motion_ids.cpu().tolist())
            expert_history = build_amp_observation(expert_state, mapping=world.mapping).reshape(4096, 10, 195)
            expert_actions = torch.zeros(4096, 69, device=world.device)  # state-only D does not consume actions
            rollout = {k: torch.stack(v) for k, v in collected.items()}
            metrics = trainer.update(rollout, expert_history, expert_actions)
            if optimization is not None:
                metrics.update({k: float(v.mean()) for k,v in posture_metrics.items()})
            metrics.update(resets=resets, falls=falls, expert_unique_clips=len(coverage),
                           route_error_mean_m=float(np.mean(route_errors)), elapsed_s=time.monotonic()-started)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if update % 8 == 0 or update + 1 == args.updates:
                print(json.dumps(metrics), flush=True)
            if (update + 1) % 64 == 0 or update + 1 == args.updates:
                trainer.provenance["finetune"].update(expert_unique_motion_ids=sorted(coverage))
                # Full optimizer/RNG bundle; physics solver caches are never described as saved.
                physical = world.state()
                runner_state = {"physical": {key: getattr(physical, key).clone() for key in
                    ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel")},
                    "world": {key: getattr(world, key).clone() for key in
                    ("age", "episode_ids", "reference_ids", "amp_history", "amp_history_episode_ids",
                     "planar_history", "history_available")}, "curriculum": curriculum.state_dict(),
                    "corpus_rng": corpus.generator.get_state(), "walking_course":None if walking_course is None else walking_course.state_dict(), "physics_solver_caches_saved": False}
                trainer.save(args.output / "last.pt", runner_state=runner_state)
        if walking_course is not None:
            report['walking_turn_course']=walking_course.describe()
            trainer.provenance['finetune']['walking_turn_course']=walking_course.describe()
            trainer.provenance['finetune']['training_commands']='training-only walking-window source commands plus explicit tight circles/straight courses; no future3D pose provided to policy'
        report["weight_changes"] = {group: {"max_abs_delta": max(float((value.cpu()-initial[name]).abs().max())
            for name, value in trainer.model.state_dict().items() if name.startswith(group)),
            "changed_tensors": sum(not torch.equal(value.cpu(), initial[name])
                for name, value in trainer.model.state_dict().items() if name.startswith(group))}
            for group in ("actor", "task_encoder", "critic", "discriminator")}
        if any(value["changed_tensors"] == 0 for value in report["weight_changes"].values()):
            raise RuntimeError("Every declared trainable network must actually update")
        if coverage != set(corpus.ids.tolist()):
            raise RuntimeError("Do not call this fixed2848 training until every expert clip was sampled")
        report.update(status="passed", updates=trainer.update_count,
            physical_transitions=trainer.physical_transition_count, falls=falls, resets=resets,
            expert_unique_motion_ids=sorted(coverage), all_train_clips_seen=True,
            task_extrapolated_knots=tail_knots, task_total_knots=queried_knots,
            validation_test_clips_used_for_gradients=False, elapsed_s=time.monotonic()-started,
            convergence_claim=False, curriculum=curriculum.describe())
        report["export"] = trainer.export(args.output / "Humanoid_AMASS.pth", coverage=coverage,
                                           manifest_sha256=digest(args.manifest))
        # Explicitly break closures over owned simulator tensors before returning.
        targets = world = curriculum = None
    finally:
        world = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json")
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth")
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--optimization-config", type=Path)
    parser.add_argument("--gait-prior", type=Path)
    parser.add_argument("--temporal-gait-prior", type=Path)
    parser.add_argument("--walking-turn-windows",type=Path)
    parser.add_argument("--task-horizons-s",type=float,nargs="+",default=[4.5])
    args = parser.parse_args()
    from CrowdSim.humanoid.training.task_horizon import horizon_frame_offsets
    horizon_frame_offsets(args.num_envs,args.task_horizons_s)
    if min(args.updates, args.num_envs, args.rollout_steps) < 1:
        parser.error("Positive actual training budget required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "parent_sha256": digest(args.parent_checkpoint),
              "manifest_sha256": digest(args.manifest), "requested_updates": args.updates,
              "requested_physical_transitions": args.updates*args.num_envs*args.rollout_steps}
    from CrowdSim.humanoid.runtime.trace_pace_lifecycle import TracePaceLifecycle
    lifecycle = TracePaceLifecycle(report)
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    launcher = import_simulator_before_torch("isaaclab")
    app = launcher({"headless": True, "device": "cuda:0", "fast_shutdown": False,
                    "kit_args": lifecycle.kit_args}).app
    owner = SimpleNamespace(world=None, physics_frame=None)
    try:
        train_physics(app, args, owner, report, lifecycle)
    except Exception as error:
        report.update(status="failed_preserved", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        import torch
        torch.cuda.synchronize()
        from CrowdSim.humanoid.runtime.trace_pace_scope import clear_completed_physics_frame
        report["completed_physics_frame_release"] = clear_completed_physics_frame(owner, train_physics)
        gc.collect()
        lifecycle.close(app, owner.world)
        owner.world = None
        gc.collect()
        if not report["cleanup_succeeded"]:
            report["status"] = "failed_cleanup"
        (args.output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({key: report.get(key) for key in ("status", "updates", "physical_transitions", "error", "cleanup_succeeded")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
