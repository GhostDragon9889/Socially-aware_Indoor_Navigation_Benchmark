#!/usr/bin/env python3
"""Validate GPU checkpoint restore, subset resets and a resumed PPO+AMP update.

Run after finetune_pacer_amass.py. This checks exposed physical state, not
bitwise continuation of PhysX solver caches. Output must be a new directory.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def validate(app, args, owner, report, lifecycle):
    import torch

    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.amp_world import AMPWorld
    from CrowdSim.humanoid.training.pacer_finetune import AnchoredPacerTrainer
    from CrowdSim.humanoid.training.rewards import paper_task_reward, paper_total_reward
    from CrowdSim.humanoid.training.task_horizon import horizon_frame_offsets
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion

    owner.physics_frame = inspect.currentframe()
    torch.set_num_threads(4)
    trainer, runner = AnchoredPacerTrainer.load(args.checkpoint, device="cuda:0", restore_rng=False)
    if runner is None or runner.get("walking_course") is not None:
        raise ValueError("Requires the complete plain AMASS fine-tuning runner state")
    recipe = trainer.provenance["finetune"]
    if any(key in recipe for key in ("locomotion_optimization", "training_gait_prior", "training_temporal_gait_prior")):
        raise ValueError("This probe supports the plain AMASS reward recipe only")
    horizons = recipe["training_task_horizons"]["horizons_s"]
    num_envs = runner["world"]["age"].numel()
    if num_envs < 2:
        raise ValueError("Subset isolation requires at least two environments")
    corpus = AMASSCorpus(args.manifest, split="train", device="cuda:0", seed=7,
                         dof_velocity_frame="child_joint")
    world = AMPWorld(app, corpus, num_envs=num_envs, episode_steps=300,
                     terrain_extent_m=256., spawn_margin_m=96., amass_root_training=True)
    owner.world = world
    lifecycle.verify_loaded()
    report["physics"] = world.physics_manifest()
    teacher = ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"
    trainer.bind_teacher(teacher, world.lower, world.upper)
    fields = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel")
    world_fields = tuple(runner["world"])
    all_ids = torch.arange(num_envs, device=world.device)
    world.simulator.reset_envs(ResetState(
        **{key: runner["physical"][key].to(world.device) for key in fields},
        state_conversion=StateConversion.COMMON), env_ids=all_ids)
    restored = world.state()
    errors = {}
    report["restore"] = {"max_absolute_errors": errors, "num_envs": num_envs,
                         "physics_solver_caches_restored": False}
    for key in fields:
        expected = runner["physical"][key].to(world.device)
        actual = getattr(restored, key)
        errors[key] = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5,
                                   msg=lambda message: f"Restored {key}: {message}")
    for key in world_fields:
        getattr(world, key).copy_(runner["world"][key])
    curriculum = world.amass_root_curriculum
    curriculum.load_state_dict(runner["curriculum"])
    corpus.generator.set_state(runner["corpus_rng"])
    trainer.restore_rng()

    # Reverse/non-contiguous IDs catch scatter mistakes that contiguous slices miss.
    selected = all_ids[::2].flip(0)
    untouched = all_ids[1::2]
    before = {key: getattr(world.state(), key).clone() for key in fields}
    before_world = {key: getattr(world, key).clone() for key in world_fields}
    before_curriculum = curriculum.state_dict()["actors"]
    world.reset(selected)
    after = world.state()
    source_frames = (corpus.data["length_starts"][curriculum.motion_ids[selected].cpu()]
                     + curriculum.initial_frames[selected].cpu())
    reference = corpus.state_at(source_frames)
    reset_position = reference.root_pos.clone()
    reset_position[:, :2] = world.origins[selected]
    reset_position[:, 2] += world.ground_height_at(reset_position[:, :2])
    for key in fields:
        expected = reset_position if key == "root_pos" else getattr(reference, key)
        torch.testing.assert_close(getattr(after, key)[selected], expected,
                                   atol=2e-5, rtol=1e-5,
                                   msg=lambda message: f"Subset reset {key}: {message}")
    for key in fields:
        torch.testing.assert_close(getattr(after, key)[untouched], before[key][untouched],
                                   atol=2e-5, rtol=1e-5)
    for key in world_fields:
        assert torch.equal(getattr(world, key)[untouched], before_world[key][untouched]), key
    for key, value in curriculum.state_dict()["actors"].items():
        assert torch.equal(value[untouched.cpu()], before_curriculum[key][untouched.cpu()]), key
    assert torch.equal(world.episode_ids[selected], before_world["episode_ids"][selected] + 1)
    assert bool((world.age[selected] == 0).all())
    assert bool((world.amp_history_episode_ids[selected] == world.episode_ids[selected, None]).all())
    before_empty = world.episode_ids.clone()
    world.reset(all_ids[:0])
    assert torch.equal(before_empty, world.episode_ids)
    report["reset_isolation"] = {"selected": selected.cpu().tolist(),
                                "untouched": untouched.cpu().tolist(),
                                "selected_state_matches_source": True, "passed": True}

    offsets = torch.as_tensor(horizon_frame_offsets(num_envs, horizons).copy())

    def targets():
        phases = (curriculum.initial_frames + world.age).cpu()
        return curriculum._sample_xy(curriculum.motion_ids.cpu(), phases[:, None] + offsets,
                                     curriculum.offset_xy, curriculum.end_velocity_xy)[0]

    initial_updates = trainer.update_count
    initial_transitions = trainer.physical_transition_count
    weights = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
    collected = {}
    terminations = truncations = 0
    # Exercise the timeout path in real physics on the first step.
    world.episode_steps = int(world.age.max()) + 1
    for _ in range(args.steps):
        with torch.no_grad():
            own, task = world.observations(targets())
            episodes = world.episode_ids.clone()
            action = trainer.act(own, task)
            state, terminated, truncated = world.step(action["applied_actions"])
            amp_history = world.amp_history.clone()
            amp_reward = trainer.amp_reward(amp_history, action["applied_actions"])
            next_targets = targets()
            delta = (state.root_pos[:, :2] - next_targets[:, 0]).square().sum(-1)
            yaw = world.planar(state)[:, 4]
            forward = state.root_vel[:, 0] * yaw.cos() + state.root_vel[:, 1] * yaw.sin()
            components = paper_task_reward(delta, state.dof_forces, state.dof_vel, forward,
                direction_mode=trainer.config.direction_mode,
                signed_power_w=world.measured_signed_power_w())
            reward = paper_total_reward(amp_reward, components,
                amp_weight=trainer.config.amp_reward_weight, task_weight=trainer.config.task_reward_weight)
            next_own, next_task = world.observations(next_targets, state)
            transition = dict(self_obs=own, task_obs=task, amp_history=amp_history,
                actions=action["actions"], applied_actions=action["applied_actions"],
                log_prob=action["log_prob"], values=action["values"],
                next_values=trainer.value(next_own, next_task), rewards=reward,
                terminated=terminated, truncated=truncated, episode_ids=episodes,
                history_episode_ids=world.amp_history_episode_ids.clone(), amp_rewards=amp_reward,
                trajectory_rewards=components["trajectory"], forward_velocity=forward)
            for key, value in transition.items():
                if value.is_floating_point():
                    assert bool(torch.isfinite(value).all()), key
                collected.setdefault(key, []).append(value.detach().clone())
            terminations += int(terminated.sum())
            truncations += int(truncated.sum())
            world.reset((terminated | truncated).nonzero().flatten())
    assert truncations > 0, "The physical timeout path was not exercised"
    expert_history, expert_actions = world.expert_batch(4096)
    metrics = trainer.update({key: torch.stack(values) for key, values in collected.items()},
                             expert_history, expert_actions)
    changes = {group: max(float((value - weights[name]).abs().max())
                         for name, value in trainer.model.state_dict().items() if name.startswith(group))
               for group in ("actor", "critic", "task_encoder", "discriminator")}
    assert all(value > 0 for value in changes.values()), changes
    assert trainer.update_count == initial_updates + 1
    assert trainer.physical_transition_count == initial_transitions + num_envs * args.steps
    assert all(bool(torch.isfinite(value).all()) for value in trainer.model.state_dict().values())
    trainer.save(args.output / "resumed_optimizer.pt")
    report["saved_checkpoint"] = {"path": str(args.output / "resumed_optimizer.pt"),
                                  "physical_runner_state_included": False}
    report["resumed_update"] = {"updates_before": initial_updates, "updates_after": trainer.update_count,
        "new_physical_transitions": num_envs * args.steps, "terminations": terminations,
        "truncations": truncations, "weight_changes": changes, "metrics": metrics}
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least two")
    args.output.mkdir(parents=True, exist_ok=False)
    from social_navigation.sdk_paths import require_cuda_driver
    require_cuda_driver()
    from CrowdSim.humanoid.runtime.trace_pace_lifecycle import TracePaceLifecycle
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    report = {"status": "starting", "checkpoint": str(args.checkpoint.resolve())}
    lifecycle = TracePaceLifecycle(report)
    launcher = import_simulator_before_torch("isaaclab")
    app = launcher({"headless": True, "device": "cuda:0", "fast_shutdown": False,
                    "kit_args": lifecycle.kit_args}).app
    owner = SimpleNamespace(world=None, physics_frame=None)
    try:
        validate(app, args, owner, report, lifecycle)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        from CrowdSim.humanoid.runtime.trace_pace_scope import clear_completed_physics_frame
        report["frame_release"] = clear_completed_physics_frame(owner, validate)
        gc.collect()
        lifecycle.close(app, owner.world)
        owner.world = None
        gc.collect()
        if not report["cleanup_succeeded"]:
            report["status"] = "failed_cleanup"
        (args.output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({key: report.get(key) for key in ("status", "error", "cleanup_succeeded")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
