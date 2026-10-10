#!/usr/bin/env python3
"""Compare transferred and trained AMP actors in actual flat-ground physics.

The analytic trajectory is an isolated controller diagnostic. It is not a
Diffusion rollout, a navigation benchmark, or a reproduction of author scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/diffusion_amp/amass_manifest.json")
    parser.add_argument("--checkpoint", default="output/diffusion_amp/amp_training_corrected_power/step_1000.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if min(args.steps, args.num_envs) < 1 or not 0 < args.speed <= 3:
        parser.error("Positive bounds and speed in (0,3] m/s are required")
    output = Path(args.output).resolve()
    if (output / "result.json").exists():
        parser.error("Use a new output directory to preserve previous evidence")
    output.mkdir(parents=True, exist_ok=True)
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    app = import_simulator_before_torch("isaaclab")({"headless": True, "device": "cuda:0",
        "kit_args": "--/app/renderer/enabled=false --/renderer/enabled=false"}).app
    import numpy as np
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
    from CrowdSim.humanoid.amp_world import AMPWorld
    from CrowdSim.humanoid.training import PaperAMPTrainer, TrainingConfig
    world = None
    report = {"status": "failed", "arguments": vars(args), "method": "isolated low-level physical AMP diagnostic",
              "diffusion_used": False, "author_experiment_reproduced": False, "policies": {}}
    try:
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        corpus = AMASSCorpus(args.manifest, "validation", device="cuda:0", seed=args.seed)
        # Keep all reference paths inside the physical mesh, including a
        # conservative 3 m/s actor excursion rather than the command alone.
        extent = max(64., 2*3*args.steps/30 + 4*np.ceil(np.sqrt(args.num_envs)) + 24)
        world = AMPWorld(app, corpus, num_envs=args.num_envs, episode_steps=args.steps+1,
                         terrain_extent_m=float(extent))
        world.origins += 3*args.steps/30 + 6
        world.reset(torch.arange(args.num_envs, device=world.device))
        report["physics"] = world.physics_manifest()
        rng = corpus.generator.get_state().clone()
        times = torch.arange(.5, 5.1, .5, device=world.device)[None, :, None]
        all_ids = torch.arange(args.num_envs, device=world.device)
        for name in ("public_pacer", "trained_amp"):
            corpus.generator.set_state(rng)
            world.reset(all_ids)
            if name == "public_pacer":
                trainer = PaperAMPTrainer.prepare(TrainingConfig(), device=world.device)
                checkpoint = ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"
                trainer.warm_start_pacer(checkpoint, world.lower, world.upper,
                    body_names=world.mapping.body_names, dof_names=world.mapping.dof_names)
            else:
                checkpoint = Path(args.checkpoint)
                trainer, _ = PaperAMPTrainer.load(checkpoint, device=world.device, restore_rng=False)
            start = world.planar().clone()
            headings = start[:, 4]
            velocity = args.speed * torch.stack((headings.cos(), headings.sin()), -1)
            bases = start[:, :2].clone()
            roots, errors, heights, falls, first_fall = [], [], [], [], torch.full_like(world.age, -1)
            elapsed_started = time.monotonic()
            for step in range(args.steps):
                seconds = world.age.float()[:, None, None] / 30
                trajectory = bases[:, None] + (seconds + times)*velocity[:, None]
                own, task = world.observations(trajectory)
                action = trainer.act(own, task, deterministic=True)
                state, terminated, _ = world.step(action["applied_actions"])
                target = bases + (world.age.float()/30)[:, None]*velocity
                roots.append(state.root_pos.cpu().numpy())
                errors.append(torch.linalg.vector_norm(state.root_pos[:, :2]-target, dim=-1).cpu().numpy())
                heights.append(state.root_pos[:, 2].cpu().numpy())
                falls.append(terminated.cpu().numpy())
                first_fall[(first_fall < 0) & terminated] = step + 1
                ids = terminated.nonzero().flatten()
                world.reset(ids)
                if len(ids):
                    reset = world.planar()[ids]
                    bases[ids] = reset[:, :2]
                    velocity[ids] = args.speed * torch.stack((reset[:, 4].cos(), reset[:, 4].sin()), -1)
            roots, errors, heights, falls = map(np.stack, (roots, errors, heights, falls))
            np.savez_compressed(output / f"{name}_states.npz", root_pos=roots, tracking_error_m=errors,
                pelvis_height_m=heights, terminated=falls, first_fall_step=first_fall.cpu().numpy())
            report["policies"][name] = {"checkpoint": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint),
                "updates": trainer.update_count, "fall_resets": int(falls.sum()),
                "actors_with_fall": int((first_fall >= 0).sum()), "actors_without_fall": int((first_fall < 0).sum()),
                "mean_tracking_error_m": float(errors.mean()), "p95_tracking_error_m": float(np.percentile(errors, 95)),
                "minimum_pelvis_height_m": float(heights.min()), "elapsed_s": time.monotonic()-elapsed_started,
                "first_fall_step": first_fall.cpu().tolist(), "normalizers_updated": False,
                "reset_on_fall": True, "trajectory": "initial episode heading, absolute straight path, 10 future points at .5..5s"}
            print(json.dumps({name: report["policies"][name]}), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        if world is not None:
            world.simulator._shutdown_exit_code = 0 if report["status"] == "passed" else 1
            world.close()
        else:
            app.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
