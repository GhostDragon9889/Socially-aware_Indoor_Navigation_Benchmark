#!/usr/bin/env python3
"""Bounded direct SMPL/PACER physical compatibility probe (not a paper result)."""
from pathlib import Path
import argparse
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--output", default="output/diffusion_amp/physics_probe")
    args = parser.parse_args()
    if min(args.steps, args.num_envs) < 1:
        parser.error("Probe bounds must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    launcher = import_simulator_before_torch("isaaclab")
    app = launcher({"headless": True, "device": "cuda:0"}).app
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.amp import PacerAMPController
    from CrowdSim.humanoid.amp_world import AMPWorld
    world = None
    report = {"status": "failed", "method": "public PACER direct PhysX compatibility probe",
              "author_paper_result": False, "steps": args.steps, "resets": 0}
    try:
        torch.manual_seed(7)
        corpus = AMASSCorpus(ROOT / "data/diffusion_amp/amass_manifest.json", device="cuda:0")
        world = AMPWorld(app, corpus, num_envs=args.num_envs)
        report["physics"] = world.physics_manifest()
        controller = PacerAMPController.from_checkpoint(ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth",
            world.lower, world.upper, body_names=world.mapping.body_names, dof_names=world.mapping.dof_names, device="cuda:0")
        min_height = float("inf")
        for step in range(args.steps):
            state = world.state()
            xy = state.root_pos[:, :2]
            velocity = state.root_vel[:, :2]
            trajectory = xy[:, None] + velocity[:, None] * torch.arange(.5, 5.1, .5, device=world.device)[None, :, None]
            action = controller.act(state, trajectory, world, return_raw=True)
            state, terminated, truncated = world.step(action)
            min_height = min(min_height, float(state.root_pos[:, 2].min()))
            ids = (terminated | truncated).nonzero().flatten()
            report["resets"] += len(ids)
            world.reset(ids)
            if step % 30 == 0:
                print(json.dumps({"probe_step": step, "minimum_height_m": min_height, "resets": report["resets"]}), flush=True)
        report.update(status="passed", minimum_height_m=min_height)
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2, default=lambda value: getattr(value, "value", str(value))) + "\n")
        print(json.dumps({key: report.get(key) for key in ("status", "minimum_height_m", "resets", "error")}), flush=True)
        if world is not None:
            world.simulator._shutdown_exit_code = 0 if report["status"] == "passed" else 1
            world.close()
        else:
            app.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
