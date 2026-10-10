"""Train the independent five-state diffusion model on declared AMASS windows.

This training set has individual motion trajectories, an obstacle-free map, and
no observed neighboring pedestrians. It is an explicit reconstruction data
assumption, not the authors' diffusion dataset or a crowd-interaction benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/diffusion_amp/amass_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/diffusion_amp/diffusion_training")
    parser.add_argument("--steps", type=int, default=1000, help="additional optimizer steps; also applies after resume")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--normalization-batches", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    for name in ("steps", "batch_size", "normalization_batches", "eval_batches", "eval_interval", "checkpoint_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch

    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.diffusion import PaperDiffusionConfig, PaperDiffusionPlanner, PaperDiffusionTrainBatch
    from CrowdSim.humanoid.diffusion.paper import world_to_agent_state
    from protomotions.utils.checkpoint import atomic_write_text

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA training but CUDA is unavailable")
    train = AMASSCorpus(args.manifest, split="train", device=args.device, seed=args.seed)
    validation = AMASSCorpus(args.manifest, split="validation", device=args.device, seed=args.seed + 1)
    test = AMASSCorpus(args.manifest, split="test", device=args.device, seed=args.seed + 2)
    assumptions = {
        "dataset": "Individual AMASS root trajectories under the recorded disjoint motion-file splits",
        "neighbors": "Empty: AMASS supplies no synchronized pedestrian interaction observations",
        "raster": "Uniform walkable=1, obstacles=0: AMASS supplies no scene occupancy maps",
        "author_dataset": False, "author_checkpoint": False,
        "limitations": "This run trains five-state locomotion prediction, not validated social/map-conditioned avoidance",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        normalization_samples = []
        for _ in range(args.normalization_batches):
            sample = train.diffusion_batch(args.batch_size)
            normalization_samples.append(world_to_agent_state(sample["future"], sample["states"]).reshape(-1, 5))
        samples = torch.cat(normalization_samples)
        mean = samples.mean(dim=0)
        mean[4] = torch.atan2(samples[:, 4].sin().mean(), samples[:, 4].cos().mean())
        residual = samples - mean
        residual[:, 4] = torch.remainder(residual[:, 4] + math.pi, 2 * math.pi) - math.pi
        std = residual.square().mean(dim=0).sqrt().clamp_min(0.25)
        config = PaperDiffusionConfig(state_mean=tuple(mean.cpu().tolist()), state_std=tuple(std.cpu().tolist()))
        planner = PaperDiffusionPlanner(config, device=args.device)
    else:
        planner = PaperDiffusionPlanner(checkpoint_path=args.resume, device=args.device)
    optimizer = torch.optim.Adam(planner.model.parameters(), lr=args.learning_rate)
    if args.resume is not None:
        artifact = planner.restore_training_checkpoint(args.resume, optimizer=optimizer)
        previous_dataset = artifact["dataset_metadata"]
        if previous_dataset.get("train", {}).get("manifest_sha256") != train.provenance()["manifest_sha256"]:
            raise ValueError("Resume manifest differs; retaining optimizer/RNG would be ambiguous")
        for corpus in (train, validation, test):
            corpus.generator.set_state(artifact["dataset_metadata"]["sampler_rng"][corpus.split])
    raster = torch.zeros((args.batch_size, 2, 224, 224), device=planner.device)
    raster[:, 0] = 1
    configuration = {"schema": "navisaaclab.five_state_training_run.v1", "model": asdict(planner.config),
                     "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                     "assumptions": assumptions, "train": train.provenance(),
                     "validation": validation.provenance(), "test": test.provenance()}
    atomic_write_text(json.dumps(configuration, indent=2), args.output_dir / "run_config.json")

    def batch_from(corpus):
        sample = corpus.diffusion_batch(args.batch_size)
        return PaperDiffusionTrainBatch.from_world(planner, sample["states"], sample["history"], None, raster, sample["future"])

    def evaluate(corpus):
        losses = []
        devices = [] if planner.device.type == "cpu" else [planner.device.index or 0]
        sampler_state = corpus.generator.get_state()
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.random.default_generator.manual_seed(args.seed + 100)
                if devices:
                    torch.cuda.default_generators[devices[0]].manual_seed(args.seed + 100)
                corpus.generator.manual_seed(args.seed + 200)
                for _ in range(args.eval_batches):
                    batch = batch_from(corpus)
                    loss, _ = planner.training_loss(batch.clean_trajectory, batch.context,
                                                     drop_conditioning=False, training=False)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite held-out diffusion loss")
                    losses.append(float(loss))
        finally:
            corpus.generator.set_state(sampler_state)
        return sum(losses) / len(losses)

    def save():
        dataset_metadata = {"train": train.provenance(), "validation": validation.provenance(), "test": test.provenance(),
                            "assumptions": assumptions,
                            "sampler_rng": {corpus.split: corpus.generator.get_state() for corpus in (train, validation, test)}}
        planner.save_training_checkpoint(args.output_dir / "last.ckpt", optimizer=optimizer, dataset_metadata=dataset_metadata)

    started = time.monotonic()
    initial_validation = evaluate(validation)
    metrics = []
    start_step = planner.trained_steps
    for index in range(args.steps):
        values = planner.train_step(batch_from(train), optimizer)
        values["step"] = planner.trained_steps
        values["elapsed_seconds"] = time.monotonic() - started
        if index == 0 or (index + 1) % args.eval_interval == 0 or index + 1 == args.steps:
            values["validation_loss"] = evaluate(validation)
            print(json.dumps(values), flush=True)
        metrics.append(values)
        if index == 0 or (index + 1) % args.checkpoint_interval == 0:
            save()
    save()
    result = {"status": "completed", "start_step": start_step, "completed_step": planner.trained_steps,
              "initial_validation_loss": initial_validation, "final_validation_loss": evaluate(validation),
              "test_loss": evaluate(test), "elapsed_seconds": time.monotonic() - started,
              "checkpoint": str((args.output_dir / "last.ckpt").resolve()),
              "metrics": metrics, "assumptions": assumptions}
    atomic_write_text(json.dumps(result, indent=2), args.output_dir / "training_result.json")
    print(json.dumps({key: value for key, value in result.items() if key != "metrics"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
