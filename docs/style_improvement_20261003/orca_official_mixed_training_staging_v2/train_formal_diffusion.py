"""Formal independent five-state candidate: official two-domain data, EMA/accum.

No Isaac SDK. Full training/GPU execution is owned by root's future launch plan.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD_STAGE = HERE.parent / "maps_interact_mixed_training_staging_v1"
INTERACT_STAGE = HERE.parent / "orca_interact_usable_staging_v1"
DEFAULT_INITIAL = ROOT / "output/diffusion_amp/diffusion_training/step_1000.ckpt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-manifest", type=Path, default=ROOT / "data/diffusion_amp/orca_maps_official_usable_1000_v1/manifest.json")
    parser.add_argument("--interact-manifest", type=Path, default=ROOT / "data/diffusion_amp/orca_interact_usable_1000_v1/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/diffusion_amp/official_ORCA_maps_interact_EMA_accum_v2")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint", type=Path)
    group.add_argument("--resume", type=Path)
    parser.add_argument("--updates", type=int, default=40000, help="Additional complete optimizer updates, not microsteps")
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Default no clipping, as author source; explicit local choice")
    parser.add_argument("--missing-map-encoding", choices=("trace_unknown_raster", "skip_encoder"), default="trace_unknown_raster")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-memory-fraction", type=float, default=.8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--sample-windows", type=int, default=16)
    parser.add_argument("--skip-sampling", action="store_true", help="Explicit pilot-only omission, recorded in result")
    args = parser.parse_args(argv)
    for name in ("updates", "microbatch", "accumulation", "eval_batches", "eval_interval", "checkpoint_interval", "sample_windows"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Positive finite learning rate required")
    if args.max_grad_norm is not None and (not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0):
        parser.error("Explicit clipping must be positive finite")
    if not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1:
        parser.error("CUDA memory fraction must be in(0,1]")
    if args.init_checkpoint is None and args.resume is None:
        args.init_checkpoint = DEFAULT_INITIAL
    if args.output_dir.resolve().is_relative_to(DEFAULT_INITIAL.parent.resolve()):
        parser.error("Output must be independent of frozen AMASS initialization")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        parser.error("Fresh candidate output must be empty")
    if args.init_checkpoint and args.init_checkpoint.resolve().is_relative_to(args.output_dir.resolve()):
        parser.error("Initialization cannot live inside the candidate output")
    return args


@contextmanager
def using_model(planner, model):
    raw, previous_training = planner.model, model.training
    planner.model = model
    try:
        model.eval()
        yield
    finally:
        model.train(previous_training)
        planner.model = raw


def main(argv=None):
    args = parse_args(argv)
    for path in (ROOT, HERE, INTERACT_STAGE):
        sys.path.insert(0, str(path))
    # The old trainer only supplies unchanged tensor conditioning boundary.
    if str(OLD_STAGE) not in sys.path:
        sys.path.append(str(OLD_STAGE))
    import torch
    from mixed_corpus import MixedORCACorpus, sha256
    from train_mixed_diffusion import make_training_batch, encoding_map_mask
    from paper_interact import PaperDiffusionPlanner, PaperDiffusionConfig
    from metrics import offline_plan_metrics
    from training_state import (PHASE, WholeStateEMA, recipe_identity, optimizer_update,
                                checkpoint_payload, restore_training, validate_resume)
    from protomotions.utils.checkpoint import atomic_torch_save, atomic_write_text

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable")
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    corpora = {name: MixedORCACorpus(args.maps_manifest, args.interact_manifest, name,
                device=str(device), seed=args.seed + index)
               for index, name in enumerate(("train", "validation", "test"))}
    corpus_identity = {name: corpus.resume_identity() for name, corpus in corpora.items()}
    identity = recipe_identity(microbatch=args.microbatch, accumulation=args.accumulation,
        learning_rate=args.learning_rate, max_grad_norm=args.max_grad_norm,
        missing_map_encoding=args.missing_map_encoding)
    if args.resume:
        artifact = torch.load(args.resume, map_location="cpu", weights_only=True)
        config = PaperDiffusionConfig(**artifact["config"])
        validate_resume(artifact, config=config, identity=identity, corpus_identity=corpus_identity)
        planner = PaperDiffusionPlanner(config=config, device=str(device))
        initialization = artifact["dataset_metadata"]["initialization"]
    else:
        planner = PaperDiffusionPlanner(checkpoint_path=args.init_checkpoint, device=str(device))
        initialization = {"path": str(args.init_checkpoint.resolve()), "sha256": sha256(args.init_checkpoint),
                          "initial_model_step": planner.trained_steps, "optimizer_and_RNG_reset": True}
        planner.trained_steps = 0
    ema = WholeStateEMA(planner.model)
    optimizer = torch.optim.Adam(planner.model.parameters(), lr=args.learning_rate)
    if args.resume:
        restore_training(artifact, planner, ema, optimizer, corpora, identity=identity, corpus_identity=corpus_identity)
        del artifact
    if sha256(initialization["path"]) != initialization["sha256"]:
        raise ValueError("Frozen initialization changed")
    devices = [] if device.type == "cpu" else [device.index or 0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"schema": "local_orca.official_five_state_formal_training.v2", "phase": PHASE,
                  "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "model": asdict(planner.config), "training_recipe": identity, "initialization": initialization,
                  "corpora": {name: corpus.provenance() for name, corpus in corpora.items()},
                  "author_data_or_checkpoint_claim": False, "paper_400batch_explicit_claim": False,
                  "paper_40000_plus_official_config_batch400_combined_candidate": True,
                  "BN_microbatch_differs_from_author_batch400": True,
                  "default_deployment_model_key_is_EMA": True}
    atomic_write_text(json.dumps(run_config, indent=2, allow_nan=False), args.output_dir / "run_config_latest.json")
    if not args.resume:
        atomic_write_text(json.dumps(run_config, indent=2, allow_nan=False), args.output_dir / "run_config_initial.json")

    def make_batch(sample):
        return make_training_batch(planner, sample, args.missing_map_encoding)

    def evaluate_loss(model, union):
        reports = {}
        with using_model(planner, model):
            for name, corpus in union.domains.items():
                previous = copy.deepcopy(corpus.generator.bit_generator.state)
                losses = []
                try:
                    corpus.generator = np.random.default_rng(args.seed + 200)
                    with torch.random.fork_rng(devices=devices), torch.no_grad():
                        torch.manual_seed(args.seed + 100)
                        for _ in range(args.eval_batches):
                            batch = make_batch(corpus.diffusion_batch(args.microbatch))
                            loss, _ = planner.training_loss(batch.clean_trajectory, batch.context,
                                                             drop_conditioning=False, training=False)
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("Nonfinite heldout loss")
                            losses.append(float(loss))
                finally:
                    corpus.generator.bit_generator.state = previous
                reports[name] = sum(losses) / len(losses)
        return reports

    def sample_metrics(model, union):
        reports = {}
        with using_model(planner, model):
            for name, corpus in union.domains.items():
                previous = copy.deepcopy(corpus.generator.bit_generator.state)
                batches, indices = [], []
                try:
                    corpus.generator = np.random.default_rng(args.seed + 300)
                    for start in range(0, args.sample_windows, args.microbatch):
                        sample = corpus.diffusion_batch(min(args.microbatch, args.sample_windows - start))
                        prediction = planner.plan(sample["states"], sample["history"], sample["neighbors"], sample["raster"],
                            history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"],
                            map_condition_mask=encoding_map_mask(sample, args.missing_map_encoding), seed=args.seed + 400 + start)
                        cpu = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                               for key, value in sample.items()}
                        batches.append(offline_plan_metrics(corpus, cpu, prediction.xy[:, 1:].detach().cpu().numpy()))
                        indices.extend(cpu["window_indices"].tolist())
                finally:
                    corpus.generator.bit_generator.state = previous
                total = sum(row["windows"] for row in batches)
                summary = {"windows": total, "window_indices": indices, "scope": batches[0]["scope"],
                           "qa_tolerance_m": batches[0]["qa_tolerance_m"]}
                for key in ("ADE_m", "FDE_m", "neighbor_collision_point_fraction", "neighbor_collision_trajectory_fraction",
                            "obstacle_collision_point_fraction", "obstacle_collision_trajectory_fraction"):
                    summary[key] = (sum(row[key] * row["windows"] for row in batches) / total
                                    if batches[0][key] is not None else None)
                for key in ("min_neighbor_surface_gap_m", "min_obstacle_surface_gap_m"):
                    finite = [row[key] for row in batches if row[key] is not None]
                    summary[key] = min(finite) if finite else None
                if name == "maps" and summary["min_obstacle_surface_gap_m"] is None:
                    summary["min_obstacle_clearance_reason"] = "known_empty_map_no_finite_boundary"
                reports[name] = summary
        return reports

    def save(path):
        payload = checkpoint_payload(planner, ema, optimizer, corpora, identity=identity,
                                      corpus_identity=corpus_identity, initialization=initialization)
        atomic_torch_save(payload, path)

    started = time.monotonic()
    start_update = planner.trained_steps
    initial_loss = {"EMA": evaluate_loss(ema.model, corpora["validation"]),
                    "raw": evaluate_loss(planner.model, corpora["validation"])}
    records = []
    for update in range(start_update, start_update + args.updates):
        values = optimizer_update(planner, ema, optimizer, corpora["train"], make_batch=make_batch,
                                  identity=identity, update_index=update)
        values["elapsed_seconds"] = time.monotonic() - started
        if device.type == "cuda":
            values["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            values["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        if update == start_update or planner.trained_steps % args.eval_interval == 0 or update + 1 == start_update + args.updates:
            values["validation_loss"] = {"EMA": evaluate_loss(ema.model, corpora["validation"]),
                                         "raw": evaluate_loss(planner.model, corpora["validation"])}
            print(json.dumps(values, allow_nan=False), flush=True)
        records.append(values)
        if update == start_update or planner.trained_steps % args.checkpoint_interval == 0:
            save(args.output_dir / "last.ckpt")
    save(args.output_dir / "last.ckpt")
    snapshot = args.output_dir / f"step_{planner.trained_steps}.ckpt"
    if snapshot.exists():
        raise FileExistsError("Distinct candidate snapshot already exists")
    with (args.output_dir / "last.ckpt").open("rb") as source, snapshot.open("xb") as target:
        shutil.copyfileobj(source, target)
    sampling = None if args.skip_sampling else {
        "EMA": sample_metrics(ema.model, corpora["test"]),
        "raw": sample_metrics(planner.model, corpora["test"])}
    if sha256(initialization["path"]) != initialization["sha256"]:
        raise ValueError("Frozen initialization changed during training")
    result = {"schema": "local_orca.official_five_state_formal_training_result.v2", "status": "completed",
              "phase": PHASE, "start_optimizer_update": start_update, "completed_optimizer_updates": planner.trained_steps,
              "additional_optimizer_updates": args.updates,
              "microsteps": planner.trained_steps * args.accumulation,
              "training_windows": planner.trained_steps * args.microbatch * args.accumulation,
              "training_recipe": identity, "initial_validation_loss": initial_loss,
              "final_validation_loss": {"EMA": evaluate_loss(ema.model, corpora["validation"]),
                                        "raw": evaluate_loss(planner.model, corpora["validation"])},
              "test_loss": {"EMA": evaluate_loss(ema.model, corpora["test"]),
                            "raw": evaluate_loss(planner.model, corpora["test"])},
              "test_sampling": sampling, "sampling_omitted": args.skip_sampling,
              "snapshot": str(snapshot.resolve()), "snapshot_sha256": sha256(snapshot),
              "default_inference_weight_source": "EMA", "raw_training_state_preserved": True,
              "elapsed_seconds": time.monotonic() - started, "updates": records,
              "physical_or_paper_effect_proven": False}
    atomic_write_text(json.dumps(result, indent=2, allow_nan=False), args.output_dir / f"training_result_step_{planner.trained_steps}.json")
    atomic_write_text(json.dumps(result, indent=2, allow_nan=False), args.output_dir / "training_result.json")
    print(json.dumps({key: value for key, value in result.items() if key != "updates"}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
