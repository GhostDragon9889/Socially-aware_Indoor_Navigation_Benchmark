"""Fine-tune the independent five-state model on newly generated ORCA experts.

Model/normalization initialization may come from the independent AMASS run;
the optimizer and RNG start a new social-training phase. This never overwrites
the frozen AMASS checkpoint used by an already-running AMP experiment.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PHASE = "new_maps_interact_union_uniform_window_finetune"
DEFAULT_INITIAL = ROOT / "output/diffusion_amp/diffusion_training/step_1000.ckpt"


def polygon_surface_gap(points, polygons, radius=.4):
    """Numpy exact segment distances/ray casting, no extra SDK dependency."""
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 2)
    signed = np.full(len(flat), np.inf)
    for polygon in polygons:
        vertices = np.asarray(polygon, dtype=np.float64)
        if np.array_equal(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        start, end = vertices, np.roll(vertices, -1, axis=0)
        segment = end - start
        offset = flat[:, None] - start
        fraction = np.sum(offset * segment, axis=-1) / np.sum(segment * segment, axis=-1)
        closest = start + np.clip(fraction, 0, 1)[..., None] * segment
        distance = np.linalg.norm(flat[:, None] - closest, axis=-1).min(axis=-1)
        x, y = flat[:, 0, None], flat[:, 1, None]
        crosses_y = (start[:, 1] > y) != (end[:, 1] > y)
        denominator = end[:, 1] - start[:, 1]
        denominator = np.where(denominator == 0, 1., denominator)
        cross_x = start[:, 0] + (y - start[:, 1]) * (end[:, 0] - start[:, 0]) / denominator
        inside = np.count_nonzero(crosses_y & (x < cross_x), axis=-1) % 2 == 1
        signed = np.minimum(signed, np.where(inside, -distance, distance))
    return (signed - radius).reshape(points.shape[:-1])


def _interaction_offline_plan_metrics(corpus, sample, predicted_xy):
    """Original ADE/FDE formulas; real radii and no-map metric N/A semantics."""
    predicted_xy = np.asarray(predicted_xy)
    if predicted_xy.shape != (len(sample["states"]), 50, 2) or not np.isfinite(predicted_xy).all():
        raise ValueError("Finite predicted future XY[B,50,2] required")
    target = np.asarray(sample["future"])[..., :50, :2]
    error = np.linalg.norm(predicted_xy - target, axis=-1)
    pair_gaps = []
    for row, index in enumerate(np.asarray(sample["window_indices"])):
        episode_index, agent, frame = corpus.windows[index]
        entry = corpus.episodes[episode_index]
        others = np.delete(entry["states"][:, frame + 1:frame + 51, :2], agent, axis=0)
        radii = np.asarray(entry["agent_radii_m"])
        gaps = np.linalg.norm(predicted_xy[row][None] - others, axis=-1)
        pair_gaps.append((gaps - (radii[agent] + np.delete(radii, agent))[:, None]).min(axis=0))
    pair_gaps = np.asarray(pair_gaps)
    collision = pair_gaps < -1e-4
    return {"windows": len(error), "ADE_m": float(error.mean()), "FDE_m": float(error[:, -1].mean()),
            "min_neighbor_surface_gap_m": float(pair_gaps.min()), "min_obstacle_surface_gap_m": None,
            "neighbor_collision_point_fraction": float(collision.mean()), "obstacle_collision_point_fraction": None,
            "neighbor_collision_trajectory_fraction": float(collision.any(axis=1).mean()),
            "obstacle_collision_trajectory_fraction": None, "qa_tolerance_m": 1e-4,
            "scope": "Open-loop ego predictions against recorded neighbors at aligned10Hz with actual agent radii; no obstacle map available, not a closed-loop rollout"}


def offline_plan_metrics(corpus, sample, predicted_xy):
    """Dispatch to each unchanged domain metric, preserving no-map N/A."""
    if corpus.domain == "maps":
        from scripts.train_orca_diffusion import offline_plan_metrics as maps_metrics
        return maps_metrics(corpus.corpus, sample, predicted_xy)
    return _interaction_offline_plan_metrics(corpus.corpus, sample, predicted_xy)


def encoding_map_mask(sample, mode):
    """Keep data presence false; separately select the encoder input policy."""
    mask = sample["map_condition_mask"]
    if mode == "skip_encoder":
        return mask
    if mode != "trace_unknown_raster":
        raise ValueError("Unknown missing-map encoding")
    return mask.new_ones(mask.shape) if hasattr(mask, "new_ones") else np.ones_like(mask, dtype=bool)


def make_training_batch(planner, sample, mode):
    from paper_interact import PaperDiffusionTrainBatch
    return PaperDiffusionTrainBatch.from_world(planner, sample["states"], sample["history"],
        sample["neighbors"], sample["raster"], sample["future"],
        history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"],
        map_condition_mask=encoding_map_mask(sample, mode))


def validate_resume_metadata(metadata, corpora, mode):
    requested = {corpus.split: corpus.resume_identity() for corpus in corpora}
    if (metadata.get("phase") != PHASE or metadata.get("corpus_identity") != requested
            or metadata.get("missing_map_encoding") != mode):
        raise ValueError("Resume phase, union corpora/sampling identity or missing-map encoding differs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-manifest", type=Path, default=ROOT / "data/diffusion_amp/orca_maps_expanded_1000_v1/manifest.json")
    parser.add_argument("--interact-manifest", type=Path, default=ROOT / "data/diffusion_amp/orca_interact_usable_1000_v1/manifest.json")
    parser.add_argument("--missing-map-encoding", choices=("trace_unknown_raster", "skip_encoder"), default="trace_unknown_raster")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/diffusion_amp/social_diffusion_maps_interact_staged_v1")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint", type=Path)
    group.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=1000, help="additional phase optimizer updates")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-memory-fraction", type=float, default=.15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--sample-windows", type=int, default=16)
    args = parser.parse_args(argv)
    for name in ("steps", "batch_size", "eval_batches", "eval_interval", "checkpoint_interval", "sample_windows"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= .15:
        parser.error("--cuda-memory-fraction must be within(0,.15]")
    if args.init_checkpoint is None and args.resume is None:
        args.init_checkpoint = DEFAULT_INITIAL
    if args.output_dir.resolve().is_relative_to(DEFAULT_INITIAL.parent.resolve()):
        parser.error("Output must be separate from the frozen AMASS training directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        parser.error("Output is nonempty; use an explicit phase resume or a fresh output directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    import torch

    from mixed_corpus import MixedORCACorpus, INTERACT_STAGE, sha256
    sys.path.insert(0, str(INTERACT_STAGE))
    from paper_interact import PaperDiffusionPlanner
    from protomotions.utils.checkpoint import atomic_write_text

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is unavailable")
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = MixedORCACorpus(args.maps_manifest, args.interact_manifest, "train", device=args.device, seed=args.seed)
    validation = MixedORCACorpus(args.maps_manifest, args.interact_manifest, "validation", device=args.device, seed=args.seed + 1)
    test = MixedORCACorpus(args.maps_manifest, args.interact_manifest, "test", device=args.device, seed=args.seed + 2)
    assumptions = {"phase": PHASE,
                   "data": "Existing Maps reconstruction plus official raw Interact resampled to100 true t0..9.9 points; preserved splits",
                   "author_dataset": False, "author_checkpoint": False,
                   "body_reference": "Velocity-derived disc heading proxy, not SMPL/AMP poses; each domain retains its original metric radii",
                   "conditions": "31 inclusive ego/history frames,19 neighbor slots with real masks; true no-map .5 raster and false data presence",
                   "missing_map_encoding": args.missing_map_encoding,
                   "missing_map_encoding_definition": "trace_unknown_raster encodes .5 rasters with encoder availability True; skip_encoder passes actual presence and is a local ablation",
                   "sampling": "Uniform over union valid windows with replacement; no imposed50:50 domain ratio, not author epoch-shuffle identity",
                   "initialization": "Independent AMASS model and normalization; new optimizer/RNG, no author or TRACE4D weights",
                   "validation": "Per-domain original open-loop metrics; Interact obstacle metrics remain N/A, not zero; no physical rollout claim"}
    checkpoint = args.resume or args.init_checkpoint
    if checkpoint.resolve() == (args.output_dir / "last.ckpt").resolve() and args.resume is None:
        raise ValueError("Initialization source cannot be overwritten by the training output")
    planner = PaperDiffusionPlanner(checkpoint_path=checkpoint, device=args.device)
    if args.resume is None:
        initialization = {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint),
                          "model_step": planner.trained_steps, "optimizer_reset": True, "rng_reset": True}
        first_gradient_evidence = None
    else:
        metadata = torch.load(checkpoint, weights_only=True, map_location="cpu")["dataset_metadata"]
        validate_resume_metadata(metadata, (train, validation, test), args.missing_map_encoding)
        initialization = metadata["initialization"]
        first_gradient_evidence = metadata.get("first_condition_gradient_evidence")
    if sha256(initialization["path"]) != initialization["sha256"]:
        raise ValueError("Frozen initialization checkpoint changed")
    optimizer = torch.optim.Adam(planner.model.parameters(), lr=args.learning_rate)
    if args.resume is not None:
        restored = planner.restore_training_checkpoint(checkpoint, optimizer=optimizer)
        for corpus in (train, validation, test):
            corpus.generator.bit_generator.state = restored["dataset_metadata"]["sampler_rng"][corpus.split]

    def from_sample(sample):
        return make_training_batch(planner, sample, args.missing_map_encoding)

    devices = [] if device.type == "cpu" else [device.index or 0]

    def evaluate(corpus):
        if hasattr(corpus, "domains"):
            return {name: evaluate(domain) for name, domain in corpus.domains.items()}
        previous = copy.deepcopy(corpus.generator.bit_generator.state)
        losses = []
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(args.seed + 100)
                corpus.generator = np.random.default_rng(args.seed + 200)
                for _ in range(args.eval_batches):
                    batch = from_sample(corpus.diffusion_batch(args.batch_size))
                    loss, _ = planner.training_loss(batch.clean_trajectory, batch.context, drop_conditioning=False, training=False)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite heldout diffusion loss")
                    losses.append(float(loss))
        finally:
            corpus.generator.bit_generator.state = previous
        return sum(losses) / len(losses)

    def sample_evaluation(corpus):
        if hasattr(corpus, "domains"):
            return {name: sample_evaluation(domain) for name, domain in corpus.domains.items()}
        previous = copy.deepcopy(corpus.generator.bit_generator.state)
        reports, indices = [], []
        try:
            corpus.generator = np.random.default_rng(args.seed + 300)
            for start in range(0, args.sample_windows, args.batch_size):
                sample = corpus.diffusion_batch(min(args.batch_size, args.sample_windows - start))
                result = planner.plan(sample["states"], sample["history"], sample["neighbors"], sample["raster"],
                    history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"], map_condition_mask=encoding_map_mask(sample, args.missing_map_encoding), seed=args.seed + 400 + start)
                cpu = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
                reports.append(offline_plan_metrics(corpus, cpu, result.xy[:, 1:].cpu().numpy()))
                indices.extend(cpu["window_indices"].tolist())
        finally:
            corpus.generator.bit_generator.state = previous
        total = sum(row["windows"] for row in reports)
        result = {key: sum(row[key] * row["windows"] for row in reports) / total
                  for key in reports[0] if reports[0][key] is not None and key not in ("scope", "qa_tolerance_m", "windows", "min_neighbor_surface_gap_m", "min_obstacle_surface_gap_m")}
        result.update(windows=total, scope=reports[0]["scope"], qa_tolerance_m=1e-4, window_indices=indices,
                      min_neighbor_surface_gap_m=min(row["min_neighbor_surface_gap_m"] for row in reports),
                      min_obstacle_surface_gap_m=(min(row["min_obstacle_surface_gap_m"] for row in reports) if reports[0]["min_obstacle_surface_gap_m"] is not None else None))
        for key in ("obstacle_collision_point_fraction", "obstacle_collision_trajectory_fraction"):
            if reports[0][key] is None:
                result[key] = None
        return result

    def save():
        metadata = {"phase": PHASE, "initialization": initialization, "assumptions": assumptions,
                    "missing_map_encoding": args.missing_map_encoding,
                    "corpus_identity": {corpus.split: corpus.resume_identity() for corpus in (train, validation, test)},
                    "train": train.provenance(), "validation": validation.provenance(), "test": test.provenance(),
                    "first_condition_gradient_evidence": first_gradient_evidence,
                    "sampler_rng": {corpus.split: copy.deepcopy(corpus.generator.bit_generator.state) for corpus in (train, validation, test)}}
        planner.save_training_checkpoint(args.output_dir / "last.ckpt", optimizer=optimizer, dataset_metadata=metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {"schema": "navisaaclab.orca_conditional_training_run.v1", "model": asdict(planner.config),
                     "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                     "initialization": initialization, "assumptions": assumptions,
                     "train": train.provenance(), "validation": validation.provenance(), "test": test.provenance()}
    atomic_write_text(json.dumps(configuration, indent=2), args.output_dir / "run_config_latest.json")
    if args.resume is None:
        atomic_write_text(json.dumps(configuration, indent=2), args.output_dir / "run_config_initial.json")
    started = time.monotonic()
    initial_validation = evaluate(validation)
    initial_sampling = sample_evaluation(test)
    if args.resume is None:
        # Evaluate the trained initialization before counting this fresh phase.
        planner.trained_steps = 0
    start_step = planner.trained_steps
    metrics = []
    for index in range(args.steps):
        sample = train.diffusion_batch(args.batch_size)
        # Until each encoder has evidence, do not drop its available condition.
        evidence_complete = (first_gradient_evidence is not None
                             and first_gradient_evidence["map_encoder"] is not None
                             and first_gradient_evidence["neighbor_encoder"] is not None)
        values = planner.train_step(from_sample(sample), optimizer, drop_conditioning=evidence_complete)
        if first_gradient_evidence is None:
            first_gradient_evidence = {"map_encoder": None, "neighbor_encoder": None, "first_all_missing_map": None}
        if not evidence_complete:
            for name, module in (("map_encoder", planner.model.map_encoder), ("neighbor_encoder", planner.model.neighbor_encoder)):
                gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
                norm = math.sqrt(sum(float(gradient.detach().square().sum()) for gradient in gradients))
                encoded_map = bool(encoding_map_mask(sample, args.missing_map_encoding).any())
                required = name != "map_encoder" or encoded_map
                if required and not (math.isfinite(norm) and norm > 0):
                    raise RuntimeError(f"Actual available conditional gradients are absent: {name}={norm}")
                if not required and gradients:
                    raise RuntimeError("Absent map encoder unexpectedly has gradients")
                if required and first_gradient_evidence[name] is None:
                    first_gradient_evidence[name] = {
                        "step": planner.trained_steps, "grad_norm_after_clip": norm,
                        "data_map_windows": int(sample["map_condition_mask"].sum()),
                        "encoded_map_windows": int(encoding_map_mask(sample, args.missing_map_encoding).sum()),
                        "batch_real_neighbor_slots": int(sample["neighbor_mask"][:, :, -1].sum()),
                        "batch_occupied_pixels": int((sample["raster"][:, 1] == 1).sum()),
                        "conditioning_dropout": False,
                        "unknown_raster_encoding_is_obstacle_evidence": False}
                if not required and first_gradient_evidence["first_all_missing_map"] is None:
                    first_gradient_evidence["first_all_missing_map"] = {"step": planner.trained_steps, "all_map_gradients_none": True}
        values["batch_domain_counts"] = {name: sample["domain_names"].count(name) for name in ("maps", "interact")}
        values.update(step=planner.trained_steps, elapsed_seconds=time.monotonic() - started)
        if device.type == "cuda":
            values.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        if index == 0 or (index + 1) % args.eval_interval == 0 or index + 1 == args.steps:
            values["validation_loss"] = evaluate(validation)
            print(json.dumps(values), flush=True)
        metrics.append(values)
        if index == 0 or (index + 1) % args.checkpoint_interval == 0:
            save()
    save()
    snapshot = args.output_dir / f"step_{planner.trained_steps}.ckpt"
    # Save a distinct immutable snapshot; no hardlink is mutated by atomic replace.
    if not snapshot.exists():
        snapshot.hardlink_to(args.output_dir / "last.ckpt")
    final_sampling = sample_evaluation(test)
    if sha256(initialization["path"]) != initialization["sha256"]:
        raise RuntimeError("Frozen AMASS initialization changed during independent finetune")
    result = {"status": "completed", "phase": PHASE, "start_step": start_step, "completed_step": planner.trained_steps,
              "initial_validation_loss": initial_validation, "final_validation_loss": evaluate(validation),
              "test_loss": evaluate(test), "initial_heldout_sampling": initial_sampling, "final_heldout_sampling": final_sampling,
              "first_condition_gradient_evidence": first_gradient_evidence, "elapsed_seconds": time.monotonic() - started,
              "checkpoint": str((args.output_dir / "last.ckpt").resolve()), "snapshot": str(snapshot.resolve()),
              "checkpoint_sha256": sha256(snapshot), "frozen_initialization_unchanged": True,
              "metrics": metrics, "assumptions": assumptions, "initialization": initialization}
    atomic_write_text(json.dumps(result, indent=2), args.output_dir / f"training_result_step_{planner.trained_steps}.json")
    atomic_write_text(json.dumps(result, indent=2), args.output_dir / "training_result.json")
    print(json.dumps({key: value for key, value in result.items() if key != "metrics"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
