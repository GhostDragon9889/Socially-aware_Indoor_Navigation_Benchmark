"""Full validation-only loss diagnostic; original frozen model/loss APIs."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGE = ROOT / "docs/style_improvement_20261003"
PAIRED = STAGE / "orca_all_temporal_paired_validation_v1/evaluate_paired.py"
PAIRED_SHA = "9f7732c99c33984a5ffaf24d505dc8261668b84a4ccc3703c1e5c5bc07d7e113"
DOMAINS = ("maps", "interact")
MODE = "trace_unknown_raster"
LOSS_SEED = 20261101


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Ordinary file required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify(row):
    require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222,
            "Exact readonly artifact required: " + row["path"])


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def paired_api():
    require(record(PAIRED)["sha256"] == PAIRED_SHA, "Frozen paired source changed")
    spec = importlib.util.spec_from_file_location("_full_validation_existing_paired", PAIRED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ranges(total, batch):
    require(type(total) is int and total > 0 and type(batch) is int and batch > 0, "Positive exact counts required")
    return [(start, min(batch, total - start)) for start in range(0, total, batch)]


def aggregate(total, rows):
    """Only scheduling/aggregation; callback returns the original batch mean."""
    position, weighted = 0, 0.
    for row in rows:
        require(type(row["start"]) is int and row["start"] == position
                and type(row["count"]) is int and row["count"] > 0, "Contiguous whole-window coverage required")
        require(type(row["loss"]) in (float, int) and math.isfinite(row["loss"]), "Finite original batch mean required")
        position += row["count"]
        weighted += row["loss"] * row["count"]
    require(position == total, "No dropped or duplicated tail windows")
    return {"windows": total, "weighted_loss": weighted / total, "batches": len(rows), "tail_windows": rows[-1]["count"]}


def inventory(output, batch_size):
    import numpy as np
    started = time.monotonic()
    api = paired_api()
    plan, manifests = api.source_contract()
    domains = {}
    for name in DOMAINS:
        manifest = json.loads(Path(manifests[name]["path"]).read_text())
        contract = plan["expected_corpus_identity"]["validation"]
        require(manifest["view_schema"] == contract["view_schema"]
                and manifest["window_selection"] == contract["window_selection"], "Canonical all-temporal contract differs")
        history, future = manifest["history_frames"], manifest["future_frames"]
        require((history, future, manifest["data_hz"]) == (31, 52, 10), "Frozen temporal contract differs")
        episodes, chunks, position, scene_ids, episode_ids = [], [], 0, set(), set()
        for scene in manifest["splits"]["validation"]["scenes"]:
            require(scene["original_registered_split"] == "val" and scene["scene_id"] not in scene_ids,
                    "Unique registered validation scene required")
            scene_ids.add(scene["scene_id"])
            folder = Path(scene["folder"])
            map_row = None if scene["map"] is None else {**scene["map"], "path": str(folder / scene["map"]["path"])}
            if map_row is not None:
                verify(map_row)  # Bytes only: no raster array/crop construction.
            for episode in scene["episodes"]:
                require(episode["episode_id"] not in episode_ids, "Unique validation episode required")
                episode_ids.add(episode["episode_id"])
                state = {"path": str(folder / episode["path"]), "bytes": episode["bytes"], "sha256": episode["sha256"]}
                verify(state)  # Byte identity only: no state NPZ arrays loaded.
                verify(episode["window_index"])
                windows = np.load(episode["window_index"]["path"], allow_pickle=False)
                agents, frames = np.meshgrid(np.arange(episode["agents"]),
                                            np.arange(history - 1, episode["frames"] - future), indexing="ij")
                expected = np.column_stack((agents.ravel(), frames.ravel())).astype(np.int32)
                require(windows.dtype == np.int32 and np.array_equal(windows, expected)
                        and len(windows) == episode["conditional_window_count"], "Every legal canonical window exactly once required")
                chunks.append(np.column_stack((np.full(len(windows), len(episodes), dtype=np.int32), windows)))
                episodes.append({"scene_id": scene["scene_id"], "episode_id": episode["episode_id"],
                                 "agents": episode["agents"], "frames": episode["frames"],
                                 "anchors_inclusive": [history - 1, episode["frames"] - future - 1],
                                 "local_window_range_exclusive": [position, position + len(windows)],
                                 "window_index": episode["window_index"], "state": state, "map": map_row})
                position += len(windows)
        all_windows = np.concatenate(chunks).astype(np.int32)
        require(position == contract["domains"][name]["conditional_windows"], "Actual registered validation count differs")
        domains[name] = {"windows": position, "scenes": len(scene_ids), "episodes": episodes,
                         "batch_count": len(ranges(position, batch_size)), "tail_windows": ranges(position, batch_size)[-1][1],
                         "window_array": {"dtype": "int32", "shape": list(all_windows.shape),
                                          "sha256_C_bytes": hashlib.sha256(all_windows.tobytes()).hexdigest()}}
    require("torch" not in sys.modules, "Inventory must not import Torch")
    report = {"schema": "official_orca.full_validation_window_inventory.v1", "status": "actual_CPU_inventory_complete",
              "entry": record(__file__), "reused_paired_entry": record(PAIRED), "formal_plan": record(api.PLAN),
              "manifests": manifests, "validation_corpus_identity": plan["expected_corpus_identity"]["validation"],
              "split": "validation", "domain_order": list(DOMAINS), "domains": domains,
              "total_validation_windows": sum(row["windows"] for row in domains.values()), "batch_size": batch_size,
              "loss_seed": LOSS_SEED, "seed_rule": "base+domain_index*1000000+canonical_local_window_index",
              "random_rule": "independent CPU torch.Generator per window: randint100[1], then randn[52,5] float32",
              "formal_curve_scope": "800 fixed with-replacement windows per domain, not the full validation distribution",
              "inventory_NPY_indices_only": True, "state_and_map_bytes_verified_not_arrays_loaded": True,
              "raster_crops_generated": False, "NN_or_checkpoint_executed": False, "GPU_or_SDK_used": False,
              "test_data_or_sealed34_read": False, "future_terminal_audit": None, "elapsed_seconds": time.monotonic() - started}
    write_json(output, report)
    print(json.dumps({"status": report["status"], "domains": {name: row["windows"] for name, row in domains.items()},
                      "total": report["total_validation_windows"], "inventory": record(output)}, allow_nan=False))


def evaluate(args):
    # No weight reads or model imports occur before this existing terminal gate.
    api = paired_api()
    registration = json.loads((api.HERE / "registration.json").read_text())
    terminal, terminal_record = api.admit(registration, args.terminal_audit_sha256)
    selected = record(args.inventory)
    require(selected["sha256"] == args.inventory_sha256, "Actual root-selected inventory SHA required")
    verify(selected)
    inventory_document = json.loads(args.inventory.read_text())
    plan, manifests = api.source_contract()
    require(inventory_document["entry"] == record(__file__) and inventory_document["manifests"] == manifests
            and inventory_document["formal_plan"] == record(api.PLAN)
            and inventory_document["validation_corpus_identity"] == plan["expected_corpus_identity"]["validation"], "Inventory/source identity differs")
    checkpoints = [{"checkpoint": terminal["checkpoint"], "optimizer_updates": 40000}]
    if args.history_records is not None:
        history_row = record(args.history_records)
        require(history_row["sha256"] == args.history_records_sha256, "Root-selected historical byte records SHA required")
        verify(history_row)
        checkpoints += json.loads(args.history_records.read_text())["checkpoints"]
    require(not args.output_dir.exists(), "New exclusive diagnostic output required")
    require(len({row["optimizer_updates"] for row in checkpoints}) == len(checkpoints), "Unique checkpoint update counts required")
    for row in checkpoints:
        verify(row["checkpoint"])
        require(Path(row["checkpoint"]["path"]).stat().st_nlink == 1, "Independent readonly model copy required")
    for path in (ROOT, api.VIEW, api.INTERACT, api.MIXED, STAGE / "orca_official_mixed_training_staging_v2"):
        sys.path.insert(0, str(path))
    import numpy as np
    import torch
    from all_temporal_corpus import OfficialMapsORCACorpus, InteractionORCACorpus
    from paper_interact import PaperDiffusionConfig, PaperDiffusionPlanner
    from train_mixed_diffusion import make_training_batch
    from training_state import recipe_identity, validate_resume
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(), "CPU diagnostic cannot initialize CUDA")
    corpora = {name: constructor(manifests[name]["path"], "validation", device="cpu", seed=7, verify_hashes=True)
               for name, constructor in (("maps", OfficialMapsORCACorpus), ("interact", InteractionORCACorpus))}
    for name, corpus in corpora.items():
        require(hashlib.sha256(corpus.windows.tobytes()).hexdigest() == inventory_document["domains"][name]["window_array"]["sha256_C_bytes"],
                "Exact inventoried canonical window order required")
    private = {name: copy.deepcopy(corpus.generator.bit_generator.state) for name, corpus in corpora.items()}
    recipe = recipe_identity(microbatch=200, accumulation=2, learning_rate=.0002, max_grad_norm=None, missing_map_encoding=MODE)
    args.output_dir.mkdir(parents=True)
    results, normalization, config = [], None, None
    for selected_model in checkpoints:
        checkpoint = selected_model["checkpoint"]
        payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True, mmap=True)
        model_config = PaperDiffusionConfig(**payload["config"])
        require(model_config.diffusion_steps == 100, "Original100 diffusion steps required")
        validate_resume(payload, config=model_config, identity=recipe, corpus_identity=plan["expected_corpus_identity"])
        require(payload["optimizer_updates"] == selected_model["optimizer_updates"]
                and payload["dataset_metadata"]["initialization"] == plan["expected_initialization"]
                and payload["inference_weight_source"] == "EMA", "Historical full training/EMA metadata differs")
        require(config is None or asdict(model_config) == config, "Architecture differs between diagnostic checkpoints")
        config = asdict(model_config)
        with torch.random.fork_rng(devices=[]):
            planner = PaperDiffusionPlanner(config=model_config, device="cpu")
        for model_key in ("model", "training_raw_model"):
            planner.model.load_state_dict(payload[model_key], strict=True)
            planner.model.eval()
            norm = {key: getattr(planner.model, key).tolist() for key in ("state_mean", "state_std")}
            require(normalization is None or norm == normalization, "Normalization differs between raw/EMA/checkpoints")
            normalization = norm
            per_domain = {}
            for domain_index, name in enumerate(DOMAINS):
                corpus, batches = corpora[name], []
                t_digest, noise_digest = hashlib.sha256(), hashlib.sha256()
                for start, count in ranges(len(corpus.windows), args.batch_size):
                    indices = np.arange(start, start + count, dtype=np.int64)
                    sample = corpus.numpy_batch(count, indices=indices)
                    timesteps, noises = [], []
                    for index in indices:
                        generator = torch.Generator(device="cpu").manual_seed(LOSS_SEED + domain_index * 1000000 + int(index))
                        timesteps.append(torch.randint(100, (1,), generator=generator))
                        noises.append(torch.randn((52, 5), generator=generator, dtype=torch.float32))
                    timestep, noise = torch.cat(timesteps), torch.stack(noises)
                    t_digest.update(timestep.numpy().tobytes())
                    noise_digest.update(noise.numpy().tobytes())
                    with torch.no_grad():
                        batch = make_training_batch(planner, sample, MODE)
                        loss, parts = planner.training_loss(batch.clean_trajectory, batch.context,
                            timesteps=timestep, noise=noise, drop_conditioning=False, training=False)
                    batches.append({"start": start, "count": count, "loss": float(loss),
                                    "parts": {key: float(value) for key, value in parts.items()}})
                per_domain[name] = {**aggregate(len(corpus.windows), batches), "batch_records": batches,
                                    "timestep_int64_C_SHA": t_digest.hexdigest(), "noise_float32_C_SHA": noise_digest.hexdigest()}
                for component in batches[0]["parts"]:
                    per_domain[name][component] = aggregate(len(corpus.windows), [
                        {"start": row["start"], "count": row["count"], "loss": row["parts"][component]} for row in batches])["weighted_loss"]
            results.append({"checkpoint": checkpoint, "optimizer_updates": payload["optimizer_updates"],
                            "weight_key": model_key, "full_validate_resume_passed": True, "normalization": norm,
                            "domains": per_domain, "union_window_weighted_loss": sum(row["weighted_loss"] * row["windows"]
                                for row in per_domain.values()) / sum(row["windows"] for row in per_domain.values())})
        verify(checkpoint)  # Full source byte identity before/after actual diagnostic.
        del payload, planner
    require(all(corpus.generator.bit_generator.state == private[name] for name, corpus in corpora.items()), "Explicit indices consumed private data RNG")
    require(not torch.cuda.is_initialized(), "CPU diagnostic initialized CUDA")
    api.admit(registration, args.terminal_audit_sha256)
    report = {"schema": "official_orca.full_validation_loss_actual_diagnostic.v1", "status": "passed",
              "entry": record(__file__), "inventory": selected, "actual_terminal_audit": terminal_record,
              "batch_size": args.batch_size, "split": "validation", "results": results,
              "all_legal_windows_retained": True, "tail_weighted_by_windows": True,
              "per_window_timestep_noise_independent_of_batch_boundaries": True, "data_private_RNG_unchanged": True,
              "loss_scope": "one fixed t/noise realization per legal window; not full expectation over diffusion noise",
              "historical_diagnostic_not_model_selection": True, "formal_40000_main_comparison_changed": False,
              "GPU_or_SDK_used": False, "test_or_sealed34_read": False, "promotion": False}
    write_json(args.output_dir / "diagnostic.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inventory", "evaluate"), required=True)
    parser.add_argument("--inventory", type=Path, default=HERE / "actual_window_inventory.json")
    parser.add_argument("--inventory-sha256")
    parser.add_argument("--terminal-audit-sha256")
    parser.add_argument("--history-records", type=Path)
    parser.add_argument("--history-records-sha256")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/diffusion_amp/formal_full_validation_loss_diagnostic_v1")
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and args.batch_size > 0, "Hidden CUDA and positive batch required")
    if args.mode == "inventory":
        inventory(args.inventory, args.batch_size)
    else:
        require(args.inventory_sha256 and args.terminal_audit_sha256
                and (args.history_records is None or args.history_records_sha256), "Actual evidence SHA arguments required")
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
