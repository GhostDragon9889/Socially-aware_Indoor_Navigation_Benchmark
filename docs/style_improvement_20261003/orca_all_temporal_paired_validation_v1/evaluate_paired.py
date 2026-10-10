"""Fixed episode-balanced validation comparison; original model/loss/metrics APIs.

prepare-selection imports NumPy and frozen validation loaders only. Evaluation
requires the actual formal40000 terminal audit and its closed root owner.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent / "orca_official_all_temporal_formal_preparation_v1"
VIEW = HERE.parent / "orca_all_temporal_view_staging_v3"
INTERACT = HERE.parent / "orca_interact_usable_staging_v1"
MIXED = HERE.parent / "maps_interact_mixed_training_staging_v1"
PLAN = FORMAL / "commands_and_audit_plan.json"
PLAN_SHA = "2beed1bd10777658eb1c6ec8e1a2aa3bbc7c8571646ddf8f5fa9f3fbec9ab945"
AUDIT = FORMAL / "actual_terminal40000_audit.json"
OLD = {"path": str(ROOT / "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt"),
       "bytes": 279052920, "sha256": "63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac"}
SEEDS = {"maps": 20261301, "interact": 20261302}
DOMAINS = ("maps", "interact")
STRATA = ("episode_balanced_main", "zero_current_neighbor_diagnostic")
BATCH = 8
MODE = "trace_unknown_raster"
ENV = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify(row, *, readonly=True):
    require(record(row["path"]) == row, "Artifact identity changed: " + row["path"])
    require(not readonly or not Path(row["path"]).stat().st_mode & 0o222,
            "Readonly artifact required: " + row["path"])


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_contract():
    require(record(PLAN)["sha256"] == PLAN_SHA, "Reviewed formal plan changed")
    plan = json.loads(PLAN.read_text())
    verify(plan["source_registration"])
    source = json.loads(Path(plan["source_registration"]["path"]).read_text())
    binding = source["source_binding"]
    for key in ("reviewed_formal_sources", "reused_staged_conditioning_and_input_sources",
                "selected_production_helpers_verified_against_epoch", "additional_reused_helpers_outside_185_epoch",
                "all_temporal_view_sources"):
        for row in binding[key]:
            verify(row, readonly=False)
    verify(binding["production_epoch"], readonly=False)
    for row in plan["production_deployment_sources"]:
        verify(row, readonly=False)
    manifests = source["data"]["manifest_records_from_actual_all_temporal_view_QC"]
    for row in manifests.values():
        verify(row)
    return plan, manifests


def load_validation(manifests):
    # No Torch import in this boundary; original constructors load validation only.
    for path in (ROOT, VIEW, INTERACT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from all_temporal_corpus import MixedORCACorpus
    return MixedORCACorpus(manifests["maps"]["path"], manifests["interact"]["path"],
                           "validation", device="cpu", seed=7, verify_hashes=True)


def make_selection(union):
    import numpy as np
    selected = {}
    for name in DOMAINS:
        corpus = union.domains[name]
        generator = np.random.Generator(np.random.PCG64(SEEDS[name]))
        scenes = {scene["scene_id"]: scene for scene in corpus.manifest["splits"]["validation"]["scenes"]}
        main = []
        for episode in range(len(corpus.episodes)):
            candidates = np.flatnonzero(corpus.windows[:, 0] == episode)
            require(len(candidates) > 0, "Every validation episode must have canonical windows")
            main.append(int(candidates[int(generator.integers(len(candidates)))]))
        require(len(main) == 100, "Registered population is100 episodes per domain")
        main_set = set(main)
        rows, zero = {}, []
        for index, (episode, agent, frame) in enumerate(corpus.windows):
            entry = corpus.episodes[int(episode)]
            positions = entry["states"][:, int(frame), :2]
            distances = np.linalg.norm(positions - positions[int(agent)], axis=-1)
            distances[int(agent)] = np.inf
            neighbors = int(np.count_nonzero(distances <= 15.))
            if neighbors == 0:
                zero.append(index)
            if index not in main_set and neighbors != 0:
                continue
            scene = scenes[entry["scene_id"]]
            original = next(row for row in scene["episodes"] if row["episode_id"] == entry["episode_id"])
            state_row = {"path": str(Path(scene["folder"]) / original["path"]),
                         "bytes": original["bytes"], "sha256": original["sha256"]}
            map_row = None if scene["map"] is None else {
                **scene["map"], "path": str(Path(scene["folder"]) / scene["map"]["path"])}
            rows[index] = {"domain": name, "local_window_index": index, "scene_id": entry["scene_id"],
                           "episode_index": int(episode), "episode_id": entry["episode_id"],
                           "agent": int(agent), "frame": int(frame), "state_artifact": state_row,
                           "window_artifact": original["window_index"], "map_artifact": map_row,
                           "current_conditioning_neighbors": neighbors,
                           "other_recorded_agents": len(entry["states"]) - 1,
                           "overlaps_main_population": index in main_set}
        require(len(zero) == {"maps": 355, "interact": 149}[name], "Whole zero-neighbor stratum count changed")
        selected[name] = {"episode_balanced_main": [rows[index] for index in main],
                          "zero_current_neighbor_diagnostic": [rows[index] for index in zero],
                          "diagnostic_overlap_with_main": sorted(main_set.intersection(zero)),
                          "validation_legal_windows": len(corpus.windows)}
    return selected


def prepare():
    started = time.monotonic()
    require(not (HERE / "selection.json").exists() and not (HERE / "registration.json").exists(),
            "Selection/registration must be new exclusive artifacts")
    plan, manifests = source_contract()
    union = load_validation(manifests)
    private_before = copy.deepcopy(union.generator.bit_generator.state)
    domain_before = {key: copy.deepcopy(value.generator.bit_generator.state) for key, value in union.domains.items()}
    selection = {"schema": "official_orca.paired_validation_selection.v1", "split": "validation",
                 "selection_seeds_PCG64": SEEDS, "manifests": manifests,
                 "validation_corpus_identity": union.resume_identity(), "domains": make_selection(union),
                 "main_denominator": 200, "separate_diagnostic_denominator": 504,
                 "episode_balanced_not_full_window_distribution": True, "author_benchmark_claimed": False,
                 "diagnostic_added_to_main_denominator": False, "model_outputs_used_for_selection": False}
    require(union.generator.bit_generator.state == private_before
            and all(value.generator.bit_generator.state == domain_before[key] for key, value in union.domains.items()),
            "Explicit selection must preserve union and child private RNG")
    require("torch" not in sys.modules, "Preparation cannot import Torch")
    write_json(HERE / "selection.json", selection)
    registration = {"schema": "official_orca.paired_validation_registration.v1", "status": "prepared_selection_actual",
                    "entry": record(__file__), "source_plan": record(HERE / "SOURCE_PLAN.md"),
                    "formal_plan": record(PLAN), "source_registration": plan["source_registration"],
                    "selection": record(HERE / "selection.json"), "OLD": OLD,
                    "NEW": {"path": str(Path(plan["formal_output"]) / "step_40000.ckpt"), "bytes": None, "sha256": None},
                    "terminal_audit_path": str(AUDIT), "terminal_reader": plan["terminal_auditor"],
                    "terminal_audit_schema": "official_orca.all_temporal_actual_formal40000_CPU_checkpoint_audit.v1",
                    "domain_order": list(DOMAINS), "stratum_order": list(STRATA), "batch": BATCH,
                    "missing_map_encoding": MODE, "loss_seed_base": 20261101, "plan_seed_base": 20261401,
                    "batch_seed_rule": "base+domain_index*1000000+stratum_index*100000+batch_start_local_row",
                    "loss_random_generator": "CPU torch.Generator; randint100 then randn[B,52,5] float32; bothmodels share explicit arrays",
                    "plan_random_generator": "Original Planner.plan local CPU generator; same batch/seed for bothmodels",
                    "model_config_and_normalization_must_match_exactly": True,
                    "sampling": {"num_samples": 1, "waypoints": None, "guidance_weight": 1.0,
                                 "guidance_mode": "legacy_posterior", "collision_guidance": 0.0},
                    "CPU_environment": ENV, "future_model_effect_claimed": False, "GPU_used": False,
                    "NN_executed": False, "sealed34_outputs_read": False}
    write_json(HERE / "registration.json", registration)
    print(json.dumps({"status": "selection_prepared", "main": 200, "diagnostic": 504,
                      "overlaps": {name: len(selection["domains"][name]["diagnostic_overlap_with_main"]) for name in DOMAINS},
                      "selection": record(HERE / "selection.json"), "registration": record(HERE / "registration.json"),
                      "Torch_imported": False, "elapsed_seconds": time.monotonic() - started}, allow_nan=False))


def admit(registration, audit_sha):
    plan, _ = source_contract()
    verify(registration["entry"])
    verify(registration["selection"])
    require(registration["formal_plan"] == record(PLAN)
            and registration["source_registration"] == plan["source_registration"]
            and registration["terminal_reader"] == plan["terminal_auditor"], "Registered source identity differs")
    audit_record = record(AUDIT)
    require(audit_record["sha256"] == audit_sha, "Root-selected actual terminal audit SHA required")
    verify(audit_record)
    audit = json.loads(AUDIT.read_text())
    require(audit.get("schema") == registration["terminal_audit_schema"] and audit.get("status") == "passed"
            and audit.get("CPU_only") is True and audit.get("reader") == registration["terminal_reader"]
            and audit.get("plan") == registration["formal_plan"], "Actual77025 terminal audit required")
    for key in ("full_raw_EMA_Adam_strict_load_passed", "original_validate_resume_passed",
                "production_deployment_model_key_strict_load_equal_to_EMA"):
        require(audit.get(key) is True, "Actual terminal flag missing: " + key)
    for key in ("reader", "owner", "training_result", "checkpoint"):
        verify(audit[key])
    owner = json.loads(Path(audit["owner"]["path"]).read_text())
    require(type(owner.get("actual_exit_code")) is int and owner["actual_exit_code"] == 0
            and owner.get("child_exit_observed") is True and owner.get("command") == plan["formal_GPU_argv"],
            "Canonical root actual closed40000 owner required")
    require(audit["checkpoint"]["path"] == registration["NEW"]["path"], "Actual terminal checkpoint path differs")
    verify(registration["OLD"])
    return audit, audit_record


def array_record(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256_C_bytes": hashlib.sha256(array.tobytes()).hexdigest()}


def summarize(rows, loss_batches):
    total = len(rows)
    result = {"windows": total, "scope": rows[0]["scope"], "qa_tolerance_m": rows[0]["qa_tolerance_m"],
              "clean_loss": sum(loss * count for loss, count in loss_batches) / total}
    for key in ("ADE_m", "FDE_m", "neighbor_collision_point_fraction", "neighbor_collision_trajectory_fraction",
                "obstacle_collision_point_fraction", "obstacle_collision_trajectory_fraction"):
        result[key] = None if all(row[key] is None for row in rows) else sum(row[key] for row in rows) / total
    for key in ("min_neighbor_surface_gap_m", "min_obstacle_surface_gap_m"):
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = min(values) if values else None
    result["recorded_peer_observations"] = {
        key: sum(row["recorded_peer_observations"][key] for row in rows)
        for key in ("all_ego_future_points_collision_denominator", "windows_with_recorded_other_agents",
                    "known_no_other_agent_windows", "recorded_other_agent_point_comparisons")}
    result["recorded_peer_observations"].update(known_empty_windows_retained_in_denominator=True,
                                               conditioning_radius_restricts_metric_peers=False)
    for key in ("min_neighbor_clearance_reason", "min_obstacle_clearance_reason"):
        if key in rows[0] and all(row.get(key) == rows[0][key] for row in rows):
            result[key] = rows[0][key]
    return result


def evaluate(output, audit_sha):
    import numpy as np
    started = time.monotonic()
    registration = json.loads((HERE / "registration.json").read_text())
    audit, audit_record = admit(registration, audit_sha)
    require(not output.exists(), "Actual comparison output must be a new exclusive directory")
    output.mkdir(parents=True)
    selection = json.loads((HERE / "selection.json").read_text())
    for path in (FORMAL, MIXED, INTERACT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from paper_interact import PaperDiffusionPlanner
    from train_mixed_diffusion import make_training_batch, encoding_map_mask
    from metrics_all_temporal import offline_plan_metrics
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(), "CPU comparison cannot initialize CUDA")
    union = load_validation(selection["manifests"])
    require(union.resume_identity() == selection["validation_corpus_identity"], "Validation corpus identity changed")
    private = {"union": copy.deepcopy(union.generator.bit_generator.state),
               **{key: copy.deepcopy(value.generator.bit_generator.state) for key, value in union.domains.items()}}
    with torch.random.fork_rng(devices=[]):
        models = {name: PaperDiffusionPlanner(checkpoint_path=row["path"], device="cpu")
                  for name, row in (("OLD", registration["OLD"]), ("NEW", audit["checkpoint"]))}
    old, new = models["OLD"], models["NEW"]
    require(old.trained_steps == 5000 and new.trained_steps == 40000 and asdict(old.config) == asdict(new.config)
            and old.config.diffusion_steps == 100, "Original architecture/config/actual steps mismatch")
    normalization = {}
    for key in ("state_mean", "state_std"):
        require(torch.equal(getattr(old.model, key), getattr(new.model, key)), "Model normalization mismatch")
        normalization[key] = getattr(old.model, key).tolist()
    summaries, batch_rows, artifact_rows = {}, [], []
    for domain_index, domain in enumerate(DOMAINS):
        corpus = union.domains[domain]
        summaries[domain] = {}
        for stratum_index, stratum in enumerate(STRATA):
            rows = selection["domains"][domain][stratum]
            per_model = {name: [] for name in models}
            losses = {name: [] for name in models}
            for start in range(0, len(rows), BATCH):
                chosen = rows[start:start + BATCH]
                indices = np.asarray([row["local_window_index"] for row in chosen], dtype=np.int64)
                sample = corpus.numpy_batch(len(chosen), indices=indices)
                seed_offset = domain_index * 1000000 + stratum_index * 100000 + start
                loss_seed = 20261101 + seed_offset
                plan_seed = 20261401 + seed_offset
                generator = torch.Generator(device="cpu").manual_seed(loss_seed)
                timesteps = torch.randint(100, (len(chosen),), generator=generator)
                noise = torch.randn((len(chosen), 52, 5), generator=generator, dtype=torch.float32)
                metadata = {"domain": domain, "stratum": stratum, "start": start, "indices": indices.tolist(),
                            "loss_seed": loss_seed, "plan_seed": plan_seed,
                            "input_arrays": {key: array_record(value) for key, value in sample.items() if isinstance(value, np.ndarray)},
                            "data_map_presence_mask": sample["map_condition_mask"].tolist(),
                            "encoder_policy_mask": encoding_map_mask(sample, MODE).tolist(), "models": {}}
                arrays = {"indices": indices, "loss_timesteps": timesteps.numpy(), "loss_noise": noise.numpy()}
                for name, planner in models.items():
                    batch = make_training_batch(planner, sample, MODE)
                    with torch.no_grad():
                        loss, loss_parts = planner.training_loss(batch.clean_trajectory, batch.context,
                            timesteps=timesteps, noise=noise, drop_conditioning=False, training=False)
                        prediction = planner.plan(sample["states"], sample["history"], sample["neighbors"], sample["raster"],
                            history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"],
                            map_condition_mask=encoding_map_mask(sample, MODE), num_samples=1, seed=plan_seed)
                    for key in ("xy", "yaw", "vel", "velocity", "time", "sample_states", "selected_sample"):
                        value = getattr(prediction, key).detach().cpu().numpy()
                        require(np.isfinite(value).all(), "Nonfinite original prediction; retain partial output")
                        arrays[name + "_" + key] = value
                    require(bool(torch.isfinite(loss)), "Nonfinite original clean loss")
                    future_xy = arrays[name + "_xy"][:, 1:]
                    window_metrics = []
                    for row_index in range(len(chosen)):
                        single = {key: value[row_index:row_index + 1] for key, value in sample.items()}
                        window_metrics.append(offline_plan_metrics(corpus, single, future_xy[row_index:row_index + 1]))
                    per_model[name].extend(window_metrics)
                    losses[name].append((float(loss), len(chosen)))
                    metadata["models"][name] = {"clean_loss": float(loss),
                        "loss_components": {key: float(value) for key, value in loss_parts.items()},
                        "batch_metrics": offline_plan_metrics(corpus, sample, future_xy), "per_window_metrics": window_metrics}
                filename = f"{domain}_{stratum}_{start:04d}.npz"
                with (output / filename).open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                artifact_rows.append(record(output / filename))
                write_json(output / (filename + ".json"), metadata)
                artifact_rows.append(record(output / (filename + ".json")))
                batch_rows.append(metadata)
                print(json.dumps({"domain": domain, "stratum": stratum, "start": start, "windows": len(chosen)}, allow_nan=False), flush=True)
            summaries[domain][stratum] = {name: summarize(per_model[name], losses[name]) for name in models}
    require(private["union"] == union.generator.bit_generator.state
            and all(private[key] == corpus.generator.bit_generator.state for key, corpus in union.domains.items()),
            "Explicit evaluation indices consumed corpus private RNG")
    require(not torch.cuda.is_initialized(), "CPU comparison initialized CUDA")
    # Recheck closed evidence after actual execution; no training/source mutation.
    admit(registration, audit_sha)
    report = {"schema": "official_orca.paired_validation_actual_comparison.v1", "status": "passed",
              "registration": record(HERE / "registration.json"), "selection": registration["selection"],
              "terminal_audit": audit_record, "NEW": audit["checkpoint"], "OLD": registration["OLD"],
              "NEW_inference_weight_source": "EMA_model_key_verified_by77025",
              "config": asdict(old.config), "normalization": normalization, "results": summaries,
              "main_denominator": 200, "separate_diagnostic_denominator": 504,
              "diagnostic_overlap_with_main": {name: selection["domains"][name]["diagnostic_overlap_with_main"] for name in DOMAINS},
              "all_selected_windows_retained": True, "private_corpus_RNG_unchanged": True,
              "explicit_paired_loss_timestep_noise": True, "paired_plan_local_generator_seeds": True,
              "different_hardware_bitwise_claimed": False, "author_benchmark_or_closed_loop_physical_effect_claimed": False,
              "artifacts": artifact_rows, "batch_records": batch_rows, "CPU_only": True,
              "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
              "elapsed_seconds": time.monotonic() - started}
    write_json(output / "comparison.json", report)
    print(json.dumps({"status": "passed", "report": record(output / "comparison.json")}, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare-selection", "evaluate"), required=True)
    parser.add_argument("--terminal-audit-sha256")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/diffusion_amp/formal_EMA40000_vs_63e_validation_v1")
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "This entry requires CUDAhidden CPU")
    if args.mode == "prepare-selection":
        prepare()
    else:
        require(args.terminal_audit_sha256 is not None, "Actual terminal audit SHA argument required")
        evaluate(args.output_dir, args.terminal_audit_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
