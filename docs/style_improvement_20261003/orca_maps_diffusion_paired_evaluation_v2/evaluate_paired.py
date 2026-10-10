"""Prepared paired new-validation comparison using original935 Diffusion math."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REGISTRATION = HERE / "registration.json"
REGISTRATION_SHA = "57c652b52f89e92b5e2d76292bf66c17766c4241b61aa1077aabadc2897f7cff"


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_registration():
    if record(REGISTRATION)["sha256"] != REGISTRATION_SHA:
        raise ValueError("Frozen paired comparison registration changed")
    registration = json.loads(REGISTRATION.read_text())
    if registration["schema"] != "local_orca_maps.prepared_paired_validation_evaluation.v1":
        raise ValueError("Registered paired comparison required")
    for row in registration["sources"].values():
        if record(row["path"]) != row:
            raise ValueError("Registered comparison source changed: " + row["path"])
    selected = registration["window_selection"]
    if len(selected) != 100 or len({row["scene_id"] for row in selected}) != 100:
        raise ValueError("Exactly100 preselected validation scenes/windows required")
    for row in selected:
        for key in ("episode", "map", "obstacles"):
            if record(row[key]["path"]) != row[key]:
                raise ValueError("Preselected validation input changed")
    return registration


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--new-checkpoint-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registration = read_registration()
    if args.prepare_only:
        if args.new_checkpoint_sha256 or args.output:
            parser.error("Preparation does not evaluate models or outputs")
        print(json.dumps({"status": "prepared_not_executed", "windows": 100, "scenes": 100,
                          "future_effect": False, "checkpoint_loaded": False, "GPU_used": False,
                          "SDK_started": False, "sealed34_outputs_read": False}))
        return 0
    if not args.new_checkpoint_sha256 or args.output != Path(registration["output"]):
        parser.error("Actual sealed final5000 SHA and registered output required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This comparison is CPU only")
    if args.output.exists():
        raise FileExistsError("Retain first actual comparison output/failure")
    future = registration["future_checkpoint"]
    new_record = record(future["path"])
    if new_record["sha256"] != args.new_checkpoint_sha256 or Path(future["path"]).stat().st_mode & 0o222:
        raise ValueError("Actual readonly selected final5000 checkpoint required")
    owner = json.loads(Path(registration["sources"]["actual_training_owner_receipt"]["path"]).read_text())
    seal = json.loads(Path(registration["sources"]["actual_terminal_root_seal"]["path"]).read_text())
    training = json.loads(Path(registration["sources"]["training_registration"]["path"]).read_text())
    if (owner.get("schema") != "local_orca_maps.actual_diffusion_training_launch.v1"
            or owner.get("status") != "closed" or type(owner.get("exit_code")) is not int or owner["exit_code"] != 0
            or owner.get("child_exit_observed") is not True or owner.get("driver_completed") is not True
            or owner.get("command") != training["command"]
            or owner.get("selected_source_before") != owner.get("selected_source_after")
            or owner.get("selected_source_after", {}).get("manifest") != registration["sources"]["isolated_source_manifest"]
            or owner.get("selected_source_after", {}).get("regular_files") != 935
            or owner.get("selected_source_after", {}).get("all_declared_files_verified") is not True
            or seal.get("schema") != "local_orca_maps.actual_diffusion_terminal_root_seal.v1"
            or seal.get("actual_root_session_exit0") is not True or seal.get("completed_optimizer_updates") != 5000
            or seal.get("selected_source_before_after_equal") is not True or seal.get("all5000_metric_rows_retained") is not True
            or {key: seal["exclusive_terminal_copy"][key] for key in ("path", "bytes", "sha256")} != new_record
            or Path(future["path"]).stat().st_nlink != 1):
        raise ValueError("Actual closed source-bound training and independent sealed checkpoint copy required")
    completed = json.loads(Path(registration["future_training_result"]).read_text())
    if (completed.get("status") != "completed" or completed.get("completed_step") != 5000
            or completed.get("snapshot") != registration["trainer_snapshot_original_path"] or completed.get("checkpoint_sha256") != args.new_checkpoint_sha256
            or completed.get("frozen_initialization_unchanged") is not True):
        raise ValueError("Actual completed final5000 training result required")
    args.output.mkdir(parents=True)
    write(args.output / "actual_inputs.json", {"registration": record(REGISTRATION), "new_checkpoint": new_record,
          "old_checkpoint": registration["sources"]["old_model"],
          "actual_training_result": record(registration["future_training_result"]),
          "actual_closed_training_owner": registration["sources"]["actual_training_owner_receipt"],
          "actual_terminal_root_seal": registration["sources"]["actual_terminal_root_seal"]})
    launcher = load("paired935_import_owner", registration["sources"]["launcher"]["path"])
    launcher.verify()
    ledger = launcher.ImportLedger(args.output / "CPU_imports.json")
    launcher.activate(ledger)
    status = "failed"
    try:
        import numpy as np
        import torch

        from CrowdSim.humanoid.diffusion import PaperDiffusionPlanner, PaperDiffusionTrainBatch
        from CrowdSim.humanoid.orca_corpus import ORCACorpus
        from scripts import train_orca_diffusion as mathematics

        if record(mathematics.__file__) != registration["sources"]["original_metrics_entry"]:
            raise ValueError("Original metric function import escaped935")
        torch.set_num_threads(4)
        corpus = ORCACorpus(registration["sources"]["manifest"]["path"], "validation", device="cpu", seed=registration["seed"])
        indices = [row["global_window_index"] for row in registration["window_selection"]]
        common = corpus.diffusion_batch(100, indices=indices)
        for index, row in enumerate(registration["window_selection"]):
            if (common["scene_ids"][index] != row["scene_id"] or common["episode_ids"][index] != row["episode_id"]
                    or int(common["agent_ids"][index]) != row["agent_id"] or int(common["frame_indices"][index]) != row["frame"]):
                raise ValueError("Original corpus explicit selection differs from preregistration")
        numpy_common = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                        for key, value in common.items()}
        np.savez_compressed(args.output / "common_inputs.npz", **numpy_common)
        batches, configurations = [], []
        for model_name, checkpoint in (("old63e7", registration["sources"]["old_model"]["path"]), ("expanded_final5000", future["path"])):
            planner = PaperDiffusionPlanner(checkpoint_path=checkpoint, device="cpu", compile_forward=False)
            configurations.append(asdict(planner.config))
            if planner.trained_steps != 5000 or planner.config.diffusion_steps != 100:
                raise ValueError("Original100-step, preselected trained5000 models required")
            if len(configurations) == 2 and configurations[0] != configurations[1]:
                raise ValueError("Paired model architecture/configuration differs")
            model_rows = []
            for start in range(0, 100, registration["batch_size"]):
                stop = min(start + registration["batch_size"], 100)
                sample = {key: value[start:stop] for key, value in common.items()}
                batch = PaperDiffusionTrainBatch.from_world(planner, sample["states"], sample["history"], sample["neighbors"],
                    sample["raster"], sample["future"], history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"])
                generator = torch.Generator(device="cpu").manual_seed(registration["loss_seed"] + start)
                timesteps = torch.randint(100, (stop - start,), generator=generator)
                noise = torch.randn((stop - start, 52, 5), generator=generator)
                with torch.no_grad():
                    loss, metrics = planner.training_loss(batch.clean_trajectory, batch.context, timesteps=timesteps,
                        noise=noise, drop_conditioning=False, training=False)
                    prediction = planner.plan(sample["states"], sample["history"], sample["neighbors"], sample["raster"],
                        history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"],
                        seed=registration["plan_seed"] + start)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite paired validation loss; first failure retained")
                cpu_sample = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
                original = mathematics.offline_plan_metrics(corpus, cpu_sample, prediction.xy[:, 1:].cpu().numpy())
                model_rows.append({"start": start, "stop": stop, "validation_loss": float(loss),
                    "original_loss_metrics": {key: float(value) for key, value in metrics.items()},
                    "original_open_loop_metrics": original,
                    "loss_seed": registration["loss_seed"] + start, "plan_seed": registration["plan_seed"] + start})
                np.savez_compressed(args.output / f"{model_name}_batch_{start:03d}.npz",
                    window_indices=np.asarray(indices[start:stop]), timesteps=timesteps.numpy(), noise=noise.numpy(),
                    xy=prediction.xy.numpy(), yaw=prediction.yaw.numpy(), velocity=prediction.velocity.numpy(),
                    sample_states=prediction.sample_states.numpy(), selected_sample=prediction.selected_sample.numpy())
                write(args.output / f"{model_name}_batch_{start:03d}.json", model_rows[-1])
            batches.append({"model": model_name, "batches": model_rows,
                "mean_validation_loss": sum(row["validation_loss"] * (row["stop"] - row["start"]) for row in model_rows) / 100})
            del planner
        if torch.cuda.is_initialized() or any(name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules):
            raise RuntimeError("Paired offline evaluation started CUDA/SDK")
        write(args.output / "comparison.json", {"schema": "local_orca_maps.actual_paired_validation_evaluation.v1",
            "status": "passed", "CPU_only": True, "GPU_used": False, "SDK_started": False,
            "registration": record(REGISTRATION), "windows": 100, "unique_scenes": 100,
            "model_results": batches, "shared_loss_timesteps_noise": True, "shared_plan_seed_per_batch": True,
            "all_windows_and_collision_failures_retained": True, "original_geometry_metrics_unchanged": True,
            "offline_is_closed_loop_quality_proof": False, "promotion_claimed": False, "sealed34_outputs_read": False})
        status = "passed"
    finally:
        ledger.finish(status)
    print(json.dumps({"status": "passed", "output": str(args.output), "GPU_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
