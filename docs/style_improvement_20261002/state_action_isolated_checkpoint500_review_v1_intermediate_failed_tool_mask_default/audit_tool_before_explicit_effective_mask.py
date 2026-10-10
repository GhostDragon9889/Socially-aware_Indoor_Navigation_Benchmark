"""Restricted CPU review of actual isolated state-action training milestones.

Reads sealed checkpoints and an as-observed synchronous v3 import ledger.
Never writes training artifacts, replays physics, or restores CUDA RNG.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_v1"
CODE = BUNDLE / "code"
TRAIN = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_isolated_v1"
SNAPSHOTS = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_isolated_snapshots_v1"
MANIFEST_SHA = "f2af12c16bb7f5a0808657e4f2589bc3c4db91370c4201086c5b4416bdafb8fa"
LAUNCHER_SHA = "2f2a8137599f83a259073e210bac4c283303bae03c765682aefb4885bc24d3a5"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("CPU-only and fresh output directory required")
    args.output.mkdir(parents=True)
    sys.path[:0] = [str(CODE), str(CODE / "src")]
    os.environ["PROTOMOTIONS_ASSET_ROOT"] = str(CODE / "protomotions/data/assets")
    import numpy as np
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.application.configuration import load_amp_profile
    from CrowdSim.humanoid.application.settings import TrainingConfig
    from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
    from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
    from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig

    torch.set_num_threads(4)
    checks, rows = [], []

    def check(name, condition, **detail):
        checks.append({"name": name, "passed": bool(condition), **detail})
        if not condition:
            raise ValueError(name)

    def finite(value):
        if isinstance(value, torch.Tensor):
            return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)

    def same(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
        return type(left) is type(right) and left == right

    report = {"schema": "state_action.actual_training_checkpoint_audit.v1", "status": "failed",
        "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
        "checks": checks, "checkpoints": rows, "auditor": record(__file__)}
    try:
        source = json.loads((BUNDLE / "source_manifest_v3.json").read_text())
        source_record, launcher_record = record(BUNDLE / "source_manifest_v3.json"), record(BUNDLE / "launch_isolated_v3.py")
        check("exact_known_actual_v3_manifest_and_launcher", source_record["sha256"] == MANIFEST_SHA and launcher_record["sha256"] == LAUNCHER_SHA)
        declared = {item["path"]: item for item in source["files"]}
        actual_names = {str(path.relative_to(BUNDLE)) for path in CODE.rglob("*") if path.is_file() and not path.is_symlink()}
        check("exact926_copied_source_set", len(declared) == 926 and set(declared) == actual_names)
        for name, expected in declared.items():
            path = BUNDLE / name
            actual = record(path)
            check("actual_copied_source:" + name, not path.is_symlink() and path.resolve().is_relative_to(CODE) and
                  actual["sha256"] == expected["sha256"] and actual["bytes"] == expected["bytes"] and not path.stat().st_mode & 0o222)
        for link in source["shared_resource_links"]:
            path = BUNDLE / link["path"]
            check("exact_shared_readonly_resource_link:" + link["path"], path.is_symlink() and str(path.resolve()) == link["target"])
        for resource in source["resources"]:
            stat = Path(resource["path"]).stat()
            check("shared_resource_stat:" + resource["path"], stat.st_size == resource["bytes"] and stat.st_mtime_ns == resource["mtime_ns"])
        # Copy the atomic JSON and precisely its committed event prefix. New
        # observed imports can appear later without rewriting this audit.
        ledger_path = BUNDLE / "fullfresh2000_v3_GPU_imports.json"
        ledger_bytes = ledger_path.read_bytes()
        ledger = json.loads(ledger_bytes)
        (args.output / "GPU_import_ledger_as_observed.json").write_bytes(ledger_bytes)
        all_event_lines = Path(ledger["event_ledger"]).read_text().splitlines()
        event_lines = all_event_lines[:ledger["event_count"]]
        events = [json.loads(line) for line in event_lines]
        with (args.output / "GPU_import_events_as_observed.jsonl").open("x") as stream:
            stream.write("\n".join(event_lines) + "\n")
        check("actual_live_synchronous_ledger_header", ledger["schema"] == "paper_amp.actual_owned_import_ledger.v3" and
              ledger["source_manifest_sha256"] == MANIFEST_SHA and ledger["launcher_sha256"] == LAUNCHER_SHA and
              ledger["all_observed_owned_modules_isolated"] is True and ledger["code"] == str(CODE))
        check("exact_committed_event_sequence", [event["sequence"] for event in events] == list(range(1, ledger["event_count"] + 1)))
        observed = {event["module"]: event for event in events if event["event"] == "owned_execution"}
        check("atomic_ledger_matches_committed_observations", len(ledger["owned_modules"]) == len(observed) > 0)
        for row in ledger["owned_modules"]:
            path = Path(row["path"])
            relative = "code/" + str(path.relative_to(CODE))
            expected = declared[relative]
            check("actual_GPU_owned_execution:" + row["module"], path.is_relative_to(CODE) and
                  row["execution_observed"] is True and row["sha256"] == expected["sha256"] and row["bytes"] == expected["bytes"] and
                  observed[row["module"]]["sha256"] == row["sha256"] and observed[row["module"]]["path"] == str(path))
        check("actual_runner_entry_observed_before_runpy", any(event["event"] == "runner_entry_selected" and
              event["path"] == str(CODE / "scripts/train_paper_amp.py") and event["sha256"] == declared["code/scripts/train_paper_amp.py"]["sha256"] for event in events))
        for name, module in tuple(sys.modules.items()):
            if name.startswith(("CrowdSim", "protomotions")) and getattr(module, "__file__", None):
                check("reviewer_isolated_module:" + name, Path(module.__file__).resolve().is_relative_to(CODE))
        transport = json.loads((SNAPSHOTS / "ledger.json").read_text())
        profile = load_amp_profile(CODE / "configs/diffusion_amp/isolated_state_action_absolute.yaml")
        expected_config = json.loads(json.dumps(asdict(TrainingConfig(**profile["training"]))))
        selected_paths = [path.resolve() for path in args.checkpoint]
        check("only_unique_immutable_transport_milestones", len(set(selected_paths)) == len(selected_paths) and
              all(path.parent == SNAPSHOTS and path.name in {"step_500.pt", "step_1000.pt", "step_2000.pt"} for path in selected_paths))
        metrics_lines = (TRAIN / "metrics.jsonl").read_text().splitlines()
        requested_updates = [int(path.stem.split("_")[-1]) for path in selected_paths]
        maximum_update = max(requested_updates)
        metrics = [json.loads(line) for line in metrics_lines[:maximum_update]]
        check("actual_committed_metric_update_prefix", [row["update_count"] for row in metrics] == list(range(1, maximum_update + 1)) and
              all(row["physical_transitions"] == row["update_count"] * 8192 for row in metrics))
        with (args.output / "training_metrics_committed_prefix.jsonl").open("x") as stream:
            stream.write("\n".join(metrics_lines[:maximum_update]) + "\n")
        for path, update in zip(selected_paths, requested_updates, strict=True):
            prefix = f"step{update}:"
            checkpoint = record(path)
            transport_rows = [row for row in transport["snapshots"] if Path(row["path"]).resolve() == path]
            check(prefix + "exact_sole_watcher_sealed_bytes", len(transport_rows) == 1 and
                  checkpoint["sha256"] == transport_rows[0]["sha256"] and checkpoint["bytes"] == transport_rows[0]["bytes"] and
                  transport_rows[0]["identical_source_bytes_verified"] is True)
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            cfg, runner = state["config"], state["runner_state"]
            check(prefix + "complete_restricted_payload", state["format"] == "navisaaclab_method_reconstruction_training" and
                  state["version"] == PaperAMPTrainer.checkpoint_version and state["physical_runner_state_included"] is True)
            check(prefix + "actual_committed_updates_transitions_physics_count", type(state["update_count"]) is int and
                  state["update_count"] == update and state["physical_transition_count"] == update * 8192 and runner["control_step"] == update * 32)
            check(prefix + "state_action2019_independent_D_and_RMS", cfg["discriminator_input"] == "state_action" and
                  cfg["warm_start_discriminator"] is False and cfg["amp_normalization"] == "per_history_coordinate" and
                  tuple(state["model"]["discriminator.0.weight"].shape) == (1024, 2019) and
                  state["provenance"]["initialization"]["published_state_only_discriminator_transferred"] is False and
                  state["provenance"]["initialization"]["published_amp_history_moments_transferred"] is False and
                  state["provenance"]["initialization"]["normalizer_counts_before_cap"]["amp_normalizer"] == 0.)
            check(prefix + "exact_profile_training_settings", json.loads(json.dumps(cfg)) == expected_config and
                  cfg["direction_mode"] == "backwards_penalty" and cfg["expert_sampling"] is None and "expert_sampling" not in runner)
            check(prefix + "every_training_tensor_finite", finite(state))
            optimizer_steps = {}
            for name, metric_name in (("optimizer", "optimizer_minibatches"), ("discriminator_optimizer", "discriminator_minibatches")):
                optimizer = state[name]
                ids = [item for group in optimizer["param_groups"] for item in group["params"]]
                steps = {float(item["step"]) for item in optimizer["state"].values()}
                expected_steps = sum(item[metric_name] for item in metrics[:update])
                check(prefix + name + "_actual_Adam_prefix_steps", len(ids) == len(set(ids)) and set(ids) == set(optimizer["state"]) and steps == {expected_steps})
                check(prefix + name + "_nonzero_moments", any(bool((item["exp_avg"] != 0).any()) for item in optimizer["state"].values()))
                optimizer_steps[name] = expected_steps
            loaded, loaded_runner = PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
            check(prefix + "real_isolated_loader_model_optimizers_normalizers_RNG_exact", same(loaded.model.state_dict(), state["model"]) and
                  same(loaded.optimizer.state_dict(), state["optimizer"]) and same(loaded.discriminator_optimizer.state_dict(), state["discriminator_optimizer"]) and
                  all(same(getattr(loaded, name).state_dict(), value) for name, value in state["normalizers"].items()) and
                  same(loaded._pending_rng_state, state["rng"]) and same(loaded_runner, runner))
            check(prefix + "real_loader_replay_exact", same(loaded.policy_replay.state_dict(), state["policy_replay"]))
            del loaded, loaded_runner
            count_expected = {"self_normalizer": 200000 + update * 8192, "task_normalizer": 200000 + update * 8192,
                "value_normalizer": 200000 + update * 8192, "amp_normalizer": update * (8192 + 4096)}
            for name, count in count_expected.items():
                norm = state["normalizers"][name]
                check(prefix + "actual_RMS_count:" + name, float(norm["count"]) == count and bool((norm["variance"] >= 0).all()))
            replay, mask = state["policy_replay"], torch.tensor(cfg["action_mask"])
            check(prefix + "real_replay_ring_and_applied57_contract", replay["accepted"] == update * 8192 and replay["capacity"] == 65536 and
                  replay["size"] == 65536 and replay["cursor"] == 0 and replay["history"].shape == (65536, 10, 195) and
                  replay["actions"].shape == (65536, 69) and mask.count_nonzero() == 57 and not bool((replay["actions"][:, mask == 0] != 0).any()))
            physical_shapes = {"root_pos": (256, 3), "root_rot": (256, 4), "root_vel": (256, 3), "root_ang_vel": (256, 3), "dof_pos": (256, 69), "dof_vel": (256, 69)}
            check(prefix + "actual_physical_common_XYZW_history_same_episode", set(runner["physical"]) == set(physical_shapes) and
                  all(tuple(runner["physical"][name].shape) == shape for name, shape in physical_shapes.items()) and
                  torch.allclose(runner["physical"]["root_rot"].norm(dim=-1), torch.ones(256), atol=1e-4, rtol=0) and
                  runner["amp_history"].shape == (256, 10, 195) and torch.equal(runner["amp_history_episode_ids"], runner["episode_ids"][:, None].expand(256, 10)))
            check(prefix + "exact_frozen_training_clock", runner["runner_contract"] == {"replan_steps": 15, "episode_steps": 300,
                  "planner_batch_size": 64, "trajectory_source": "mixed_amass_procedural", "planning_clock": "synchronized"})
            rng = state["rng"]
            random.Random().setstate(rng["python"])
            numpy_rng = rng["numpy"]
            np.random.RandomState().set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
            torch.Generator(device="cpu").set_state(rng["torch_cpu"])
            check(prefix + "saved_CPU_RNG_valid_CUDA_bytes_retained_only", len(rng["torch_cuda"]) == 1 and
                  rng["torch_cuda"][0].dtype == torch.uint8 and rng["torch_cuda"][0].ndim == 1)
            corpus = AMASSCorpus(ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json", "train", device="cpu", seed=7, dof_velocity_frame="child_joint")
            corpus.generator.set_state(runner["corpus_rng"])
            curriculum = MixedTrajectoryCurriculum(corpus, 256, device="cpu", bounds_xy=(0., 0., 192., 192.), tail_policy="linear_velocity",
                procedural_config=PacerTrajectoryConfig(mode="smooth", speed_max_mps=2., scalar_accel_max_mps2=1.5, vector_accel_max_mps2=1.5), seed=7)
            curriculum.load_state_dict(runner["curriculum_state"])
            ids = torch.arange(256)
            origins = torch.stack((48 + (ids // 16) * 4, 48 + (ids % 16) * 4), -1).float()
            curriculum.validate_runner_state(runner["age"], runner["plan_age"], runner["plans"], runner["reference_ids"], runner["episode_ids"], replan_steps=15, origins_xy=origins)
            check(prefix + "complete_actual_curriculum_native151_and_phase_restore", same(curriculum.state_dict(), runner["curriculum_state"]))
            statistics = dict(curriculum.statistics)
            check(prefix + "actual50percent_actor_transition_exposure", statistics["control_steps"] == update * 32 and
                  statistics["amass_actor_transitions"] == statistics["procedural_actor_transitions"] == update * 4096)
            check(prefix + "actual_bank_and_manifest_provenance", state["provenance"]["dataset"]["source_sha256"] == corpus.entry["sha256"] and
                  state["provenance"]["dataset"]["manifest_sha256"] == record(corpus.manifest_path)["sha256"])
            cpu_rng_before = torch.get_rng_state().clone()
            order = torch.tensor([255, 1, 128, 0])

            def next_draws():
                reset = curriculum.sample_reference(order, history_frames=10)
                expert = corpus.sample_reference(6, history_frames=10)
                plans = curriculum.build_plan(SimpleNamespace(**runner["physical"]), runner["age"], order, runner["episode_ids"])
                return ({"state": vars(reset[0]), "history": vars(reset[1]), "next": reset[2], "ids": reset[3], "frames": reset[4]},
                    {"state": vars(expert[0]), "history": vars(expert[1]), "next": expert[2], "ids": expert[3], "frames": expert[4]},
                    {"xy": plans.xy, "native_xy": plans.native_xy}, corpus.generator.get_state(),
                    curriculum.amass.corpus.generator.get_state(), curriculum.procedural.generator.get_state())

            first = next_draws()
            corpus.generator.set_state(runner["corpus_rng"])
            curriculum.load_state_dict(runner["curriculum_state"])
            second = next_draws()
            check(prefix + "actual_next_subset_reset_expert_plan_three_RNG_repeat", same(first, second) and torch.equal(cpu_rng_before, torch.get_rng_state()))
            metric = metrics[update - 1]
            row = {"checkpoint": checkpoint, "updates": update, "transitions": update * 8192, "D_input": 2019,
                "full_payload_resume_validated": True, "public_D_or_AMP_RMS_transferred": False,
                "control_steps": update * 32, "group_exposure": statistics, "optimizer_steps": optimizer_steps,
                "RMS_counts": count_expected, "replay": {key: replay[key] for key in ("size", "capacity", "accepted", "cursor")},
                "cumulative_training_falls_at_milestone": metric["fall_resets"], "cumulative_training_time_limits_at_milestone": metric["time_limit_resets"],
                "physics_resume_limit": runner["physics_resume"], "actual_metric_row": metric,
                "CUDA_RNG_executed_by_reviewer": False}
            rows.append(row)
        check("reviewer_never_started_CUDA_Kit_USD", not torch.cuda.is_initialized() and
              not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr"} for name in sys.modules))
        report.update(status="passed", actual_observed_owned_imports_isolated=True,
            source_manifest=source_record, launcher=launcher_record, copied_files_hashed=926,
            observed_imports=len(ledger["owned_modules"]), observed_ledger_runtime_finalization=ledger["runtime_finalization_observed"],
            live_ledger_full_runtime_finalization_not_inferred=True,
            shared_large_resource_full_SHA_recomputed=False,
            CPU_next_logical_draws_exact=True, physics_trajectory_bitwise_resume_verified=False,
            limitations=["No solver/contact-cache serialization or physical replay; no bitwise PhysX resume claim.",
                "CUDA RNG bytes preserved, not restored or executed on CPU.",
                "Synchronous GPU ledger validates actual observed owned imports; unobserved future imports/finalization are not fabricated.",
                "All926 copied files hashed; large shared resources checked by declared size/mtime/link targets, not rehashed here.",
                "Training fall counters are cumulative training events, not validation/navigation outcomes."])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["check_count"] = len(checks)
        write(args.output / "audit.json", report)
    print(json.dumps({"status": report["status"], "checks": len(checks), "updates": [row["updates"] for row in rows]}))


if __name__ == "__main__":
    run()
