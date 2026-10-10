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

from state_prior_checks import read_committed_metric_prefix, validate_actual_result, validate_policy, verify_launch_registration, verify_source_archive, verify_milestone_launch_registration, validate_state_prior

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_pacer_reward_v2"
CODE = BUNDLE / "code"
TRAIN = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_pacer_reward_isolated_v1"
SNAPSHOTS = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_pacer_reward_isolated_snapshots_v1"
PILOT = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_pacer_reward_isolated_pilot_v1"
MANIFEST_SHA = "1fd9b5e14dc377f753b04ecfe6ce93054b300d6f46c5cb341aa466f6ac887a21"
LAUNCHER_SHA = "6d56a3efb73534ca2cd54cc5ab82063fbb102b4a1a58f413c2421c26be199914"


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
    parser.add_argument("--phase", choices=("pilot4", "pilot8", "milestone", "terminal2000"), required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seal", type=Path, action="append", default=[])
    args = parser.parse_args()
    if (args.phase == "milestone") != (args.result is None):
        raise ValueError("Milestones have no terminal result; all closed phases require actual --result")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("CPU-only and fresh output directory required")
    args.output.mkdir(parents=True)
    sys.path[:0] = [str(CODE), str(CODE / "src")]
    os.environ["PROTOMOTIONS_ASSET_ROOT"] = str(CODE / "protomotions/data/assets")
    import numpy as np
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.amp.observations import PACER_DOF_NAMES
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

    report = {"schema": "pacer_reward.actual_training_checkpoint_audit.v1", "status": "failed",
        "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
        "checks": checks, "checkpoints": rows, "auditor": record(__file__)}
    try:
        source = json.loads((BUNDLE / "source_manifest.json").read_text())
        source_record, launcher_record = record(BUNDLE / "source_manifest.json"), record(BUNDLE / "launch_isolated.py")
        check("exact_known_actual_v3_manifest_and_launcher", source_record["sha256"] == MANIFEST_SHA and launcher_record["sha256"] == LAUNCHER_SHA)
        declared = {item["path"]: item for item in source["files"]}
        actual_names = {str(path.relative_to(BUNDLE)) for path in CODE.rglob("*") if path.is_file() and not path.is_symlink()}
        check("exact935_copied_source_set", len(declared) == 935 and set(declared) == actual_names)
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
        report["source_full_archive_independent"] = verify_source_archive(BUNDLE, source, record, check)
        # Copy the atomic JSON and precisely its committed event prefix. New
        # observed imports can appear later without rewriting this audit.
        ledger_path = BUNDLE / {"pilot4": "pilot4_GPU_imports.json", "pilot8": "resume4to8_GPU_imports.json", "terminal2000": "fullfresh2000_GPU_imports.json", "milestone": "fullfresh2000_GPU_imports.json"}[args.phase]
        ledger_bytes = ledger_path.read_bytes()
        ledger = json.loads(ledger_bytes)
        (args.output / "GPU_import_ledger_as_observed.json").write_bytes(ledger_bytes)
        all_event_lines = Path(ledger["event_ledger"]).read_text().splitlines()
        event_lines = all_event_lines[:ledger["event_count"]]
        events = [json.loads(line) for line in event_lines]
        with (args.output / "GPU_import_events_as_observed.jsonl").open("x") as stream:
            stream.write("\n".join(event_lines) + "\n")
        report["actual_import_ledger_committed_prefix"] = {
            "source_path": str(ledger_path), "event_count": ledger["event_count"],
            "observed_owned_module_count": len(ledger["owned_modules"]),
            "atomic_snapshot": record(args.output / "GPU_import_ledger_as_observed.json"),
            "committed_events": record(args.output / "GPU_import_events_as_observed.jsonl"),
            "runtime_finalization_observed": ledger["runtime_finalization_observed"],
        }
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
        result = None if args.result is None else json.loads(args.result.read_text())
        if args.phase in {"terminal2000", "milestone"}:
            transport = json.loads((SNAPSHOTS / "ledger.json").read_text())
        else:
            seals = [json.loads(path.read_text()) for path in args.seal]
            seal_records = [row for seal in seals for row in seal["files"]]
            check("all_supplied_pilot_seals_are_closed_actual_processes", len(seals) == (2 if args.phase == "pilot8" else 1) and all(seal["schema"] == "state_prior.actual_pilot_seal.v1" and seal["status"] == "sealed_after_actual_exit" and seal["actual_launch"]["status"] == "closed" and seal["actual_launch"]["exit_code"] == 0 for seal in seals))
            check("result_and_metrics_are_root_sealed", args.result.resolve() in {Path(row["path"]).resolve() for row in seal_records} and args.metrics.resolve() in {Path(row["path"]).resolve() for row in seal_records})
            for sealed in seal_records:
                check("root_sealed_actual_pilot_artifact:" + sealed["path"], all(record(sealed["path"])[key] == sealed[key] for key in ("path", "bytes", "sha256")) and not Path(sealed["path"]).stat().st_mode & 0o222)
            transport = {"snapshots": [row for row in seal_records if row["path"].endswith(".pt")]}
        report["root_prelaunch_registration"] = (verify_milestone_launch_registration(BUNDLE, MANIFEST_SHA, ledger_path, record, check)
            if args.phase == "milestone" else verify_launch_registration(BUNDLE, args.phase, MANIFEST_SHA, ledger_path, result, record, check))
        profile = load_amp_profile(CODE / "configs/diffusion_amp/isolated_terrain_state_prior_pacer_reward_absolute.yaml")
        expected_config = json.loads(json.dumps(asdict(TrainingConfig(**profile["training"]))))
        if expected_config["action_mask"] is None:
            # The profile leaves automatic mask derivation to the live SMPL
            # world; compare the explicit declared canonical57 contract.
            expected_config["action_mask"] = [int(name.rsplit("_", 1)[0] not in
                {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}) for name in PACER_DOF_NAMES]
        reward_contract = {name: expected_config[name] for name in ("amp_reward_scale", "amp_reward_weight", "task_reward_weight", "direction_mode", "amp_probability_epsilon")}
        check("registered_pacer_reward_five_fields_exact", all(type(reward_contract[name]) is type(value) and reward_contract[name] == value for name, value in {"amp_reward_scale": 2.0, "amp_reward_weight": 0.5, "task_reward_weight": 0.5, "direction_mode": "backwards_penalty", "amp_probability_epsilon": 0.0001}.items()))
        report["reward_contract"] = reward_contract
        selected_paths = [path.resolve() for path in args.checkpoint]
        check("only_unique_immutable_transport_milestones", len(set(selected_paths)) == len(selected_paths) and
              all(path.parent == (SNAPSHOTS if args.phase in {"terminal2000", "milestone"} else PILOT)
                  and path.name in ({"step_500.pt", "step_1000.pt", "step_2000.pt"} if args.phase == "milestone" else {"step_2000.pt"} if args.phase == "terminal2000" else {"step_4.pt", "step_8.pt"}) for path in selected_paths))
        requested_updates = [int(path.stem.split("_")[-1]) for path in selected_paths]
        maximum_update = max(requested_updates)
        check("explicit_sealed_phase_update_set", (len(requested_updates) == 1 and requested_updates[0] in (500, 1000, 2000))
              if args.phase == "milestone" else requested_updates == {"pilot4": [4], "pilot8": [4, 8], "terminal2000": [2000]}[args.phase])
        report["actual_training_result"] = None if result is None else validate_actual_result(result, args.phase, requested_updates, check)
        report["terminal_training_result"] = None if args.result is None else record(args.result)
        report["milestone_only_admission"] = args.phase == "milestone"
        report["actual_terminal_fresh2000_confirmed"] = args.phase == "terminal2000"
        metrics_lines = read_committed_metric_prefix(args.metrics, maximum_update,
            require_closed=args.phase != "milestone")
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
                  not path.stat().st_mode & 0o222 and checkpoint["sha256"] == transport_rows[0]["sha256"] and checkpoint["bytes"] == transport_rows[0]["bytes"] and
                  (args.phase not in {"terminal2000", "milestone"} or transport_rows[0]["identical_source_bytes_verified"] is True))
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            cfg, runner = state["config"], state["runner_state"]
            check(prefix + "complete_restricted_payload", state["format"] == "navisaaclab_method_reconstruction_training" and
                  state["version"] == PaperAMPTrainer.checkpoint_version and state["physical_runner_state_included"] is True)
            check(prefix + "actual_committed_updates_transitions_physics_count", type(state["update_count"]) is int and
                  state["update_count"] == update and state["physical_transition_count"] == update * 8192 and runner["control_step"] == update * 32)
            state_prior_proof = validate_state_prior(state, check, prefix)
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
            protected_proof = validate_policy(state, path, loaded, check, prefix)
            check(prefix + "trained_action_columns_preserved_by_real_load", torch.equal(loaded.model.discriminator[0].weight[:, 1950:], state["model"]["discriminator.0.weight"][:, 1950:]))
            del loaded, loaded_runner
            count_expected = {"self_normalizer": 200000 + update * 8192, "task_normalizer": 200000 + update * 8192,
                "value_normalizer": 200000 + update * 8192, "amp_normalizer": 200000 + update * (8192 + 4096)}
            for name, count in count_expected.items():
                norm = state["normalizers"][name]
                check(prefix + "actual_RMS_count:" + name, float(norm["count"]) == count and bool((norm["variance"] >= 0).all()))
            replay, mask = state["policy_replay"], torch.tensor(cfg["action_mask"])
            check(prefix + "real_replay_ring_and_applied57_contract", type(replay["accepted"]) is int and
                  (replay["accepted"] == update * 8192 if update * 8192 <= 65536
                   else 65536 <= replay["accepted"] <= update * 8192) and replay["capacity"] == 65536 and
                  replay["size"] == min(replay["accepted"], 65536) and replay["cursor"] == replay["accepted"] % 65536
                  and replay["history"].shape == (replay["size"], 10, 195) and
                  replay["actions"].shape == (replay["size"], 69) and mask.count_nonzero() == 57 and not bool((replay["actions"][:, mask == 0] != 0).any()))
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
                "full_payload_resume_validated": True, "public_D_or_AMP_RMS_transferred": True, "state_prior_initialization_validated": True,
                "state_prior": state_prior_proof,
                "protected_policy_independent_checks": True, "terrain_policy": protected_proof["policy"],
                "inference_probe": protected_proof,
                "control_steps": update * 32, "group_exposure": statistics, "optimizer_steps": optimizer_steps,
                "RMS_counts": count_expected, "replay": {key: replay[key] for key in ("size", "capacity", "accepted", "cursor")},
                "cumulative_training_falls_at_milestone": metric["fall_resets"], "cumulative_training_time_limits_at_milestone": metric["time_limit_resets"],
                "physics_resume_limit": runner["physics_resume"], "actual_metric_row": metric,
                "CUDA_RNG_executed_by_reviewer": False}
            rows.append(row)
        report["protected_reference_policy_validated"] = True
        report["state_prior_initialization_validated"] = True
        report["public_D_or_AMP_RMS_transferred"] = True
        report["phase"] = args.phase
        report["actual_sealed_results"] = [record(p) for p in args.seal]
        report["actual_closed_metrics"] = None if args.phase == "milestone" else record(args.metrics)
        report["actual_committed_metrics_prefix"] = record(args.output / "training_metrics_committed_prefix.jsonl")
        report["fall_counter_scope"] = "fresh_run_cumulative" if args.phase in {"milestone", "terminal2000"} else "per_segment_reset_counters; first4 and resumed4 reported separately"
        if args.phase == "pilot8":
            sealed_first_metrics = [json.loads(line) for line in (PILOT / "sealed_step4/metrics.jsonl").read_text().splitlines() if line.strip()]
            check("resume_metrics_first4_sealed_prefix_bitwise", metrics[:4] == sealed_first_metrics)
        if args.phase == "pilot8":
            first = torch.load(selected_paths[0], map_location="cpu", weights_only=True, mmap=True)
            second = torch.load(selected_paths[1], map_location="cpu", weights_only=True, mmap=True)
            check("resume8_names_sealed4_actual_source", Path(result["arguments"]["resume"]).resolve() == selected_paths[0])
            check("resume_preserves_public_reference_and_policy_suffix",
                  same(first["provenance"], second["provenance"])
                  and torch.equal(first["task_terrain_statistics_policy"]["suffix_mean"], second["task_terrain_statistics_policy"]["suffix_mean"])
                  and torch.equal(first["task_terrain_statistics_policy"]["suffix_variance"], second["task_terrain_statistics_policy"]["suffix_variance"]))
            check("resume_policy_and_D_genuinely_updated", not torch.equal(first["model"]["actor.4.weight"], second["model"]["actor.4.weight"])
                  and not torch.equal(first["model"]["discriminator.0.weight"], second["model"]["discriminator.0.weight"]))
            del first, second
        later_ledger_bytes = ledger_path.read_bytes()
        later_ledger = json.loads(later_ledger_bytes)
        (args.output / "GPU_import_ledger_after_audit.json").write_bytes(later_ledger_bytes)
        check("later_live_ledger_same_source_monotone_committed_sequence",
              later_ledger["source_manifest_sha256"] == MANIFEST_SHA
              and later_ledger["launcher_sha256"] == LAUNCHER_SHA
              and later_ledger["code"] == str(CODE)
              and type(later_ledger["event_count"]) is int
              and later_ledger["event_count"] >= ledger["event_count"])
        report["actual_import_ledger_later_observation"] = {
            "source_path": str(ledger_path), "event_count": later_ledger["event_count"],
            "observed_owned_module_count": len(later_ledger["owned_modules"]),
            "atomic_snapshot": record(args.output / "GPU_import_ledger_after_audit.json"),
            "runtime_finalization_observed": later_ledger["runtime_finalization_observed"],
            "new_events_after_initial_snapshot_used_as_admission": False,
        }
        check("reviewer_never_started_CUDA_Kit_USD", not torch.cuda.is_initialized() and
              not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr"} for name in sys.modules))
        report.update(status="passed", actual_observed_owned_imports_isolated=True,
            source_manifest=source_record, launcher=launcher_record, copied_files_hashed=935,
            observed_imports=len(ledger["owned_modules"]), observed_ledger_runtime_finalization=ledger["runtime_finalization_observed"],
            live_ledger_full_runtime_finalization_not_inferred=True,
            shared_large_resource_full_SHA_recomputed=False,
            CPU_next_logical_draws_exact=True, physics_trajectory_bitwise_resume_verified=False,
            limitations=["No solver/contact-cache serialization or physical replay; no bitwise PhysX resume claim.",
                "CUDA RNG bytes preserved, not restored or executed on CPU.",
                "Synchronous GPU ledger validates actual observed owned imports; unobserved future imports/finalization are not fabricated.",
                "All935 copied files hashed; large shared resources checked by declared size/mtime/link targets, not rehashed here.",
                "Training fall counters follow explicit fall_counter_scope, not validation/navigation outcomes."])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["check_count"] = len(checks)
        write(args.output / "audit.json", report)
    print(json.dumps({"status": report["status"], "checks": len(checks), "updates": [row["updates"] for row in rows]}))


if __name__ == "__main__":
    run()
