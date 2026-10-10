"""Additional strict CPU checks; this module never loads a simulator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from unittest.mock import patch

PUBLIC_SHA = "bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615"


def read_committed_metric_prefix(path, count, *, require_closed):
    """Read only complete committed lines; a live suffix is not evidence."""
    if type(count) is not int or count <= 0:
        raise ValueError("A positive integer committed metric count is required")
    lines = []
    with Path(path).open() as stream:
        for _ in range(count):
            line = stream.readline()
            if not line.endswith("\n"):
                raise ValueError("Metric prefix is missing or has an incomplete committed line")
            lines.append(line.rstrip("\r\n"))
        if require_closed and stream.read():
            raise ValueError("Closed metric artifact contains rows beyond its declared update budget")
    return lines


def validate_policy(state, path, loaded, check, prefix):
    import numpy as np
    import torch
    from CrowdSim.humanoid.runtime.amp_inference import ReconstructionAMPInference
    from CrowdSim.humanoid.training.terrain_statistics import STATE_KEYS

    saved = state["task_terrain_statistics_policy"]
    check(prefix + "protected_exact_internal_policy_keys", set(saved) == STATE_KEYS)
    check(prefix + "protected_config_calls_global_count_reference_are_separate",
          state["config"]["task_terrain_statistics_mode"] == "preserve_declared_suffix"
          and type(saved["post_update_calls"]) is int
          and saved["post_update_calls"] == state["update_count"]
          and saved["reference_count"] == 61931520001.
          and saved["warm_start_task_count"] == 200000.
          and float(state["normalizers"]["task_normalizer"]["count"])
          == saved["warm_start_task_count"] + state["physical_transition_count"])
    check(prefix + "protected_bound_public_source_provenance",
          saved["reference_initialization"]["sha256"] == PUBLIC_SHA
          and saved["reference_initialization"]["mode"] == "transfer_warm_start_public_pacer_actor_critic"
          and saved["reference_initialization"]["path"] == state["provenance"]["initialization"]["path"])
    digest = hashlib.sha256()
    for name, label in (("suffix_mean", b"mean:f64le:1024\0"),
                        ("suffix_variance", b"variance:f64le:1024\0")):
        tensor = saved[name]
        check(prefix + "protected_suffix_shape_dtype_finite:" + name,
              tensor.shape == (1024,) and tensor.dtype == torch.float64
              and tensor.device.type == "cpu" and torch.isfinite(tensor).all())
        digest.update(label)
        digest.update(tensor.numpy().astype("<f8", copy=False).tobytes())
    check(prefix + "protected_independent_array_SHA", saved["reference_statistics_sha256"] == digest.hexdigest())
    snapshot = Path(__file__).resolve().parents[3] / "docs/style_improvement_20261002/staged_terrain_rms_candidate_v1/public_task_statistics_snapshot.npz"
    check(prefix + "independent_public_suffix_fixture_immutable",
          hashlib.sha256(snapshot.read_bytes()).hexdigest()
          == "fe32378b59ebc11ccae283ff543d6e3f71cd79fc9e9eb8b6be1b85e6a2998d2f")
    with np.load(snapshot, allow_pickle=False) as public:
        check(prefix + "protected_suffix_bitwise_actual_public_statistics_fixture",
              np.array_equal(saved["suffix_mean"].numpy(), public["task_mean"][20:])
              and np.array_equal(saved["suffix_variance"].numpy(), public["task_variance"][20:])
              and public["reference_count"].item() == saved["reference_count"])
    norm = state["normalizers"]["task_normalizer"]
    check(prefix + "protected_saved_suffix_exact_moments_no_replacement",
          torch.equal(saved["suffix_mean"], norm["mean"][20:])
          and torch.equal(saved["suffix_variance"], norm["variance"][20:]))
    check(prefix + "protected_actual_loader_internal_reference_roundtrip",
          loaded.terrain_statistics_policy.describe()["reference_statistics_sha256"] == digest.hexdigest()
          and loaded.terrain_statistics_policy.post_update_calls == state["update_count"])
    private = torch.Generator(device="cpu").manual_seed(19)
    own = torch.randn(3, 357, generator=private)
    task = torch.randn(3, 1044, generator=private)
    before_rng = torch.get_rng_state().clone()
    original_load = torch.load
    loads = []

    def audited_load(input_path, *args, **kwargs):
        if not isinstance(input_path, (str, Path)) or Path(input_path).resolve() != path.resolve():
            raise AssertionError("Inference attempted to load a different/external model file")
        loads.append(str(Path(input_path).resolve()))
        return original_load(input_path, *args, **kwargs)

    with patch.object(torch, "load", audited_load), patch.object(np, "load", side_effect=AssertionError("External statistics NPZ load forbidden")):
        inference = ReconstructionAMPInference.from_checkpoint(path)
        inference_before = {key: value.clone() for key, value in inference.task_normalizer.state_dict().items()}
        actual = inference.act(own, task)
    expected = loaded.act(own, task, deterministic=True)
    check(prefix + "protected_inference_sole_checkpoint_load_no_external_statistics", loads == [str(path.resolve())])
    check(prefix + "protected_inference_actual_synthetic_forward_raw_applied_bitwise",
          torch.equal(expected["actions"], actual["actions"])
          and torch.equal(expected["applied_actions"], actual["applied_actions"]))
    check(prefix + "protected_inference_caller_CPU_RNG_unchanged", torch.equal(before_rng, torch.get_rng_state()))
    check(prefix + "protected_inference_task_moments_frozen",
          all(torch.equal(value, inference_before[key]) for key, value in inference.task_normalizer.state_dict().items()))
    check(prefix + "inference_preserves_trained_D_action_columns_validates_discards_AMP_moments",
          torch.equal(inference.model.discriminator[0].weight, state["model"]["discriminator.0.weight"])
          and not hasattr(inference, "amp_normalizer") and not hasattr(inference, "value_normalizer"))
    check(prefix + "isolated_old_inference_full_network_truthfully_retained",
          set(inference.model._modules) == {"task_encoder", "actor", "critic", "discriminator"}
          and inference.model.log_std.shape == (69,) and not hasattr(inference, "optimizer")
          and not hasattr(inference, "policy_replay"))
    return {"policy": loaded.terrain_statistics_policy.describe(), "inference_model": "original full PaperAMPNetwork; not modern actor-only pruning",
            "inference_probe_inputs": "3 explicitly synthetic CPU self357/task1044 inputs from private seed19; not saved pre-action observations/GPU parity",
            "external_statistics_or_public_checkpoint_loaded_by_inference": False,
            "CPU_caller_RNG_preserved_by_inference": True, "CUDA_RNG_executed": False}


def validate_actual_result(result, phase, selected_updates, check):
    maximum = max(selected_updates)
    check("actual_closed_terminal_result_budget_and_profile", result["status"] == "passed"
          and result.get("error") is None and result["update_count"] == maximum
          and result["physical_transition_count"] == maximum * 8192
          and result["training_config"]["task_terrain_statistics_mode"] == "preserve_declared_suffix")
    args = result["arguments"]
    expected = {"num_envs": 256, "rollout_steps": 32, "replan_steps": 15, "episode_steps": 300,
                "terrain_extent_m": 192., "spawn_margin_m": 48., "trajectory_source": "mixed_amass_procedural",
                "curriculum_mode": "smooth", "curriculum_speed_max_mps": 2., "curriculum_accel_max_mps2": 1.5,
                "synchronized_replanning": True, "physics_only": True, "seed": 7, "from_scratch": False}
    check("actual_exact_registered_training_recipe", all(type(args[key]) is type(value) and args[key] == value for key, value in expected.items()))
    check("actual_segment_update_request_and_resume_phase", args["updates"] == (2000 if phase == "terminal2000" else 4)
          and (args["resume"] is None if phase != "pilot8" else Path(args["resume"]).name == "step_4.pt"))
    check("actual_group_exposure_terminal", result["group_transition_exposure"]["amass_actor_transitions"]
          == result["group_transition_exposure"]["procedural_actor_transitions"] == maximum * 4096)
    return {"result_status_execution_only": result["status"], "fall_resets_segment_or_fresh": result["fall_resets"],
            "time_limit_resets_segment_or_fresh": result["time_limit_resets"], "convergence_claim": False,
            "closed_result_args": args, "updates_cumulative": maximum, "transitions_cumulative": maximum * 8192}


STRATEGY = "public_pacer_state_prior_zero_action_extension"
PUBLIC_COUNTS = {"self_normalizer": 61931520001., "task_normalizer": 61931520001.,
                 "amp_normalizer": 15482880001., "value_normalizer": 6881280001.}
EXTENSION = {"schema": "navisaaclab.action_state_prior_initialization.v1",
             "history_columns": 1950, "action_columns": 69,
             "action_column_initialization": "zeros",
             "public_prior_training_data_overlap_with_current_amass": "unknown"}
PLAN_SHA = "2e21c1facc64186598ade1d72375f5a0a5dde07ad5f6fd05978ac019383b51c7"
ARCHIVE_SHA = "94ef331884ed9fa40fbc01e088903515bcb0b4ed64e9cf69d485ebb53443bee2"


def validate_state_prior(state, check, prefix=""):
    """Independent saved-payload check; public files are never opened here."""
    import torch
    config, initialization = state["config"], state["provenance"]["initialization"]
    check(prefix + "state_action2019_public_D_and_history_RMS_config",
          config.get("warm_start_action_discriminator_from_state_prior") is True
          and config["warm_start_discriminator"] is False
          and config["discriminator_input"] == "state_action"
          and config["amp_normalization"] == "per_history_coordinate"
          and config["history_frames"] == 10 and config["amp_frame_dim"] == 195
          and tuple(config["discriminator_layers"]) == (1024, 512)
          and config["normalize_observations"] is True and config["normalize_values"] is True
          and config["normalization_clip"] == 5. and config["warm_start_normalizer_count_cap"] == 200000.)
    expected = {"mode": "transfer_warm_start_public_pacer_actor_critic", "sha256": PUBLIC_SHA,
                "actor_critic_task_transferred": True, "observation_value_moments_transferred": True,
                "discriminator_initialization": STRATEGY, "action_conditioned_discriminator_initialization": STRATEGY,
                "discriminator_input": "state_action", "amp_normalization": "per_history_coordinate",
                "published_state_only_discriminator_transferred": True,
                "published_amp_history_moments_transferred": True, "author_navisaaclab_weights": False}
    check(prefix + "public_state_prior_initialization_provenance_exact",
          all(type(initialization.get(key)) is type(value) and initialization[key] == value
              for key, value in expected.items())
          and initialization.get("state_prior_action_extension") == EXTENSION
          and all(type(initialization["state_prior_action_extension"].get(key)) is type(value)
                  for key, value in EXTENSION.items())
          and initialization.get("normalizer_counts_before_cap") == PUBLIC_COUNTS
          and all(type(initialization["normalizer_counts_before_cap"].get(key)) is float for key in PUBLIC_COUNTS)
          and initialization.get("normalizer_count_cap") == 200000.
          and type(initialization.get("path")) is str
          and Path(initialization["path"]).is_absolute() and "://" not in initialization["path"])
    shapes = {"discriminator.0.weight": (1024, 2019), "discriminator.0.bias": (1024,),
              "discriminator.2.weight": (512, 1024), "discriminator.2.bias": (512,),
              "discriminator.4.weight": (1, 512), "discriminator.4.bias": (1,)}
    check(prefix + "state_prior_D_shapes_dtype_and_finite",
          all(state["model"][name].shape == shape and state["model"][name].dtype == torch.float32
              and torch.isfinite(state["model"][name]).all() for name, shape in shapes.items()))
    norm = state["normalizers"]["amp_normalizer"]
    check(prefix + "state_prior_history_RMS1950_nonnegative_finite",
          norm["mean"].shape == norm["variance"].shape == (1950,)
          and norm["mean"].dtype == norm["variance"].dtype == torch.float64
          and torch.isfinite(norm["mean"]).all() and torch.isfinite(norm["variance"]).all()
          and (norm["variance"] >= 0).all())
    check(prefix + "state_prior_AMP_count_capped_public_plus_actual_updates",
          type(state["update_count"]) is int and state["update_count"] > 0
          and float(norm["count"]) == 200000. + state["update_count"] * 12288)
    mask = torch.tensor(config["action_mask"])
    weights = state["model"]["discriminator.0.weight"][:, 1950:]
    active_nonzero = (weights[:, mask != 0] != 0).any(dim=0).sum().item()
    check(prefix + "trained57_action_columns_nonzero_frozen12_zero",
          mask.shape == (69,) and mask.count_nonzero() == 57 and state["update_count"] > 0
          and active_nonzero == 57 and not (weights[:, mask == 0] != 0).any())
    return {"explicit_flag": True, "strategy": STRATEGY, "public_D_transferred": True, "public_AMP_RMS_transferred": True,
            "published_state_only_discriminator_transferred": True,
            "published_amp_history_moments_transferred": True,
            "public_AMP_count": 15482880001., "current_AMP_count": float(norm["count"]), "count_cap": 200000.,
            "initial_action_columns_zero_is_provenance_only": True, "trained_active_columns_nonzero": active_nonzero,
            "frozen_action_columns_zero": 12, "public_AMP_count_before_cap": 15482880001.,
            "warm_AMP_count": 200000., "public_training_data_overlap": "unknown",
            "author_checkpoint_compatibility_claim": False}


def _verify_launch(bundle, phase, source_sha, ledger_path, record, check, *, closed):
    root = Path(__file__).resolve().parents[3]
    trial = {"pilot4": "pilot4", "pilot8": "resume4to8", "milestone": "fullfresh2000", "terminal2000": "fullfresh2000"}[phase]
    plan_path = root / "docs/style_improvement_20261003/state_prior_pacer_reward_training_preparation_v1/root_GPU_launch_commands_v2.json"
    path = root / "docs/style_improvement_20261003/state_prior_pacer_reward_actual_runs_v1" / (trial + "_actual_launch.json")
    plan, launch = json.loads(plan_path.read_text()), json.loads(path.read_text())
    check("actual_root_launch_plan_SHA_and_command_exact", record(plan_path)["sha256"] == PLAN_SHA
          and launch["schema"] == "state_prior.actual_launch.v1" and launch["name"] == trial
          and launch["launch_plan_sha256"] == PLAN_SHA
          and launch["command"] == plan["commands"][trial] and launch["cwd"] == plan["cwd"]
          and launch["env"] == plan["env"] and launch["no_parallel_GPU_at_launch"] is True
          and launch["command"][launch["command"].index("--import-report") + 1] == str(ledger_path)
          and record(bundle / "source_manifest.json")["sha256"] == source_sha
          and launch["source_manifest"] == record(bundle / "source_manifest.json")
          and plan["source_manifest"] == record(bundle / "source_manifest.json"))
    check("actual_root_process_closed_or_live_milestone_truthful",
          type(launch["child_pid"]) is int and launch["child_pid"] > 0 and bool(launch["start_utc"])
          and ((launch["status"] == "closed" and launch["exit_code"] == 0 and bool(launch["end_utc"])
               and launch["child_exit_observed"] is True and launch["driver_completed"] is True)
               if closed else launch["status"] in {"running", "closed"}))
    return launch, record(path)


def verify_launch_registration(bundle, phase, source_sha, ledger_path, result, record, check):
    launch, proof = _verify_launch(bundle, phase, source_sha, ledger_path, record, check, closed=True)
    driver = launch["command"][launch["command"].index("--") + 1:]
    check("actual_closed_result_matches_root_launch_profile_output_budget",
          driver[driver.index("--config") + 1] == result["arguments"]["config"]
          and driver[driver.index("--output") + 1] == result["arguments"]["output"]
          and int(driver[driver.index("--updates") + 1]) == result["arguments"]["updates"])
    return proof


def verify_milestone_launch_registration(bundle, source_sha, ledger_path, record, check):
    _, proof = _verify_launch(bundle, "milestone", source_sha, ledger_path, record, check, closed=False)
    return proof


def verify_source_archive(bundle, source, record, check):
    """Verify this independent archive and exact 934-to-935 source derivation."""
    import yaml
    archive_path = Path(__file__).resolve().parent / "actual935_source.tar.gz"
    archive = record(archive_path)
    check("actual935_full_source_archive_independent_SHA", archive["sha256"] == ARCHIVE_SHA)
    declared = {row["path"]: row for row in source["files"]}
    metadata = {"source_manifest.json", "launch_isolated.py", "commands.json", "watch_snapshots.py"}
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        names = {row.name for row in members}
        check("source_archive939_unique_regular_nontraversal_members", len(members) == len(names) == 939
              and len(source["files"]) == len(declared) == 935
              and all(row.isfile() and not Path(row.name).is_absolute() and ".." not in Path(row.name).parts for row in members)
              and names == set(declared) | metadata)
        for name, row in declared.items():
            data = tar.extractfile(name).read()
            check("archive_code_member_exact:" + name, len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"])
        for name in metadata:
            check("archive_prelaunch_metadata_exact:" + name, tar.extractfile(name).read() == (bundle / name).read_bytes())
    previous = source["previous_source_manifest"]
    check("previous934_source_manifest_immutable", record(previous["path"]) == previous
          and previous["sha256"] == "3db16db2ad3a4a8585b89557e7d88a9eb89d2e90461482f721c1ab1d52d44e2a")
    old = {row["path"]: row for row in json.loads(Path(previous["path"]).read_text())["files"]}
    expected = {"code/CrowdSim/humanoid/application/configuration.py", "code/scripts/train_paper_amp.py"}
    added = {"code/configs/diffusion_amp/isolated_terrain_state_prior_pacer_reward_absolute.yaml"}
    check("exact_old934_base_and_new935_source_sets", len(old) == 934 and set(old) <= set(declared)
          and set(declared) - set(old) == added)
    changed = {name for name in old if any(declared[name][key] != old[name][key] for key in ("sha256", "bytes"))}
    epoch = source["source_epoch"]
    check("exact_old932_preserved_two_resume_guard_delta_one_new_profile", changed == expected
          and len(set(old) - changed) == 932 and epoch["base_regular_files"] == 934
          and epoch["unchanged_base_files"] == 932 and set(epoch["modified_old_paths"]) == expected
          and set(epoch["new_paths"]) == added and epoch["guard_only_affects_training_resume_admission"] is True)
    contract_record = source["candidate_inputs"]
    check("source_bound_approved_resume_guard_contract", record(contract_record["path"]) == contract_record
          and contract_record["sha256"] == "ec0a930d6cae53c5ab9b769d63a448b1ef1a9201bf44ffeec263dff2aea2734b")
    contract = json.loads(Path(contract_record["path"]).read_text())
    check("approved_resume_guard_exact_two_targets", contract["status"] == "passed_exact_delta_review"
          and len(contract["targets"]) == 2 and {row["path"] for row in contract["targets"]} == expected
          and contract["fresh_training_math_changed"] is False and contract["fresh_RNG_changed"] is False)
    for row in contract["targets"]:
        name = row["path"]
        check("approved_guard_before_after_exact:" + name,
              all(row["isolated_before"][key] == old[name][key] for key in ("bytes", "sha256"))
              and all(row["isolated_after"][key] == declared[name][key] for key in ("bytes", "sha256"))
              and record(row["isolated_before"]["path"]) == row["isolated_before"]
              and record(row["isolated_after"]["path"]) == row["isolated_after"])
    profile_name = next(iter(added))
    profile_path = bundle / profile_name
    check("final_reward_profile_record_exact", source["profile"] == record(profile_path)
          and source["profile"]["sha256"] == "fa95b41900093005c4f1edb62a41e66f70c8377bfae86a70f5f0b876ceff4079")
    old_bundle = Path(previous["path"]).parent
    old_profile = yaml.safe_load((old_bundle / "code/configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml").read_text())
    new_profile = yaml.safe_load(profile_path.read_text())
    check("reward_profile_single_numeric_mechanism_and_metadata_only",
          old_profile["training"]["amp_reward_weight"] == old_profile["reward"]["amp_weight"] == 2.0
          and type(new_profile["training"]["amp_reward_weight"]) is float
          and type(new_profile["reward"]["amp_weight"]) is float
          and new_profile["training"]["amp_reward_weight"] == new_profile["reward"]["amp_weight"] == 0.5
          and new_profile["profile"] == "fixed_slots_amass_procedural_xy_state_action_terrain_public_state_prior_pacer_reward_factor"
          and len(old_profile["assumptions"]) == len(new_profile["assumptions"])
          and {i for i, (a, b) in enumerate(zip(old_profile["assumptions"], new_profile["assumptions"], strict=True)) if a != b} == {1, 7})
    old_profile["training"]["amp_reward_weight"] = old_profile["reward"]["amp_weight"] = 0.5
    old_profile["profile"] = new_profile["profile"]
    for index in (1, 7):
        old_profile["assumptions"][index] = new_profile["assumptions"][index]
    check("all_other_profile_training_data_trajectory_metadata_exact", old_profile == new_profile)
    return {"archive": archive, "regular_members": 939, "code_files": 935,
            "old932_byte_preserved": True, "modified_old_paths": sorted(changed), "new_paths": sorted(added),
            "approved_guard_contract": contract_record, "fresh_reward_weight_only_change": True}
