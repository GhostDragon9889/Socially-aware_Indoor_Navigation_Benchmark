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
    snapshot = Path(__file__).resolve().parents[1] / "staged_terrain_rms_candidate_v1/public_task_statistics_snapshot.npz"
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


def verify_launch_registration(bundle, phase, source_sha, ledger_path, result, record, check):
    trial = {"pilot4": "pilot4", "pilot8": "resume4to8", "terminal2000": "fullfresh2000"}[phase]
    path = bundle / (trial + "_launch_registration.json")
    registration = json.loads(path.read_text())
    argv = registration["argv"]
    check("actual_prelaunch_source_and_phase_registration", registration["trial"] == trial
          and registration["source_manifest_sha256"] == source_sha
          and argv[argv.index("--import-report") + 1] == str(ledger_path)
          and registration["cwd"] == str(bundle / "code"))
    driver = argv[argv.index("--") + 1:]
    check("actual_prelaunch_profile_output_and_update_request",
          driver[driver.index("--config") + 1] == result["arguments"]["config"]
          and driver[driver.index("--output") + 1] == result["arguments"]["output"]
          and int(driver[driver.index("--updates") + 1]) == result["arguments"]["updates"])
    return record(path)


def verify_source_archive(bundle, source, record, check):
    workspace = Path(__file__).resolve().parents[3]
    ready_path = workspace / "docs/style_improvement_20261002/staged_terrain_rms_candidate_v1/proposal_v2/isolated_CPU_readiness_v1.json"
    ready = json.loads(ready_path.read_text())
    archive = ready["source_archive"]
    check("actual931_full_source_archive_independent_SHA_size", record(archive["path"]) == archive
          and archive["sha256"] == "1982602b67cf21fac0cf3f1d0ac045a195e5af9f5d732b2a70c2277ede895fe7")
    declared = {row["path"]: row for row in source["files"]}
    with tarfile.open(archive["path"], "r:gz") as tar:
        members = tar.getmembers()
        names = {row.name for row in members}
        check("source_archive_940_regular_unique_nontraversal_members", len(members) == len(names) == 940
              and all(row.isfile() and not Path(row.name).is_absolute() and ".." not in Path(row.name).parts for row in members))
        check("source_archive_all931_exact_code_members", {name for name in names if name.startswith("code/")} == set(declared))
        for name, row in declared.items():
            data = tar.extractfile(name).read()
            check("archive_code_member_exact:" + name,
                  len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"])
        for name in ("source_manifest.json", "launch_isolated.py", "commands_v2.json"):
            check("archive_prelaunch_metadata_exact:" + name, tar.extractfile(name).read() == (bundle / name).read_bytes())
    old_record = source["previous_source_manifest"]
    check("previous926_source_manifest_immutable", record(old_record["path"]) == old_record)
    old = {row["path"]: row for row in json.loads(Path(old_record["path"]).read_text())["files"]}
    changed = {name for name in old if declared[name]["sha256"] != old[name]["sha256"]}
    expected = {"code/" + name for name in (
        "CrowdSim/humanoid/application/settings.py", "CrowdSim/humanoid/application/resources.py",
        "CrowdSim/humanoid/training/trainer.py", "CrowdSim/humanoid/runtime/checkpoint_contract.py",
        "CrowdSim/humanoid/runtime/amp_inference.py")}
    check("exact_old921_preserved_and_five_RMS_delta_paths", len(old) == 926 and len(declared) == 931
          and changed == expected and len(set(old) - changed) == 921
          and set(declared) - set(old) == set(source["source_epoch"]["new_paths"]))
    return {"archive": archive, "regular_members": 940, "code_files": 931,
            "old921_byte_preserved": True, "modified_old_paths": sorted(changed), "readiness_record": record(ready_path)}


def verify_milestone_launch_registration(bundle, source_sha, ledger_path, record, check):
    """Check actual prelaunch metadata; never manufacture a terminal result."""
    path = bundle / "fullfresh2000_launch_registration.json"
    registration = json.loads(path.read_text())
    argv = registration["argv"]
    driver = argv[argv.index("--") + 1:]
    check("actual_milestone_fresh2000_prelaunch_source_and_ledger", registration["trial"] == "fullfresh2000"
          and registration["source_manifest_sha256"] == source_sha
          and argv[argv.index("--import-report") + 1] == str(ledger_path)
          and registration["cwd"] == str(bundle / "code"))
    check("actual_milestone_prelaunch_budget_no_resume", driver[driver.index("--updates") + 1] == "2000"
          and "--resume" not in driver and "--from-scratch" not in driver
          and driver[driver.index("--num-envs") + 1] == "256"
          and driver[driver.index("--rollout-steps") + 1] == "32")
    return record(path)
