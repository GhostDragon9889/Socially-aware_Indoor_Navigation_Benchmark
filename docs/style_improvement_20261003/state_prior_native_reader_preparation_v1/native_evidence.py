"""Stdlib CPU admission for production V11 native4/15; no tensor/SDK load."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "docs/style_improvement_20261003/state_prior_native_actual_runs_v1"
SOURCE = ROOT / "docs/style_improvement_20261003/native_runtime_epoch_v11/native_pedestrian_source_v11.json"
SOURCE_SHA = "b4770f3f7301af7aed294db3145161453cf22325e66a9f9da0a3d677f5aeb3ee"
TERMINAL = ROOT / "docs/style_improvement_20261003/state_prior_training_audit_v1/terminal2000_admission_v1.json"
TERMINAL_SHA = "d5154251853253a8f866a103af9e51c6fff0e6ee7e1f56c36ff3f250b7124118"
READINESS = ROOT / "docs/style_improvement_20261003/state_prior_native_compatibility_v1/actual_CPU_readiness_v1.json"
READINESS_SHA = "1be710eb463aadb815faf8b5975b8b1c4d83d299ee5901366079aaa348fd51fe"
CKPT_SHA = "19f3341b9bb6118243c0945d839a669360175d93a35dde0f792639178d508f4f"
LAUNCHERS = {4: ("native4", "49d0294a694b9b2c09e03fa1116366d6c4bca7ab2e121658ac285200b1195838"),
             15: ("native15_v2", "3951112e5f8e10e22d9f9839dce8a2573075ecc98e811e42a71c5aeeef6481af")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def run_path(actors):
    require(type(actors) is int and actors in LAUNCHERS, "Exactly4/15 integer actors")
    suffix = "" if actors == 4 else "_15"
    return ROOT / f"output/diffusion_amp/native_warehouse_state_prior{suffix}_clearance08_200s_v1"


def expected_counts(actors):
    run_path(actors)
    return {"controls": 6000, "actors": actors, "duration_s": 200, "actor_controls": 6000 * actors,
            "shared_physics_substeps": 24000, "actor_physics_substeps": 24000 * actors, "feet_observations": 96000 * actors,
            "static_denominator_actor_controls": 6000 * actors, "peer_denominator_unique_pair_controls": 6000 * actors * (actors - 1) // 2,
            "live_floor_rays": actors + 128, "live_floor_reset_origins": actors, "live_free_samples": 128}


def source_binding():
    require(record(SOURCE)["sha256"] == SOURCE_SHA, "Exact production V11 manifest required")
    manifest = read(SOURCE)
    require(manifest["source_count"] == len(manifest["files"]) == 183, "Exact183 selected sources required")
    names = []
    for row in manifest["files"]:
        path = Path(row["path"])
        require(path.is_absolute() and path == path.resolve() and path.is_relative_to(ROOT), "Bounded canonical absolute183 path required")
        names.append(str(path.relative_to(ROOT)))
        require(record(path) == row, "Current production183 source differs")
    require(len(set(names)) == 183, "Duplicate selected source path")
    require(record(manifest["archive"]["path"]) == manifest["archive"], "Exact deterministic V11 archive required")
    with tarfile.open(manifest["archive"]["path"], "r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == 183 and {member.name for member in members} == set(names) and all(member.isfile() for member in members), "Exactly183 regular relative archive members")
        for row, name in zip(manifest["files"], names):
            payload = archive.extractfile(name).read()
            require(len(payload) == row["bytes"] and hashlib.sha256(payload).hexdigest() == row["sha256"], "Archived V11 member changed")
    return {"manifest": record(SOURCE), "archive": manifest["archive"], "source_count": 183,
            "all_archive_and_current_selected_bytes_exact": True, "complete_runtime_import_ledger": False}


def terminal_binding():
    require(record(TERMINAL)["sha256"] == TERMINAL_SHA, "Exact actual full terminal admission required")
    terminal = read(TERMINAL)
    require(terminal["status"] == "passed" and terminal["terminal_state_admitted"] is True and terminal["complete_state_restore_validated"] is True and terminal["public_D_and_history_RMS_transfer_validated"] is True and terminal["trained_action_columns_preserved_on_restore"] is True and terminal["milestone_only_admission"] is False and terminal["actual_updates"] == 2000 and terminal["actual_physical_transitions"] == 16384000 and terminal["RMS_counts"]["amp_normalizer"] == 24776000, "Actual trained True prior terminal contract differs")
    for key in ("audit", "checkpoint", "actual_terminal_training_result"):
        require(record(terminal[key]["path"]) == terminal[key], "Actual bound terminal artifact changed")
    require(terminal["checkpoint"]["sha256"] == CKPT_SHA and Path(terminal["checkpoint"]["path"]).stat().st_mode & 0o222 == 0, "Exact readonly selected terminal checkpoint required")
    training = read(terminal["actual_terminal_training_result"]["path"])
    contract = ROOT / "CrowdSim/humanoid/application/state_prior_initialization.py"
    spec = importlib.util.spec_from_file_location("native_state_prior_contract", contract)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(training["training_config"]["warm_start_action_discriminator_from_state_prior"] is True, "Explicit True prior required")
    module.validate_action_state_prior_initialization(SimpleNamespace(**training["training_config"]), training["provenance"]["initialization"])
    return terminal, training


def validate_entry(entry, actors, checkpoint):
    args = entry["parsed_driver_arguments"]
    require(entry["status"] == "SDK_free_preflight_passed" and entry["SDK_started"] is False and entry["Torch_imported_by_entry"] is False and entry["physical_execution_verified"] is False, "Actual SDK-free preflight template required")
    require(entry["control_steps"] == args["evaluate_steps"] == 6000 and args["num_envs"] == actors and entry["physical_transitions_planned"] == 6000 * actors and entry["actual_duration_seconds"] == 200 and Path(args["output"]).resolve() == run_path(actors), "Exact complete native actor/control protocol required")
    require(all(args[name] is True for name in ("physics_only", "deterministic_evaluation", "pairwise_contacts", "measure_ground_support", "synchronized_replanning")) and args["diffusion_samples"] == 1 and args["seed"] == 7 and args["contact_threshold_n"] == args["ground_support_threshold_n"] == 1 and args["power_source"] == "projected_joint_effort_substep_mean" and args["near_horizon_waypoint_mass"] == 0 and args["guidance_weight"] == 20 and args["scene_collision_radius"] == .8, "Original measured native deployment recipe differs")
    request = entry["evaluation_actor_request"]
    require(request["base_checkpoint"]["sha256"] == checkpoint["sha256"] == CKPT_SHA and request["base_checkpoint"]["path"] == args["resume"] == checkpoint["path"] and request["alpha"] == 0 and request["style_checkpoint"] is None and request["experimental_arm_style"] is False, "Exact trained model without donor/blend required")
    for name in ("configuration_file", "preflight", "native_map", "probe_script_snapshot"):
        row = entry["native_scene_request"][name]
        require(record(row["path"]) == row, "Registered native resource changed")
    return args


def prepare_evidence(actors):
    binding = source_binding()
    terminal, training = terminal_binding()
    require(record(READINESS)["sha256"] == READINESS_SHA, "Exact actual production CPU compatibility proof required")
    readiness = read(READINESS)
    require(readiness["status"] == "passed" and readiness["selected_source_count"] == 183 and readiness["actual_native_physics_validated"] is False, "Cold compatibility cannot claim physical effects")
    entry = next(row for row in readiness["preflights"] if row["parsed_driver_arguments"]["num_envs"] == actors)
    validate_entry(entry, actors, terminal["checkpoint"])
    prefix, launcher_sha = LAUNCHERS[actors]
    preflight_path = RUNS / (prefix + "_actual_preflight.json")
    preflight = read(preflight_path)
    require(preflight["status"] == "closed" and preflight["exit_code"] == 0 and preflight["prepare_only"] is True and preflight["actors"] == actors and preflight["launcher"]["sha256"] == launcher_sha and record(preflight["launcher"]["path"]) == preflight["launcher"] and preflight["selected_source_before"] == preflight["selected_source_after"] and preflight["selected_source_before"]["manifest"] == binding["manifest"], "Actual closed owner SDK-free preflight required")
    return {"production_source": binding, "terminal_admission": record(TERMINAL), "checkpoint": terminal["checkpoint"], "actual_CPU_readiness": record(READINESS), "owner_closed_preflight": record(preflight_path), "preflight_command": preflight["command"], "declared_env": preflight["declared_env"], "expected_entry": entry, "expected_full_training_provenance": training["provenance"], "expected_training_config": training["training_config"], "owner_launcher": preflight["launcher"]}


def validate_result_metadata(result, registered):
    evidence = registered["evidence"]
    args = evidence["expected_entry"]["parsed_driver_arguments"]
    actor = result["evaluation_actor"]
    require(result["status"] == "passed" and result["actual_control_steps"] == 6000 and result["arguments"] == args, "Terminal completed argv/control contract differs")
    require(actor["facade"] == "EvaluationActor" and actor["checkpoint_sha256"] == CKPT_SHA and actor["checkpoint"] == args["resume"] and actor["update_count"] == 2000 and actor["physical_transition_count"] == 16384000 and actor["full_training_provenance"] == evidence["expected_full_training_provenance"] and actor["config"] == evidence["expected_training_config"] and actor["config"]["warm_start_action_discriminator_from_state_prior"] is True, "Actual actor model/config/full True provenance differs")
    require(actor["AMP_reward_available"] is False and actor["critic_value_available"] is False and actor["saved_training_rng_restored"] is False and "experimental_arm_style" not in actor, "Actor-only frozen deployment boundary differs")


def admit_closed_native(actors, registered):
    require(registered["schema"] == "public_state_prior.native200s_CPU_reader_registration.v1" and registered["actors"] == actors and registered["run"] == str(run_path(actors)) and registered["native_execution_audited"] is False, "Exact pending native registration required")
    counts = expected_counts(actors)
    require(type(registered["expected_counts"]) is dict and registered["expected_counts"] == counts and all(type(registered["expected_counts"][key]) is type(value) for key, value in counts.items()), "Exact canonical complete native denominators required")
    fresh = prepare_evidence(actors)
    require(fresh == registered["evidence"], "Registered native evidence differs from actual terminal/readiness/preflight")
    binding = fresh["production_source"]
    prefix, _ = LAUNCHERS[actors]
    launch_path = RUNS / (prefix + "_actual_launch.json")
    launch = read(launch_path)
    evidence = registered["evidence"]
    require(launch["status"] == "closed" and launch["exit_code"] == 0 and launch["prepare_only"] is False and launch["actors"] == actors and launch["cwd"] == str(ROOT) and launch["launcher"] == evidence["owner_launcher"] and record(launch["launcher"]["path"]) == launch["launcher"] and launch["command"] == evidence["preflight_command"][:-1] and evidence["preflight_command"][-1] == "--preflight-only" and launch["declared_env"] == evidence["declared_env"] and launch["complete_import_ledger_claimed"] is False, "Actual closed native launch command/source contract differs")
    require(actors == 4 or launch.get("child_exit_observed") is True, "Actual15 v2 child exit must be observed")
    for when in ("selected_source_before", "selected_source_after"):
        require(launch[when]["manifest"] == binding["manifest"] and launch[when]["archive"] == binding["archive"] and launch[when]["all183_exact"] is True and launch[when]["complete_runtime_import_ledger"] is False, "Owner selected183 pre/post proof differs")
    for row in launch["inputs"]:
        require(record(row["path"]) == row, "Actual owner native launch input changed")
    entry_path = run_path(actors) / "deployment_entry.json"
    require(read(entry_path) == evidence["expected_entry"], "Actual deployment entry differs from closed preflight template")
    result_path = run_path(actors) / "result.json"
    result = read(result_path)
    validate_result_metadata(result, registered)
    return {"schema": "public_state_prior.actual_native_source_admission.v1", "status": "passed", "actors": actors, "run": str(run_path(actors)), "result": record(result_path), "entry": record(entry_path), "closed_owner_launch": record(launch_path), "owner_log": record(launch["log"]), "production_source": binding, "terminal_admission": record(TERMINAL), "checkpoint": evidence["checkpoint"], "full_training_provenance_exact": True, "trained_action_weights_not_reinitialized_by_reader": True, "reader_deserialized_checkpoint": False, "complete_runtime_import_ledger": False, "source_boundary": "Exact production selected183 pre/post snapshots and owner closed launch. Training934 and GPU complete import ledger are not claimed."}
