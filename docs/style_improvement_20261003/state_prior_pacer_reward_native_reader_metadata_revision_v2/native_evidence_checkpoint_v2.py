"""Source-only V12 native admission; pending preparation never opens checkpoints."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/"docs/style_improvement_20261003/state_prior_pacer_reward_native_reader_preparation_v1"
SOURCE=ROOT/"docs/style_improvement_20261003/native_runtime_epoch_v12/native_pedestrian_source_v12.json"
SOURCE_SHA="9ddbfe994bc692b344917796251780a08931d8d9846a1a19ae7476ba9753535c"
PROTOCOL=ROOT/"docs/style_improvement_20261003/state_prior_pacer_reward_native_preparation_v1"
TERMINAL=ROOT/"docs/style_improvement_20261003/state_prior_pacer_reward_training_audit_v1/terminal2000/audit.json"
RUNS=ROOT/"docs/style_improvement_20261003/state_prior_pacer_reward_native_actual_runs_v1"
BINDING=HERE/"protocol_source_binding.json"


def require(condition,message):
    if not condition:raise ValueError(message)


def record(path):
    path=Path(path).absolute()
    require(path.is_file() and not path.is_symlink(),"Regular local artifact required")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):digest.update(block)
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def protocol_module():
    binding=read(BINDING)
    for row in binding["protocol_sources"]:
        require(record(row["path"])==row and not Path(row["path"]).stat().st_mode&0o222,"Exact frozen native source/cases required")
    path=PROTOCOL/"validate_native_admission.py"
    spec=importlib.util.spec_from_file_location("pacer_reward_native_physical_protocol",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run_path(actors):
    require(type(actors) is int and actors in (4,15),"Exactly4/15 integer actors")
    suffix="" if actors==4 else "_15"
    return ROOT/f"output/diffusion_amp/native_warehouse_pacer_reward{suffix}_clearance08_200s_v1"


def expected_counts(actors):
    run_path(actors)
    return {"controls": 6000, "actors": actors, "duration_s": 200, "actor_controls": 6000 * actors,
            "shared_physics_substeps": 24000, "actor_physics_substeps": 24000 * actors, "feet_observations": 96000 * actors,
            "static_denominator_actor_controls": 6000 * actors, "peer_denominator_unique_pair_controls": 6000 * actors * (actors - 1) // 2,
            "live_floor_rays": actors + 128, "live_floor_reset_origins": actors, "live_free_samples": 128}


def source_binding():
    require(record(SOURCE)["sha256"] == SOURCE_SHA, "Exact production V12 manifest required")
    manifest = read(SOURCE)
    require(manifest["source_count"] == len(manifest["files"]) == 185, "Exact185 selected sources required")
    names = []
    for row in manifest["files"]:
        path = Path(row["path"])
        require(path.is_absolute() and path == path.resolve() and path.is_relative_to(ROOT), "Bounded canonical absolute185 path required")
        names.append(str(path.relative_to(ROOT)))
        require(record(path) == row, "Current production185 source differs")
    require(len(set(names)) == 185, "Duplicate selected source path")
    require(record(manifest["archive"]["path"]) == manifest["archive"], "Exact deterministic V12 archive required")
    with tarfile.open(manifest["archive"]["path"], "r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == 185 and {member.name for member in members} == set(names) and all(member.isfile() for member in members), "Exactly185 regular relative archive members")
        for row, name in zip(manifest["files"], names):
            payload = archive.extractfile(name).read()
            require(len(payload) == row["bytes"] and hashlib.sha256(payload).hexdigest() == row["sha256"], "Archived V12 member changed")
    return protocol_module().prepare_sources()["production_source"]



def prepare_evidence(actors):
    module=protocol_module()
    protocol=module.read_registration()
    sources=source_binding()
    future=protocol["future_checkpoint"]
    require(future["sha256"] is None and future["bytes"] is None,"No future checkpoint SHA can be fabricated")
    command=module.native_command(actors)
    require(command[command.index("--output")+1]==str(run_path(actors)),"Exact registered native output required")
    return {"production_source":sources,"native_protocol_registration":record(PROTOCOL/"registration.json"),
            "native_source_validator":record(PROTOCOL/"validate_native_admission.py"),
            "expected_native_command":command,"future_checkpoint":future,
            "terminal_admission_path":str(TERMINAL),"actual_terminal_artifacts_pending":True,
            "native_execution_audited":False,"checkpoint_read_during_preparation":False}


def validate_result_metadata(result,entry,training,checkpoint,actors):
    args=entry["parsed_driver_arguments"]
    require(entry["status"]=="SDK_free_preflight_passed" and entry["SDK_started"] is False
            and entry["Torch_imported_by_entry"] is False and entry["physical_execution_verified"] is False,
            "Actual SDK-free entry identity required")
    require(result["status"]=="passed" and result["actual_control_steps"]==6000 and result["arguments"]==args,
            "Completed result arguments differ from deployment entry")
    require(entry["control_steps"]==args["evaluate_steps"]==6000 and args["num_envs"]==actors
            and entry["physical_transitions_planned"]==6000*actors and entry["actual_duration_seconds"]==200
            and Path(args["output"]).resolve()==run_path(actors),"Complete registered actor/control protocol differs")
    require(all(args[name] is True for name in ("physics_only","deterministic_evaluation","pairwise_contacts","measure_ground_support","synchronized_replanning"))
            and args["diffusion_samples"]==1 and args["seed"]==7
            and args["contact_threshold_n"]==args["ground_support_threshold_n"]==1
            and args["power_source"]=="projected_joint_effort_substep_mean"
            and args["near_horizon_waypoint_mass"]==0 and args["guidance_weight"]==20
            and args["scene_collision_radius"]==.8,"Original measured native recipe differs")
    request=entry["evaluation_actor_request"]
    require(all(request["base_checkpoint"].get(key)==checkpoint[key] for key in ("path", "sha256")) and request["base_checkpoint"]["path"]==args["resume"]
            and request["alpha"]==0 and request["style_checkpoint"] is None and request["experimental_arm_style"] is False,
            "Exact trained checkpoint without donor/blend required")
    for name in ("configuration_file","preflight","native_map","probe_script_snapshot"):
        row=entry["native_scene_request"][name]
        require(record(row["path"])==row,"Actual native resource differs")
    require(record(args["config"])["sha256"]=="6d76dbfad6bc3827904af7efaf2fea7849ba5f5ff441971745ea17501b7f1bae",
            "Native requested PACER reward profile differs")
    actor=result["evaluation_actor"]
    require(actor["facade"]=="EvaluationActor" and actor["checkpoint_sha256"]==checkpoint["sha256"]
            and actor["checkpoint"]==checkpoint["path"] and actor["update_count"]==2000
            and actor["physical_transition_count"]==16384000
            and actor["full_training_provenance"]==training["provenance"]
            and actor["config"]==training["training_config"]
            and actor["config"]["warm_start_action_discriminator_from_state_prior"] is True,
            "Actor full training provenance/config/True state-prior differs")
    require(actor["AMP_reward_available"] is False and actor["critic_value_available"] is False
            and actor["saved_training_rng_restored"] is False and "experimental_arm_style" not in actor,
            "Actor-only deployment boundary differs")


def admit_closed_native(actors,registered):
    require(registered["schema"]=="pacer_reward.native200s_CPU_reader_registration.v1"
            and registered["actors"]==actors and registered["run"]==str(run_path(actors))
            and registered["native_execution_audited"] is False,"Exact pending reader registration required")
    expected=expected_counts(actors)
    require(registered["expected_counts"]==expected
            and all(type(registered["expected_counts"][key]) is type(value) for key,value in expected.items()),
            "Exact native denominators required")
    fresh=prepare_evidence(actors)
    require(registered["evidence"]==fresh,"Source/cases/pending metadata changed")
    module=protocol_module()
    path=Path(fresh["future_checkpoint"]["path"])
    checkpoint=record(path)
    require(not path.stat().st_mode&0o222 and path.stat().st_nlink==1,"Actual selected checkpoint must be independently readonly")
    terminal=read(TERMINAL)
    module.validate_terminal(terminal,checkpoint)
    terminal_record=record(TERMINAL)
    require(not TERMINAL.stat().st_mode&0o222,"Actual terminal report must be readonly")
    training_record=terminal["terminal_training_result"]
    require(record(training_record["path"])==training_record,"Actual terminal training result changed")
    training=read(training_record["path"])
    launch_path=Path(module.read_registration()["cases"][str(actors)]["actual_launch_record"])
    require(launch_path==RUNS/(f"native{actors}_actual_launch.json"),"Exact registered owner closure path required")
    launch=read(launch_path)
    module.validate_closed_receipt(launch,actors,checkpoint)
    for row in launch.get("inputs",[]):
        require(record(row["path"])==row,"Actual root native launch input changed")
    entry_path=run_path(actors)/"deployment_entry.json"
    result_path=run_path(actors)/"result.json"
    result,entry=read(result_path),read(entry_path)
    validate_result_metadata(result,entry,training,checkpoint,actors)
    require(launch["command"][launch["command"].index("--amp-checkpoint")+1]==entry["parsed_driver_arguments"]["resume"]
            and launch["command"][launch["command"].index("--output")+1]==entry["parsed_driver_arguments"]["output"],
            "Actual entry checkpoint/output differs from closed command")
    return {"schema":"pacer_reward.actual_native_source_admission.v1","status":"passed","actors":actors,
            "run":str(run_path(actors)),"result":record(result_path),"entry":record(entry_path),
            "closed_owner_launch":record(launch_path),"owner_log":record(launch["log"]),
            "production_source":fresh["production_source"],"terminal_admission":terminal_record,"checkpoint":checkpoint,
            "native_protocol_registration":fresh["native_protocol_registration"],"native_source_validator":fresh["native_source_validator"],
            "full_training_provenance_exact":True,"trained_action_weights_not_reinitialized_by_reader":True,
            "reader_deserialized_checkpoint":False,"complete_runtime_import_ledger":False,
            "source_boundary":"Actual production185 selected pre/post, root observed closed native launch/entry and new935 terminal reward0.5 admission. Complete native runtime import ledger is not claimed."}
