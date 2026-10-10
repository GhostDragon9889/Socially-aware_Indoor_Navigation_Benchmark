"""Baseline Office3 short-case identities/closure; never deserialize a model."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREPARED = ROOT / "docs/style_improvement_20261003/local_Office_baseline_probe_preparation_v1"
PLAN_SHA = "fa49f2e9ec8b82cf160fb6a57b957f98f7d14d95b70f84c5bb1e6b95c7811488"
COMMANDS_SHA = "df47d5c61b1e14a936df9a2e379e410fc90176784240a5e07b9de031572d298d"
FREEZE_SHA = "998f4a9406b2a4223e3146e115835ed0e8bd1b82161d9b89d580507d62b5edcb"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular bounded source/artifact required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_evidence():
    for name, digest in (("plan.json", PLAN_SHA), ("commands.json", COMMANDS_SHA), ("source_freeze_v1.json", FREEZE_SHA)):
        require(record(PREPARED / name)["sha256"] == digest, "Exact frozen baseline source plan required")
    plan = read(PREPARED / "plan.json")
    commands = read(PREPARED / "commands.json")
    freeze = read(PREPARED / "source_freeze_v1.json")
    for row in freeze["source_files"]:
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222,
                "Frozen baseline Office source changed")
    driver = load(commands["adapter"]["path"], "Office_raw_registered_baseline_adapter")
    current, _ = driver.prepare_sources()
    require(current == plan, "Current baseline Office identity differs")
    from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
    request = plan["existing_native_scene_request"]
    provider = DynBenchSceneProvider.from_preflight(request["preflight"]["path"],
        expected_sha256=request["preflight"]["sha256"], probe_script_snapshot=request["probe_script_snapshot"]["path"])
    native_map = read(request["native_map"]["path"])
    require(provider.scene_name == "Office" and len(provider.static_collision_paths) == 3642
            and native_map["authored_physics_scenes_disabled_in_session"] == [],
            "Registered Office static/source PhysicsScene identity differs")
    return plan, commands, {
        "baseline_plan": record(PREPARED / "plan.json"), "baseline_commands": record(PREPARED / "commands.json"),
        "baseline_freeze": record(PREPARED / "source_freeze_v1.json"), "production_source": plan["production_source"],
        "selected_checkpoint": plan["selected_checkpoint"], "profile": plan["profile"], "observer": plan["observer"],
        "static_collider_count": 3642, "source_authored_physics_scenes": [],
        "Office_static_paths_SHA256": hashlib.sha256(json.dumps(sorted(provider.static_collision_paths), separators=(",", ":")).encode()).hexdigest(),
        "source_USD": provider.describe()["source_USD"], "original_math_files": read(HERE / "reader_input_identity.json"),
        "Office_actual_capture_read": False, "Office_actual_gravity_or_stability_observed": False,
        "checkpoint_deserialized_by_reader": False, "GPU_used": False, "SDK_started": False,
    }


def run_path():
    return Path(read(PREPARED / "plan.json")["actual_Office_capture"]["path"])


def admit_actual(registered):
    plan, commands, current = source_evidence()
    require(registered["source_evidence"] == current, "Registered short-case source identity changed")
    launch_path = Path(commands["actual_launch_record"])
    launch = read(launch_path)
    require(launch.get("schema") == commands["actual_launch_schema"] and launch.get("status") == "closed"
            and type(launch.get("exit_code")) is int and launch["exit_code"] == 0
            and launch.get("child_exit_observed") is True and launch.get("driver_completed") is True
            and launch.get("prepare_only") is False and launch.get("command") == commands["future_root_owned_command"]
            and launch.get("selected_checkpoint") == launch.get("selected_checkpoint_after") == plan["selected_checkpoint"]
            and launch.get("selected_source_before") == launch.get("selected_source_after") == plan["production_source"]
            and launch.get("adapter") == commands["adapter"] and launch.get("observer") == plan["observer"],
            "Actual source-bound root Office process closure required")
    result_path, entry_path = run_path() / "result.json", run_path() / "deployment_entry.json"
    result, entry = read(result_path), read(entry_path)
    require(entry["driver_argv"] == plan["driver_argv"] and result["arguments"] == entry["parsed_driver_arguments"]
            and result["status"] == "passed" and type(result["actual_control_steps"]) is int
            and result["actual_control_steps"] == entry["control_steps"] == 300,
            "Actual producer requested case/control/arguments differ")
    arguments = result["arguments"]
    require(arguments["num_envs"] == 3 and arguments["evaluate_steps"] == 300
            and arguments["seed"] == 7 and arguments["guidance_weight"] == 20
            and arguments["near_horizon_waypoint_mass"] == 0 and arguments["scene_collision_radius"] == .8
            and all(arguments[key] is True for key in ("physics_only", "pairwise_contacts", "measure_ground_support", "deterministic_evaluation", "synchronized_replanning")),
            "Original Office short physical recipe differs")
    request = entry["evaluation_actor_request"]
    checkpoint = plan["selected_checkpoint"]
    require(request["base_checkpoint"] == {"path": checkpoint["path"], "sha256": checkpoint["sha256"]}
            and request["style_checkpoint"] is None and request["alpha"] == 0.0
            and request["experimental_arm_style"] is False, "Normal baseline loader/source/donor ABI differs")
    actor = result["evaluation_actor"]
    require(actor["facade"] == "EvaluationActor" and actor["checkpoint"] == checkpoint["path"]
            and actor["checkpoint_sha256"] == checkpoint["sha256"] and actor["checkpoint_bytes"] == checkpoint["bytes"]
            and actor["update_count"] == 1000 and actor["physical_transition_count"] == 8192000,
            "Actual strict baseline actor identity differs")
    historical = read(plan["historical_baseline_native_stability"]["4"]["result"]["path"])["evaluation_actor"]
    require(actor["full_training_provenance"] == historical["full_training_provenance"],
            "Loaded baseline full training provenance differs")
    require(actor["AMP_reward_available"] is False and actor["critic_value_available"] is False
            and actor["saved_training_rng_restored"] is False and "experimental_arm_style" not in actor,
            "Actor-only runtime boundary differs")
    require(entry["native_scene_request"] == plan["existing_native_scene_request"], "Office resource/map/config changed")
    return {
        "schema": "local_Office.baseline62c4_actual_source_admission.v1", "status": "passed",
        "root_closed_launch": record(launch_path), "result": record(result_path), "entry": record(entry_path),
        "selected_checkpoint": checkpoint, "production_source": plan["production_source"],
        "adapter": commands["adapter"], "observer": plan["observer"],
        "full_training_provenance_exact": True, "checkpoint_deserialized_by_reader": False,
        "new09e_terminal_used_for62c4": False, "complete_runtime_import_ledger": False,
    }
