"""Fixed default62c4/newDPb563 identities and actual process admission; no model imports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "docs/style_improvement_20261003/default_newDP_physical_actual_runs_v1"
SOURCE = ROOT / "docs/style_improvement_20261003/native_runtime_epoch_v12/native_pedestrian_source_v12.json"
PYTHON = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
GPU_ENV = {**CPU_ENV, "CUDA_VISIBLE_DEVICES": "0"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular source/artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_epoch(manifest, archive, count, *, current):
    require(record(manifest["path"]) == manifest and record(archive["path"]) == archive, "Source epoch identity changed")
    epoch = read(manifest["path"])
    require(epoch["source_count"] == len(epoch["files"]) == count, "Source epoch count differs")
    with tarfile.open(archive["path"], "r:gz") as stream:
        members = {item.name: item for item in stream.getmembers()}
        require(len(members) == count and all(item.isfile() for item in members.values()), "Exact regular archive members required")
        for row in epoch["files"]:
            path = Path(row["path"])
            absolute = path if path.is_absolute() else ROOT / path
            name = str(absolute.relative_to(ROOT))
            data = stream.extractfile(members[name]).read()
            require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "Archived source changed: " + name)
            if current:
                require(record(absolute) == {"path": str(absolute), "bytes": row["bytes"], "sha256": row["sha256"]}, "Current production source changed: " + name)
    return {"manifest": manifest, "archive": archive, "source_count": count, "all_archived_source_bytes_verified": True, "current_source_bytes_verified": current, "complete_runtime_import_ledger": False}


def registration():
    freeze = read(HERE / "source_freeze_v1.json")
    require(freeze["status"] == "frozen_prepared_only", "Frozen registration required")
    for item in freeze["files"]:
        require(record(item["path"]) == item and not Path(item["path"]).stat().st_mode & 0o222, "Prepared source changed")
    return read(HERE / "registration.json")


def verify_resource_records(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys() and isinstance(value["path"], str):
            path = Path(value["path"])
            if path.is_absolute():
                require(record(path) == {key: value[key] for key in ("path", "bytes", "sha256")}, "Resource changed: " + str(path))
        for item in value.values():
            verify_resource_records(item)
    elif isinstance(value, list):
        for item in value:
            verify_resource_records(item)


def source_binding(plan):
    for item in plan["inputs"].values():
        require(record(item["path"]) == item, "Registered input changed: " + item["path"])
    controller, planner = plan["controller"], plan["planners"]["new"]
    require(not Path(planner["path"]).stat().st_mode & 0o222, "New DP sealed copy must remain readonly")
    require(Path(planner["path"]).stat().st_nlink == 1, "New DP uses independent sealed copy")
    source = verify_epoch(plan["production_source"]["manifest"], plan["production_source"]["archive"], 185, current=True)
    require(source == plan["production_source"], "Production185 source binding differs")
    owner, seal = read(plan["inputs"]["DP_closed_training"]["path"]), read(plan["inputs"]["DP_terminal_seal"]["path"])
    require(owner["schema"] == "local_orca_maps.actual_diffusion_training_launch.v1" and owner["status"] == "closed"
            and type(owner["exit_code"]) is int and owner["exit_code"] == 0 and owner["child_exit_observed"] is True
            and owner["driver_completed"] is True and owner["selected_source_before"] == owner["selected_source_after"]
            and owner["selected_source_after"]["regular_files"] == 935
            and owner["selected_source_after"]["manifest"] == plan["inputs"]["DP_training_source_manifest"]
            and seal["actual_root_session_exit0"] is True and seal["completed_optimizer_updates"] == 5000
            and seal["all5000_metric_rows_retained"] is True
            and {key: seal["exclusive_terminal_copy"][key] for key in ("path", "bytes", "sha256")} == planner,
            "Actual completed DP935 training/sealed5000 binding required")
    for baseline in plan["historical_native"].values():
        verify_epoch(baseline["source_epoch"]["manifest"], baseline["source_epoch"]["archive"], baseline["source_epoch"]["source_count"], current=False)
    for case in plan["cases"].values():
        verify_resource_records(case["preflight"]["native_scene_request"])
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "Source admission must not import a model/SDK")
    return {"production_source": source, "controller": controller, "planners": plan["planners"],
            "DP_training_source_manifest": plan["inputs"]["DP_training_source_manifest"],
            "DP_closed_training": plan["inputs"]["DP_closed_training"], "DP_terminal_seal": plan["inputs"]["DP_terminal_seal"],
            "legacy_controller_training_source_is_new935": False, "new09e_terminal_used_for62c4": False}


def admit_capture(plan, phase):
    binding = source_binding(plan)
    case = plan["cases"][phase]
    owner_path = Path(case["GPU_receipt"])
    owner = read(owner_path)
    require(owner["schema"] == "default_newDP.actual_physical_launch.v1" and owner["phase"] == phase
            and owner["status"] == "closed" and type(owner["exit_code"]) is int and owner["exit_code"] == 0
            and owner["child_exit_observed"] is True and owner["driver_completed"] is True
            and owner["command"] == case["command"] and owner["owner"] == record(HERE / "launch_registered.py")
            and owner["binding_before"] == owner["binding_after"] == binding,
            "Actual registered closed root physical process required")
    run = Path(case["output"])
    for item in owner["artifacts"]:
        require(record(item["path"]) == item and not Path(item["path"]).stat().st_mode & 0o222, "Sealed physical artifact changed")
    required = {str(run / name) for name in ("result.json", "deployment_entry.json", "physical_states.npz", "ground_support.npz", "reference_plans.npz")}
    require(required <= {item["path"] for item in owner["artifacts"]}, "All actual physical captures must be sealed")
    result, entry = read(run / "result.json"), read(run / "deployment_entry.json")
    expected = case["preflight"]
    require(entry == expected and result["arguments"] == expected["parsed_driver_arguments"]
            and result["status"] == "passed" and type(result["actual_control_steps"]) is int
            and result["actual_control_steps"] == 6000, "Actual preflight/recipe/complete control count differs")
    require(result["provenance"]["diffusion_checkpoint"] == case["planner"]["path"]
            and result["provenance"]["diffusion_sha256"] == case["planner"]["sha256"],
            "Actual loaded Diffusion path/SHA differs")
    request = entry["evaluation_actor_request"]
    require(request["base_checkpoint"] == {"path": plan["controller"]["path"], "sha256": plan["controller"]["sha256"]}
            and request["style_checkpoint"] is None and request["alpha"] == 0.0 and request["experimental_arm_style"] is False,
            "Default checkpoint producer path/SHA ABI differs")
    actor = result["evaluation_actor"]
    historical = read(plan["historical_native"]["4"]["result"]["path"])["evaluation_actor"]
    require(actor["facade"] == "EvaluationActor" and actor["checkpoint"] == plan["controller"]["path"]
            and actor["checkpoint_sha256"] == plan["controller"]["sha256"] and actor["checkpoint_bytes"] == plan["controller"]["bytes"]
            and actor["update_count"] == 1000 and actor["physical_transition_count"] == 8192000
            and actor["full_training_provenance"] == historical["full_training_provenance"]
            and actor["AMP_reward_available"] is False and actor["critic_value_available"] is False
            and actor["saved_training_rng_restored"] is False and "experimental_arm_style" not in actor,
            "Actual baseline actor-only/full training provenance identity differs")
    rng = result["evaluation_rng_preparation"]
    require(rng["schema"] == "legacy_evaluation_initialization_rng.v1" and rng["seed"] == 7
            and rng["saved_checkpoint_rng_restored"] is False and rng["trainer_created"] is False
            and rng["optimizer_created"] is False and rng["replay_created"] is False,
            "Original seed7 actor-only initialization RNG boundary differs")
    return {"schema": "default_newDP.actual_capture_source_admission.v1", "status": "passed", **binding,
            "root_closed_launch": record(owner_path), "result": record(run / "result.json"), "entry": record(run / "deployment_entry.json"),
            "full_training_provenance_exact": True, "checkpoint_deserialized_by_reader": False,
            "complete_runtime_import_ledger": False}


def zero_physical(report, phase):
    require(report["schema"] == "default_newDP.independent_physical_CPU_audit.v1" and report["status"] == "passed"
            and report["phase"] == phase and report["actual_capture_independently_audited"] is True,
            "Actual complete independent raw audit required")
    counts = report["physical_gate_counts"]
    require(type(counts["controls"]) is int and counts["controls"] == 6000 and counts["duration_s"] == 200.0
            and counts["actors"] == 4
            and all(type(counts[key]) is int and counts[key] == 0 for key in ("falls", "static_actor_controls", "peer_pair_controls")),
            "Full6000 zero falls/static/peer physical requirement failed")


def admit_new_fixture(plan):
    """For root's later native4 stage; no future native files are opened here."""
    case = plan["cases"]["fixture_new"]
    report = read(case["audit"])
    zero_physical(report, "fixture_new")
    require(report["actual_source_admission"] == admit_capture(plan, "fixture_new"), "NEW fixture source closure differs")
    owner = read(case["CPU_receipt"])
    require(owner["schema"] == "default_newDP.actual_CPU_audit_launch.v1" and owner["phase"] == "fixture_new"
            and owner["status"] == "closed" and type(owner["exit_code"]) is int and owner["exit_code"] == 0
            and owner["child_exit_observed"] is True and owner["driver_completed"] is True
            and owner["command"] == case["CPU_command"] and owner["owner"] == record(HERE / "launch_registered.py")
            and owner["binding_before"] == owner["binding_after"] == source_binding(plan)
            and owner["audit_artifact"] == record(case["audit"]), "Actual NEW fixture CPU closure/report binding required")
    return {"phase": "fixture_new", "audit": record(case["audit"]), "CPU_closed_launch": record(case["CPU_receipt"]),
            "gate": "full6000/200s zero falls/static actor-controls/peer unordered-pair controls", "style_score_used": False}
