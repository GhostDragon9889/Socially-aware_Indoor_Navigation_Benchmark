"""Future root-tool and complete capture admission; standard-library-only imports."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def record(path, *, sealed=False):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular independent file required: " + str(path))
    info = path.stat()
    if sealed:
        require(not info.st_mode & 0o222 and info.st_nlink == 1, "Readonly nlink1 artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": info.st_size, "sha256": digest.hexdigest()}


def verify(row, *, sealed=False):
    require(record(row["path"], sealed=sealed) == row, "Registered bytes changed: " + row["path"])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_binding(reg, domain):
    """Verify bytes, without imposing readonly on original production or weights."""
    verify(reg["commands"], sealed=True)
    plan = read(reg["commands"]["path"])
    require(plan["source_files"] == reg["selection_sources"] and len(reg["selection_sources"]) == 4,
            "Exact original four selection sources required")
    for row in reg["selection_sources"]:
        verify(row)
    for row in reg["preflights"].values():
        verify(row, sealed=True)
    verify(reg["support_sources"]["physical_evidence"])
    base = load("common50_original_physical_evidence", reg["support_sources"]["physical_evidence"]["path"])
    epoch = plan["production185"]
    actual = base.verify_epoch(epoch["manifest"], epoch["archive"], 185, current=True)
    require(actual == epoch == reg["production185"], "Production185 full archived/current bytes differ")
    preflight = read(reg["preflights"][domain]["path"])
    scene = record(preflight["parsed_driver_arguments"]["scene_config"])
    require(scene["sha256"] == preflight["deployment_contract"]["scene_sha256"],
            "Actual fixture YAML must match the source-bound preflight scene SHA")
    return {"production185": actual, "selection_sources": reg["selection_sources"],
            "commands": reg["commands"], "preflight": reg["preflights"][domain], "scene_configuration": scene}


def admit(reg, domain, owner_path):
    """Admit an actually closed root tools session before opening any NPZ."""
    require(reg["schema"] == "local_reference.common50_future_audit_registration.v1", "Exact reader registration schema required")
    for row in reg["auditor_sources"]:
        verify(row)
    binding = source_binding(reg, domain)
    plan, case = read(reg["commands"]["path"]), reg["cases"][domain]
    require(str(Path(owner_path).absolute()) == case["owner"], "Registered future owner path required")
    owner_record = record(owner_path, sealed=True)
    owner = read(owner_path)
    require(owner["schema"] == "local_reference.common50_fixture_root_owner.v1"
            and owner["domain"] == domain and owner["status"] == "closed"
            and type(owner["actual_exit_code"]) is int and owner["actual_exit_code"] == 0
            and owner["child_exit_observed"] is True
            and type(owner["actual_tool_session_id"]) is int and owner["actual_tool_session_id"] > 0
            and all(isinstance(owner[key], str) and owner[key] for key in ("launch_chunk", "completion_chunk"))
            and owner["command"] == plan["GPU_commands"][domain]
            and owner["env"] == plan["GPU_environment"] and owner["cwd"] == str(ROOT)
            and owner["selected_source_before"] == owner["selected_source_after"] == binding,
            "Genuine registered closed0 root-tool owner/source binding required")
    weights = {key: plan[key] for key in ("controller", "planner")}
    require(owner["weights_before"] == owner["weights_after"] == weights,
            "Actual fixed62c4/b563 weight observations before/after required")
    for row in weights.values():
        verify(row)  # Opaque bytes only; never deserialize a weight payload.
    run = Path(case["output"])
    require(run.is_dir() and not run.is_symlink(), "Actual independent capture directory required")
    all_entries = list(run.rglob("*"))
    require(not any(path.is_symlink() for path in all_entries), "Capture inventory cannot contain symlinks")
    inventory = [record(path, sealed=True) for path in sorted(all_entries) if path.is_file()]
    require(owner["artifacts"] == inventory, "Whole readonly capture inventory must be exactly retained")
    required = {str(run/name) for name in ("result.json", "deployment_entry.json", "physical_states.npz",
                                         "ground_support.npz", "reference_plans.npz")}
    require(required <= {row["path"] for row in inventory}, "All five physical/reference artifacts required")
    expected, result, entry = read(reg["preflights"][domain]["path"]), read(run/"result.json"), read(run/"deployment_entry.json")
    require(entry == expected and result["arguments"] == expected["parsed_driver_arguments"]
            and result["status"] == "passed" and type(result["actual_control_steps"]) is int
            and result["actual_control_steps"] == 6000, "Exact source-bound preflight/recipe and full6000 required")
    verify(reg["support_sources"]["parent_registration"], sealed=True)
    parent = read(reg["support_sources"]["parent_registration"]["path"])
    verify(parent["historical_native"]["4"]["result"], sealed=True)
    historic = read(parent["historical_native"]["4"]["result"]["path"])["evaluation_actor"]
    verify(reg["support_sources"]["current185_actor_reference"], sealed=True)
    current_actor = read(reg["support_sources"]["current185_actor_reference"]["path"])["evaluation_actor"]
    actor = result["evaluation_actor"]
    require(actor == current_actor and actor["full_training_provenance"] == historic["full_training_provenance"],
            "Exact current185 actor-only62c4 metadata and original full training provenance required")
    verify(reg["support_sources"]["runtime_helper"], sealed=True)
    runtime_helper = load("common50_original_runtime_metadata", reg["support_sources"]["runtime_helper"]["path"])
    source = inspect.getsource(runtime_helper.validate_runtime_planner)
    before, after = 'runtime["diffusion_samples_per_actor"] == 1', 'runtime["diffusion_samples_per_actor"] == 10'
    require(source.count(before) == 1, "One original sample-count metadata guard required")
    namespace = dict(runtime_helper.__dict__)
    exec(compile(source.replace(before, after), __file__, "exec"), namespace)
    parent["cases"]["fixture_new"]["planner"] = plan["planner"]
    runtime = namespace["validate_runtime_planner"](result, parent, "fixture_new")
    selector = load("common50_recorded_selector_contract", reg["selection_sources"][1]["path"]).ReferenceSelector(domain)
    require(result["runtime_execution"]["candidate_selection"] == selector.describe()
            == expected["deployment_contract"]["candidate_selection"], "Actual source-bound candidate selection identity differs")
    request = entry["evaluation_actor_request"]
    require(request["base_checkpoint"] == {"path": plan["controller"]["path"], "sha256": plan["controller"]["sha256"]}
            and request["style_checkpoint"] is None and request["alpha"] == 0.0 and request["experimental_arm_style"] is False,
            "Original default actor request must remain alpha0 without style weights")
    rng = result["evaluation_rng_preparation"]
    require(rng["schema"] == "legacy_evaluation_initialization_rng.v1" and rng["seed"] == 7
            and all(rng[key] is False for key in ("saved_checkpoint_rng_restored", "trainer_created", "optimizer_created", "replay_created")),
            "Original actor-only seed7 initialization boundary differs")
    return {"schema": "local_reference.common50_actual_capture_admission.v1", "status": "passed", **binding,
            "root_closed_owner": owner_record, "weights_before_after": weights, "whole_capture_inventory": inventory,
            "runtime_planner_identity": runtime, "full_legacy_actor_provenance_exact": True,
            "checkpoint_deserialized_by_reader": False, "complete_runtime_import_ledger": False}
