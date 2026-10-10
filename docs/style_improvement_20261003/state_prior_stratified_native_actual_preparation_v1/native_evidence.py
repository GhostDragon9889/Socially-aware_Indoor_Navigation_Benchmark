"""q98 matched native metadata admission; immutable native raw math is separate."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREPARED = ROOT / "docs/style_improvement_20261003/state_prior_stratified_native_preparation_v1"
BASE_DIR = ROOT / "docs/style_improvement_20261003/default_newDP_physical_preparation_v1"
CORE_DIR = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_native_reader_preparation_v1"


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


BASE = load("q_native_original_physical_metadata", BASE_DIR / "physical_evidence.py")
PREP = load("q_native_frozen_terminal_fixture", PREPARED / "native_admission.py")
read, record, require = BASE.read, BASE.record, BASE.require
CPU_ENV = BASE.CPU_ENV
GPU_ENV = {**BASE.GPU_ENV, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
SOURCE = PREP.prepare_sources()["production_source"]["manifest"]["path"]


def registration():
    frozen = read(HERE / "source_freeze_v1.json")
    require(frozen["status"] == "frozen_prepared_only", "Frozen q native source packet required")
    for row in frozen["files"]:
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222,
                "Own q native source changed")
    plan = read(HERE / "registration.json")
    for row in plan["frozen_inputs"].values():
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222,
                "Frozen q/native parent input changed")
    packet = read(PREPARED / "source_freeze_v1.json")
    for row in packet["files"]:
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222,
                "Frozen q source/preflight packet changed")
    parent = PREP.prepare_sources()
    for actors in (4, 15):
        case = plan["cases"][f"native{actors}"]
        require(case["command"] == parent["cases"][str(actors)]["command"]
                and case["output"] == parent["cases"][str(actors)]["output"]
                and case["preflight"] == read(PREPARED / f"actual_native{actors}_SDK_free_preflight.json"),
                "Only already prepared matched q native commands may execute")
    return plan


def production_binding():
    parent = PREP.prepare_sources()
    source = parent["production_source"]
    BASE.verify_epoch(source["manifest"], source["archive"], 185, current=True)
    return source


def source_binding(plan):
    require(plan == registration(), "Exact q native owner plan required")
    terminal = PREP.validate_terminal()
    fixture = PREP.validate_fixture()
    for row in plan["opaque_inputs"].values():
        require(record(row["path"]) == row, "Native opaque checkpoint/resource SHA changed")
    return {"production_source": production_binding(), "registration": record(HERE / "registration.json"),
            "source_freeze": record(HERE / "source_freeze_v1.json"), "terminal": terminal,
            "fixture": fixture, "opaque_inputs": plan["opaque_inputs"],
            "complete_runtime_import_ledger": False}


def run_path(actors):
    require(type(actors) is int and actors in (4, 15), "Integer4/15 native actor count required")
    return Path(registration()["cases"][f"native{actors}"]["output"])


def expected_counts(actors):
    # Exact existing native denominator function; no new physical metric formulas.
    original = load("q_native_original_denominators", CORE_DIR / "native_evidence.py")
    return original.expected_counts(actors)


def prepare_evidence(actors):
    plan = registration()
    PREP.validate_terminal()
    PREP.validate_fixture()
    return {"production_source": production_binding(), "native_protocol_registration": record(HERE / "registration.json"),
            "native_source_validator": record(__file__), "expected_native_command": plan["cases"][f"native{actors}"]["command"],
            "actual_checkpoint": plan["checkpoint"], "terminal_admission": plan["frozen_inputs"]["actual_q_terminal_audit"],
            "actual_terminal_artifacts_pending": False, "native_capture_pending": True,
            "native_execution_audited": False, "checkpoint_deserialized_during_preparation": False}


def result_metadata_source():
    original = load("q_native_original_result_metadata", CORE_DIR / "native_evidence.py")
    source = inspect.getsource(original.validate_result_metadata)
    changes = (
        ('all(request["base_checkpoint"].get(key)==value for key,value in checkpoint.items())',
         'all(request["base_checkpoint"].get(key)==checkpoint[key] for key in ("path", "sha256"))'),
        ('"6d76dbfad6bc3827904af7efaf2fea7849ba5f5ff441971745ea17501b7f1bae"',
         '"220f6d38d4cd71d8cc26f045bf2f3f842661ecc7f69875c11e4a8ee6df39e7c2"'),
        ('"Native requested PACER reward profile differs"', '"Native requested original weight2 profile differs"'),
    )
    revised = source
    for old, new in changes:
        require(revised.count(old) == 1, "One original producer metadata boundary required")
        revised = revised.replace(old, new)
    reverse = revised
    for old, new in reversed(changes):
        reverse = reverse.replace(new, old)
    require(reverse == source, "Only producer path/SHA ABI and requested profile identity may change")
    return original, revised


def closed_owner(owner, plan, phase, *, cpu=False):
    case = plan["cases"][phase]
    require(owner.get("schema") == ("stratified_prior.actual_native_CPU_audit_launch.v1" if cpu
                                    else "stratified_prior.actual_native_physical_launch.v1")
            and owner.get("status") == "closed" and type(owner.get("exit_code")) is int and owner["exit_code"] == 0
            and owner.get("child_exit_observed") is True and owner.get("driver_completed") is True
            and owner.get("phase") == phase and owner.get("command") == case["CPU_command" if cpu else "command"]
            and owner.get("owner") == record(HERE / "launch_registered.py")
            and owner.get("registration") == record(HERE / "registration.json")
            and owner.get("binding_before") == owner.get("binding_after") == source_binding(plan),
            "Real complete q native child/owner/source closure required")
    require(owner.get("actual_log") == record(owner["log"]) and not Path(owner["log"]).stat().st_mode & 0o222,
            "Readonly actual native owner log differs")


def admit_capture(plan, phase):
    require(plan == registration() and phase in ("native4", "native15"), "Exact native phase/plan required")
    actors = plan["cases"][phase]["actors"]
    case = plan["cases"][phase]
    owner = read(case["GPU_receipt"])
    require(not Path(case["GPU_receipt"]).stat().st_mode & 0o222, "Native GPU owner must be readonly closed")
    closed_owner(owner, plan, phase)
    artifacts = {row["path"]: row for row in owner.get("artifacts", [])}
    actual_files = {str(path) for path in Path(case["output"]).rglob("*") if path.is_file()}
    require(set(artifacts) == actual_files and actual_files, "All actual native files must remain in closed owner ledger")
    for path, row in artifacts.items():
        require(record(path) == row and not Path(path).stat().st_mode & 0o222, "Actual native file changed/unsealed")
    entry_path, result_path = Path(case["output"]) / "deployment_entry.json", Path(case["output"]) / "result.json"
    entry, result = read(entry_path), read(result_path)
    require(entry == case["preflight"], "Whole actual entry differs from real SDK-free matched preflight")
    terminal = read(plan["frozen_inputs"]["actual_q_terminal_audit"]["path"])
    training_row = terminal["terminal_training_result"]
    require(record(training_row["path"]) == training_row, "Actual q training result changed")
    training = read(training_row["path"])
    original, revised = result_metadata_source()
    namespace = {**original.__dict__, "run_path": run_path, "record": record, "require": require}
    exec(compile(revised, __file__, "exec"), namespace)
    namespace["validate_result_metadata"](result, entry, training, plan["checkpoint"], actors)
    require(result["provenance"] == training["provenance"]
            and result["provenance"]["diffusion_checkpoint"] == plan["planner"]["path"]
            and result["provenance"]["diffusion_sha256"] == plan["planner"]["sha256"]
            and result.get("evaluation_planner_change") is None,
            "Whole q historical provenance and unchanged actual old DP63e required")
    runtime = result["runtime_execution"]
    require(result["trajectory_source"] == "diffusion" and runtime["diffusion_used_for_rollouts"] is True
            and runtime["diffusion_steps"] == 100 and runtime["diffusion_samples_per_actor"] == 1
            and runtime["precision"] == "FP32" and runtime["compiled_forward"] is False,
            "Actual native old DP100steps/S1/FP32 runtime differs")
    return {"schema": "stratified_prior.actual_native_source_admission.v1", "status": "passed", "actors": actors,
            "run": case["output"], "result": record(result_path), "entry": record(entry_path),
            "closed_owner_launch": record(case["GPU_receipt"]), "owner_log": record(owner["log"]),
            "production_source": production_binding(), "terminal_admission": plan["frozen_inputs"]["actual_q_terminal_audit"],
            "checkpoint": plan["checkpoint"], "planner": plan["planner"],
            "native_protocol_registration": record(HERE / "registration.json"), "native_source_validator": record(__file__),
            "full_training_provenance_exact": True, "trained_action_weights_not_reinitialized_by_reader": True,
            "reader_deserialized_checkpoint": False, "complete_runtime_import_ledger": False,
            "source_boundary": "Actual production185 pre/post, q98 terminal and original DP63e; no MMD admission filter."}


def admit_closed_native(actors, registered):
    require(registered["actors"] == actors and registered["run"] == str(run_path(actors))
            and registered["native_execution_audited"] is False and registered["expected_counts"] == expected_counts(actors)
            and registered["evidence"] == prepare_evidence(actors), "Exact q native raw reader source/denominators required")
    return admit_capture(registration(), f"native{actors}")


def native4_gate(plan):
    case = plan["cases"]["native4"]
    admission = admit_capture(plan, "native4")
    owner, report = read(case["CPU_receipt"]), read(case["audit"])
    require(not Path(case["CPU_receipt"]).stat().st_mode & 0o222, "Actual native4 CPU owner must be readonly")
    closed_owner(owner, plan, "native4", cpu=True)
    require(owner.get("audit_artifact") == record(case["audit"])
            and owner.get("actual_capture_admission") == report.get("actual_source_admission") == admission
            and report.get("execution_entry") == record(HERE / "audit_native.py")
            and report.get("schema") == "stratified_prior.native200s_independent_CPU_audit.v1"
            and report.get("native_200s_artifacts_independently_audited") is True,
            "Real native4 original raw audit/source closure required")
    counts = report["counts"]
    require(counts["controls"] == 6000 and counts["actors"] == 4 and counts["duration_s"] == 200.0
            and all(type(counts.get(key)) is int and counts[key] == 0
                    for key in ("falls", "static_actor_controls", "peer_pair_controls")),
            "Complete native4 zero-fall/static/peer condition required before15; failed rows retained")
    return {"status": "passed_native4_raw_physical_gate", "raw_report": record(case["audit"]),
            "actual_CPU_owner": record(case["CPU_receipt"]), "MMD_launch_filter": False}


def raw_core():
    # The original d707 reader imports its original metadata module, then only hooks are replaced.
    original = load("native_evidence", CORE_DIR / "native_evidence.py")
    sys.modules["native_evidence"] = original
    core = load("q_native_immutable_full_raw_reader", CORE_DIR / "audit_state_prior_native.py")
    core.SOURCE = SOURCE
    core.run_path, core.source_binding = run_path, production_binding
    core.prepare_evidence, core.expected_counts = prepare_evidence, expected_counts
    core.admit_closed_native = admit_closed_native
    return core
