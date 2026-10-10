"""SDK-free q terminal adapter for future matched Warehouse4/15 commands."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular local source required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare_sources():
    plan = read(HERE / "registration.json")
    for row in plan["registered_sources"].values():
        require(record(row["path"]) == row, "Registered native metadata/source changed")
    epoch = plan["production_source"]
    manifest = read(epoch["manifest"]["path"])
    require(record(epoch["manifest"]["path"]) == epoch["manifest"]
            and record(epoch["archive"]["path"]) == epoch["archive"]
            and manifest["source_count"] == len(manifest["files"]) == 185,
            "Exact current production185 required")
    for row in manifest["files"]:
        require(record(row["path"]) == row, "Production185 source changed")
    return plan


def validate_terminal():
    """Use the actual q payload reader without reloading checkpoint tensors."""
    plan = prepare_sources()
    source = plan["registered_sources"]["q_evaluation_executor"]
    specification = importlib.util.spec_from_file_location("q_native_actual_terminal_admission", source["path"])
    executor = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(executor)
    evaluation = read(plan["registered_sources"]["q_evaluation_registration"]["path"])
    executor.validate_registered_dependencies(evaluation)
    document = read(plan["registered_sources"]["actual_q_terminal_audit"]["path"])
    executor.validate_admission(document, plan["checkpoint"], evaluation)
    seal = read(plan["registered_sources"]["actual_q_root_seal"]["path"])
    require(seal["checkpoint"] == plan["checkpoint"] and seal["actual_updates"] == 2000
            and seal["actual_physical_transitions"] == 16384000
            and seal["status"] == "sealed_after_actual_GPU_and_watch_exit0"
            and Path(plan["checkpoint"]["path"]).stat().st_mode & 0o222 == 0
            and Path(plan["checkpoint"]["path"]).stat().st_nlink == 1,
            "Actual q readonly independent terminal/root closure differs")
    return {"status": "passed_actual_q_terminal_metadata", "checkpoint": plan["checkpoint"],
            "terminal_audit": plan["registered_sources"]["actual_q_terminal_audit"],
            "private_RNG_registry_and8192000windows_2000batches_delegated_to_original_q_validator": True,
            "current_production185_tensor_load_performed_by_this_adapter": False,
            "native_capture_read": False, "physical_quality_verified": False}


def validate_fixture():
    """After root's actual raw closure only; do not filter with90/45 style scores."""
    plan = prepare_sources()
    validate_terminal()
    paths = plan["fixture_future_paths"]
    receipt = read(paths["actual_CPU_owner_receipt"])
    require(receipt.get("schema") == "stratified_prior.actual_CPU_metric_launch.v1"
            and receipt.get("status") == "closed" and type(receipt.get("exit_code")) is int
            and receipt["exit_code"] == 0
            and receipt.get("actual_process_owner") == "tools.exec_command / tools.write_stdin"
            and type(receipt.get("actual_tool_session_id")) is int
            and isinstance(receipt.get("actual_completion_chunk"), str) and receipt["actual_completion_chunk"]
            and receipt.get("command") == paths["actual_CPU_command"],
            "Actual root tool session/completion/exit required; no invented Popen fields")
    for role, path in (("raw_report", paths["raw_report"]), ("source_sidecar", paths["source_sidecar"])):
        require(receipt.get("artifacts", {}).get(role) == record(path)
                and not Path(path).stat().st_mode & 0o222, "Actual readonly raw/sidecar identity differs")
    report, sidecar = read(paths["raw_report"]), read(paths["source_sidecar"])
    require(report.get("schema") == "stratified_prior.actual_navigation_ground_audit.v1"
            and report.get("status") == "passed", "Actual q raw measurement report required")
    expected = {"controls": 6000, "actors": 4, "duration_s": 200.0, "physics_substeps": 24000,
                "foot_substep_observations": 384000, "falls": 0}
    counts = report.get("counts", {})
    require(all(type(counts.get(key)) is type(value) and counts[key] == value for key, value in expected.items()),
            "Full4x6000/200s zero-fall raw measurement required")
    contacts = report.get("original_obstacle_peer_channels", {})
    require(contacts.get("floor_excluded") is True and all(type(contacts.get(key)) is int and contacts[key] == 0
            for key in ("static_contact_actor_frames", "peer_contact_actor_frames", "peer_contact_unordered_pair_frames")),
            "Static/peer raw contact gate failed")
    admitted = sidecar.get("actual_source_admission", {})
    sources = plan["registered_sources"]
    require(sidecar.get("schema") == "stratified_prior.actual_CPU_metric_source_admission.v1"
            and sidecar.get("status") == "passed" and sidecar.get("metric") == "navigation_raw"
            and sidecar.get("runner") == sources["q_CPU_metric_runner"]
            and sidecar.get("catalog") == sources["q_CPU_metric_catalog"]
            and admitted.get("registered_protocol") == sources["q_evaluation_registration"]
            and admitted.get("isolated_source_manifest") == plan["q_training_source_manifest"]
            and admitted.get("resources_full_sha256_recomputed", {}).get("controller_checkpoint") == plan["checkpoint"],
            "Actual raw q controller/source admission differs")
    return {"status": "passed_actual_full_fixture_physical_gate", "MMD_launch_filter": False,
            "raw_report": record(paths["raw_report"]), "source_sidecar": record(paths["source_sidecar"]),
            "actual_CPU_owner": record(paths["actual_CPU_owner_receipt"]), "checkpoint": plan["checkpoint"]}


if __name__ == "__main__":
    print(json.dumps(validate_terminal(), indent=2))
