"""SDK-free source and physical admission for future V12 Warehouse4/15.

This module starts no child, App or GPU. Preparation never reads the future
checkpoint, capture or audit. Root owns serial launches and actual closures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REGISTRATION = HERE / "registration.json"
REGISTRATION_SHA = "dd9950701b3c7c6883c1e0349e7e7b611b9804bd5ec52ab6ec9ffd1912369995"
SOURCE = ROOT / "docs/style_improvement_20261003/native_runtime_epoch_v12/native_pedestrian_source_v12.json"
SOURCE_SHA = "9ddbfe994bc692b344917796251780a08931d8d9846a1a19ae7476ba9753535c"
PROFILE = ROOT / "configs/diffusion_amp/amp_style_mixed_state_action_state_prior_pacer_reward.yaml"
PROFILE_SHA = "6d76dbfad6bc3827904af7efaf2fea7849ba5f5ff441971745ea17501b7f1bae"
EVALUATION = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_evaluation_v1"
SDK = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular local artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def prepare_sources():
    """Hash current production bytes and archive; never inspect future outputs."""
    manifest = record(SOURCE)
    require(manifest["sha256"] == SOURCE_SHA, "Exact V12 source manifest required")
    source = json.loads(SOURCE.read_text())
    require(source["schema"] == "selected_native_runtime_source.v12"
            and type(source["source_count"]) is int and source["source_count"] == len(source["files"]) == 185
            and len({row["path"] for row in source["files"]}) == 185, "Exact185 source roster required")
    for row in source["files"]:
        require(record(row["path"]) == row, "V12 production source changed: " + row["path"])
    require(record(source["archive"]["path"]) == source["archive"], "V12 source archive changed")
    profile = record(PROFILE)
    require(profile["sha256"] == PROFILE_SHA, "Registered opt-in PACER reward profile changed")
    return {"production_source": {"manifest": manifest, "archive": source["archive"], "source_count": 185,
                                  "all_archived_source_bytes_verified": True,
                                  "complete_runtime_import_ledger": False}, "profile": profile}


def read_registration():
    require(record(REGISTRATION)["sha256"] == REGISTRATION_SHA, "Registered native commands changed")
    registration = json.loads(REGISTRATION.read_text())
    require(registration["schema"] == "pacer_reward.registered_native_protocol.v1"
            and registration["status"] == "prepared_not_executed" and registration["future_effect"] is False,
            "Prepared native protocol required")
    current = prepare_sources()
    require(current == registration["current_source_binding"], "Registered native source differs")
    for row in registration["sources"].values():
        require(record(row["path"]) == row, "Registered resource/source changed: " + row["path"])
    return registration


def native_command(actors):
    require(type(actors) is int and actors in (4, 15), "Registered native actors4/15 only")
    return list(read_registration()["cases"][str(actors)]["command"])


def validate_terminal(document, checkpoint):
    """Reuse the independently frozen complete-payload admission, not a filename."""
    registration = read_registration()
    require(checkpoint["path"] == registration["future_checkpoint"]["path"], "Registered sealed step2000 only")
    executor = load(EVALUATION / "run_registered_evaluation.py", "native_complete_terminal_admission")
    evaluation = json.loads((EVALUATION / "registration.json").read_text())
    require(record(EVALUATION / "registration.json")["sha256"] == executor.REGISTRATION_SHA,
            "Frozen evaluation registration differs")
    executor.validate_registered_dependencies(evaluation)
    return executor.validate_admission(document, checkpoint, evaluation)


def validate_closed_receipt(receipt, actors, checkpoint):
    """Require an actual observed owner closure and unchanged selected bytes."""
    expected = read_registration()
    require(type(actors) is int and actors in (4, 15), "Native actors4/15 required")
    require(receipt.get("schema") == "pacer_reward.actual_native_launch.v1"
            and receipt.get("status") == "closed" and type(receipt.get("actors")) is int and receipt["actors"] == actors
            and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
            and receipt.get("child_exit_observed") is True and receipt.get("driver_completed") is True
            and receipt.get("prepare_only") is False and receipt.get("command") == expected["cases"][str(actors)]["command"]
            and checkpoint.get("path") == expected["future_checkpoint"]["path"]
            and receipt.get("amp_checkpoint") == checkpoint
            and receipt.get("selected_source_before") == expected["current_source_binding"]["production_source"]
            and receipt.get("selected_source_after") == expected["current_source_binding"]["production_source"],
            "Actual closed source-bound native process required")
    return receipt


def validate_closed_native(document, actors, checkpoint):
    """Physical completeness and zero contacts/falls; no style-score filter."""
    expected = read_registration()
    require(type(actors) is int and actors in (4, 15), "Native actors4/15 required")
    require(document.get("schema") == "pacer_reward.native200s_independent_CPU_audit.v1"
            and document.get("status") == "raw_sensor_source_contracts_passed_bounded_zero_fall_observed"
            and document.get("native_200s_artifacts_independently_audited") is True,
            "Actual complete native physical reader required")
    counts = document.get("counts", {})
    required = {"controls": 6000, "actors": actors, "duration_s": 200.0,
                "falls": 0, "static_actor_controls": 0, "peer_pair_controls": 0}
    require(isinstance(counts, dict) and all(type(counts.get(key)) is type(value) and counts[key] == value
                                           for key, value in required.items()), "Native complete zero-physical counts required")
    source = document.get("source_epoch", {})
    production = expected["current_source_binding"]["production_source"]
    require(source.get("manifest") == production["manifest"] and source.get("archive") == production["archive"]
            and type(source.get("source_count")) is int and source["source_count"] == 185
            and source.get("all_archived_source_bytes_verified") is True, "Actual V12 reader source epoch differs")
    admitted = document.get("actual_source_admission", {})
    require(admitted.get("schema") == "pacer_reward.actual_native_source_admission.v1"
            and admitted.get("production_source") == production and admitted.get("checkpoint") == checkpoint
            and admitted.get("full_training_provenance_exact") is True
            and admitted.get("trained_action_weights_not_reinitialized_by_reader") is True
            and admitted.get("reader_deserialized_checkpoint") is False,
            "Actual closed native source/controller provenance differs")
    require(document.get("execution_reader") == expected["sources"]["native_reader_main"],
            "Independent native physical reader bytes differ")
    owner = admitted.get("closed_owner_launch", {})
    require(owner.get("path") == expected["cases"][str(actors)]["actual_launch_record"]
            and record(owner["path"]) == owner, "Actual native closed-owner artifact differs")
    validate_closed_receipt(json.loads(Path(owner["path"]).read_text()), actors, checkpoint)
    return document


def validate_fixture(report, sidecar, receipt, checkpoint):
    """Actual full4x6000 raw measurement and CPU closure admit native4."""
    registration = read_registration()
    require(report.get("schema") == "pacer_reward.actual_navigation_ground_audit.v1" and report.get("status") == "passed",
            "Actual admitted fixture raw report required")
    counts = report.get("counts", {})
    required = {"controls": 6000, "actors": 4, "duration_s": 200.0, "physics_substeps": 24000,
                "foot_substep_observations": 384000, "falls": 0}
    require(all(type(counts.get(key)) is type(value) and counts[key] == value for key, value in required.items()),
            "Full6000 zero-fall fixture required")
    contacts = report.get("original_obstacle_peer_channels", {})
    require(contacts.get("floor_excluded") is True and all(type(contacts.get(key)) is int and contacts[key] == 0
            for key in ("static_contact_actor_frames", "peer_contact_actor_frames", "peer_contact_unordered_pair_frames")),
            "Actual fixture static and peer physical contacts must be zero")
    require(sidecar.get("schema") == "pacer_reward.actual_CPU_metric_source_admission.v1"
            and sidecar.get("status") == "passed" and sidecar.get("metric") == "navigation_raw"
            and sidecar.get("runner") == registration["sources"]["fixture_metric_runner"]
            and sidecar.get("catalog") == registration["sources"]["fixture_metric_catalog"],
            "Actual raw measurement source admission required")
    admitted = sidecar.get("actual_source_admission", {})
    require(admitted.get("registered_protocol") == registration["sources"]["evaluation_registration"]
            and admitted.get("isolated_source_manifest") == registration["controller_source_manifest"]
            and admitted.get("resources_full_sha256_recomputed", {}).get("controller_checkpoint") == checkpoint,
            "Actual fixture capture/controller/source differs")
    require(receipt.get("schema") == "pacer_reward.actual_CPU_metric_launch.v1" and receipt.get("metric") == "navigation_raw"
            and receipt.get("status") == "closed" and receipt.get("CPU_only") is True
            and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
            and receipt.get("child_exit_observed") is True and receipt.get("driver_completed") is True,
            "Actual raw CPU process closure required; preparation is insufficient")
    fixture = registration["fixture_inputs"]
    raw_path, sidecar_path = Path(fixture["raw_report_path"]), Path(fixture["source_sidecar_path"])
    require(json.loads(raw_path.read_text()) == report and json.loads(sidecar_path.read_text()) == sidecar
            and receipt.get("command") == fixture["actual_CPU_command"]
            and receipt.get("artifacts", {}).get("raw_report") == record(raw_path)
            and receipt.get("artifacts", {}).get("source_sidecar") == record(sidecar_path),
            "Actual raw CPU command/artifact closure binding differs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", required=True)
    parser.parse_args()
    registration = read_registration()
    print(json.dumps({"status": "prepared_source_binding_passed", "future_effect": False,
                      "source_count": 185, "GPU_used": False, "SDK_started": False,
                      "future_checkpoint_read": False, "future_capture_read": False,
                      "registered": record(REGISTRATION), "cases": list(registration["cases"])}))


if __name__ == "__main__":
    main()
