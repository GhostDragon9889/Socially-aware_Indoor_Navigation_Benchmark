"""Archive native style CPU preparation and preserved JSON-boundary failure."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.with_name("shared_social_native_style_preparation_v1")
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import native_navigation_style as style  # noqa: E402

record = style.metric.record


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot(row, target):
    if record(row["path"]) != row or target.exists():
        raise ValueError("Source changed or snapshot already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(row["path"], target)
    copied = record(target)
    if (copied["bytes"], copied["sha256"]) != (row["bytes"], row["sha256"]):
        raise ValueError("Source snapshot differs")
    return {"source": row, "snapshot": copied}


def main():
    final = json.loads((HERE / "CPU_preparation_execution.json").read_text())
    if len(final["children"]) != 4 or any(row["actual_child_exit_code"] != 0 for row in final["children"]):
        raise ValueError("Actual final CPU children did not complete successfully")
    for row in final["children"]:
        if record(row["log"]["path"]) != row["log"]:
            raise ValueError("Observed child log changed")
    if "53 passed" not in (HERE / "tests_final.log").read_text():
        raise ValueError("Final tests differ")
    equality = json.loads((HERE / "historical_N4_style_metric_preservation.json").read_text())
    if (equality["paired_common_bin_every_field_equal"] is not True
            or equality["formal_new_native_SDK_resource_admission_run"] is not False):
        raise ValueError("Historical regression scope/metrics disagree")
    for arm in equality["runs"].values():
        if not all(arm[key] is True for key in ("inclusive_summary_every_field_equal", "pose45_every_field_equal",
                                               "motion90_population_every_field_equal")):
            raise ValueError("Historical metric changed")
    for row in equality["inputs"]:
        if record(row["path"]) != row:
            raise ValueError("Historical actual inputs changed")
    snapshots = HERE / "source_snapshot_v1"
    archived = []
    for folder in (OLD, HERE):
        registration = json.loads((folder / "CPU_registration.json").read_text())
        seen = set()
        for row in [*registration["sources"], *style._identities(registration["bindings"])]:
            path = Path(row["path"])
            if path.suffix != ".py" or row["path"] in seen:
                continue
            seen.add(row["path"])
            archived.append(snapshot(row, snapshots / folder.name / path.relative_to(ROOT)))
        mutation = json.loads((folder / "mutation_review.json").read_text())
        if (mutation["mutation_detected"] is not True or mutation["exact_source_bytes_restored"] is not True
                or mutation["actual_child"]["actual_child_exit_code"] == 0):
            raise ValueError("Actual mutation/restoration missing")
        row = record(folder / "native_style_reader_mutated_TEST_ONLY.py")
        archived.append(snapshot(row, snapshots / "TEST_ONLY_MUTATIONS" / folder.name / Path(row["path"]).name))
    # Reject the actually completed old N4 raw CPU receipt through the new CLI.
    old_fixture = ROOT / "docs/style_improvement_20261005/shared_social_S1_paired_actual_v1"
    output = HERE / "MUST_NOT_EXIST_old_N4_as_native_style.json"
    command = [sys.executable, str(ROOT / "scripts/evaluate_shared_social_native_style.py"), "--actors", "3",
        "--scene-config", str(ROOT / "docs/style_improvement_20261005/shared_social_native_SFM_preparation_v1/scenes/Warehouse_N03.json"),
        "--scene-sha256", "309abc3c5a99f71e1146f9ee0063ee38e9665330fd0d482d6af5869c906df200",
        "--diffusion-audit-execution", str(old_fixture / "diffusion200s_raw_CPU_execution.json"),
        "--sfm-audit-execution", str(old_fixture / "sfm200s_raw_CPU_execution.json"), "--output", str(output)]
    log = HERE / "actual_old_N4_CLI_rejection.log"
    with log.open("xb") as stream:
        rejected = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if (rejected.returncode == 0 or output.exists()
            or "Require externally observed successful native raw CPU audit" not in log.read_text()):
        raise ValueError("Actual old fixture receipt was not rejected before native output")
    write_new(HERE / "actual_old_N4_CLI_rejection.json", {"command": command,
        "actual_child_exit_code": rejected.returncode, "actual_child_exit_observed": True,
        "log": record(log), "output_created": False, "SDK_started": False, "quality_measured": False})
    archived.append(snapshot(record(__file__), snapshots / "final_seal" / Path(__file__).name))
    archive = HERE / "source_snapshot_v1.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(snapshots, arcname="source_snapshot_v1")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if any(record(row["path"]) != row for row in guards):
        raise ValueError("S3 source guard changed")
    write_new(HERE / "root_observed_CPU_execution.json", {
        "schema": "shared_social.native_style_root_observed_CPU_exits.v1",
        "root_observed_outer_processes": [
            {"version": 1, "launch_tool_chunk": "2ecc72", "terminal_tool_chunk": "9e4585", "actual_exit_code": 1,
             "failure": "Historical common-bin Python tuple/JSON list representation mismatch", "failure_retained": True},
            {"version": 2, "launch_tool_chunk": "b1b8eb", "terminal_tool_chunk": "acc542", "actual_exit_code": 0}],
        "final_child_execution": record(HERE / "CPU_preparation_execution.json"),
        "historical_N4_style_equality": record(HERE / "historical_N4_style_metric_preservation.json"),
        "known_representation_review": record(HERE / "historical_common_bin_representation_review.json"),
        "old_actual_fixture_rejected": record(HERE / "actual_old_N4_CLI_rejection.json"),
        "source_snapshots": archived, "source_archive": record(archive), "S3_sources_guarded": guards,
        "S3_source_guard_count": len(guards), "training_parameters_or_data_changed": False,
        "new_native_capture_or_quality_measured": False, "paper_effect_achieved": False, "default_model_promoted": False})
    paths = sorted(path for folder in (OLD, HERE) for path in folder.rglob("*") if path.is_file())
    for path in paths:
        path.chmod(0o444)
    identities = [record(path) for path in paths]
    if any(record(row["path"]) != row for row in identities):
        raise ValueError("Evidence changed while sealing")
    seal = HERE / "CPU_evidence_seal.json"
    write_new(seal, {"schema": "shared_social.native_style_CPU_readonly_seal.v1", "files": identities,
        "indexed_file_count": len(identities), "indexed_files_readonly": True,
        "external_seal_tool_exit_receipt_included_in_index": False, "new_native_capture_or_quality_measured": False})
    seal.chmod(0o444)
    print(json.dumps({"status": "native_style_CPU_evidence_sealed", "sealed_files": len(identities),
        "receipt": record(HERE / "root_observed_CPU_execution.json"), "seal": record(seal),
        "S3_training_guards_unchanged": True, "new_native_capture_or_quality_measured": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
