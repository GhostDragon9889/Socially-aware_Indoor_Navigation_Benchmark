"""Archive tested versions and seal CPU evidence, including retained failures."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FOLDERS = [HERE.parent / f"shared_social_native_initial_preparation_v{version}" for version in (1, 2, 3)]


def record(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Require an ordinary independent source/evidence file")
    content = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def same_bytes(left, right):
    return left["bytes"] == right["bytes"] and left["sha256"] == right["sha256"]


def main():
    import numpy as np
    final = json.loads((HERE / "CPU_preparation_execution.json").read_text())
    if (final["CPU_preparation_passed"] is not True or len(final["observed_children"]) != 6
            or not all(row["actual_child_returncode"] == 0 and row["actual_child_exit_observed"] is True
                       and record(row["log"]["path"]) == row["log"] for row in final["observed_children"])
            or (HERE / "tests_final.log").read_text().strip().splitlines()[-1] != "79 passed in 7.76s"):
        raise ValueError("Complete final actual CPU checks are missing")
    mutation = json.loads((HERE / "mutation_review.json").read_text())
    if (mutation["mutation_detected"] is not True or mutation["restored_exact_source_bytes"] is not True
            or mutation["actual_child"]["actual_child_returncode"] != 1
            or "3 failed, 39 passed" not in (HERE / "mutation_expected_failure.log").read_text()):
        raise ValueError("Mandatory mutation failure and restore are not observed")
    before = json.loads((HERE / "CPU_registration.json").read_text())["sources"]
    if any(record(row["path"]) != row for row in before):
        raise ValueError("Final tested source changed")

    actual_maps = json.loads((HERE / "actual_map_CPU_contracts.json").read_text())
    frames = []
    for case in actual_maps["cases"]:
        if record(case["archive"]["path"]) != case["archive"]:
            raise ValueError("Actual CPU map archive changed")
        with np.load(case["archive"]["path"], allow_pickle=False) as data:
            shape = data["native_scene_labels"].shape
            origin, bounds = data["native_scene_origin_xy"], data["native_scene_bounds_xy"]
            if (not np.allclose(bounds[:2], origin, rtol=0, atol=2e-4)
                    or not np.allclose(bounds[2:] - origin, np.array(shape[::-1]) * .1, rtol=0, atol=2e-4)):
                raise ValueError("Actual native lower-left bounds/grid extent differ")
            frames.append({"actors": case["protocol"]["actors"], "shape_hw": list(shape),
                "origin_xy_m": origin.tolist(), "bounds_xy_m": bounds.tolist(),
                "grid_extent_xy_m": (np.array(shape[::-1]) * .1).tolist(), "coordinate_fit_performed": False,
                "archive": case["archive"], "map_channels_only": True, "live_body_or_RNG_captured": False})
    write_new(HERE / "map_frame_CPU_review.json", {"cases": frames, "CPU_only": True,
              "native_SDK_execution_verified": False, "paper_effect_achieved": False})

    # Preserve the exact original path + bytes for each registration, including
    # the two prior source versions rather than relabeling them as final code.
    snapshots = HERE / "source_snapshot_v1"
    snapshots.mkdir(exist_ok=False)
    old_tests = FOLDERS[1] / "test_native_paired_initial_tested_v2.py"
    candidates = [*(folder / "native_initial_before_mutation.py" for folder in FOLDERS), old_tests]
    source_rows = []
    for folder in FOLDERS:
        registration = json.loads((folder / "CPU_registration.json").read_text())
        for selected in {row["path"]: row for row in registration["sources"]}.values():
            original = Path(selected["path"])
            archived_from = original if same_bytes(record(original), selected) else next(
                (path for path in candidates if path.exists() and same_bytes(record(path), selected)), None)
            if archived_from is None:
                raise ValueError("Missing exact registered prior source: " + selected["path"])
            target = snapshots / folder.name / original.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archived_from, target)
            archived = record(target)
            if not same_bytes(archived, selected):
                raise ValueError("Source archive bytes differ from the actual tested version")
            source_rows.append({"registration_directory": str(folder), "original": selected,
                                "copied_from": record(archived_from), "archive": archived})
    selected = record(__file__)
    target = snapshots / "final_seal" / Path(__file__).name
    target.parent.mkdir(parents=True)
    shutil.copyfile(__file__, target)
    source_rows.append({"original": selected, "archive": record(target)})
    archive = HERE / "source_snapshot_v1.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(snapshots, arcname="source_snapshot_v1")
    owner = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())
    guards = owner["bindings"]["guarded_sources"]
    if any(record(row["path"]) != row for row in guards):
        raise ValueError("Active S3 training source guard changed")
    root_observed = {"schema": "shared_social.native_initial_root_observed_CPU_exits.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True,
        "outer_preparation_exits": [
            {"version": 1, "launch_tool_chunk": "0c7977", "terminal_tool_chunk": "06f77f", "actual_exit_code": 1},
            {"version": 2, "launch_tool_chunk": "c8bf83", "terminal_tool_chunk": "192a75", "actual_exit_code": 0},
            {"version": 3, "launch_tool_chunk": "a016eb", "terminal_tool_chunk": "4df74c", "actual_exit_code": 0}],
        "final_CPU_child_execution": record(HERE / "CPU_preparation_execution.json"),
        "source_archive": record(archive), "sources": source_rows, "seal_script": selected,
        "active_training_guards_unchanged": True, "active_training_guard_count": len(guards),
        "actual_native_SDK_exit_observed": False, "native_initial_pairing_verified": False,
        "paper_effect_achieved": False, "default_model_promoted": False}
    write_new(HERE / "root_observed_CPU_execution.json", root_observed)
    files = sorted(path for folder in FOLDERS for path in folder.rglob("*") if path.is_file())
    identities = [record(path) for path in files]
    for path in files:
        path.chmod(0o444)
    if [record(path) for path in files] != identities:
        raise ValueError("CPU evidence bytes changed while applying readonly permissions")
    index = HERE / "CPU_evidence_seal.json"
    write_new(index, {"schema": "shared_social.native_initial_CPU_readonly_seal.v1", "files": identities,
        "count": len(identities), "bytes_unchanged_after_readonly": True,
        "actual_native_SDK_execution_verified": False, "paper_effect_achieved": False})
    index.chmod(0o444)
    print(json.dumps({"status": "native_initial_CPU_evidence_sealed", "sealed_files": len(identities),
        "receipt": record(HERE / "root_observed_CPU_execution.json"), "seal": record(index),
        "actual_native_SDK_execution_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
