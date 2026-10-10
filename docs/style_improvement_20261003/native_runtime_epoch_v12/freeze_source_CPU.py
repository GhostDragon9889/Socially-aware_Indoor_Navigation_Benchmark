"""Archive185 explicitly selected production sources; leave all source files intact."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def emit(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def make_tar(payloads):
    result = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=result, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for relative, contents in sorted(payloads.items()):
                entry = tarfile.TarInfo(relative)
                entry.size = len(contents)
                entry.mode, entry.mtime = 0o644, 0
                entry.uid = entry.gid = 0
                entry.uname = entry.gname = ""
                archive.addfile(entry, io.BytesIO(contents))
    return result.getvalue()


def main():
    parent = ROOT / "docs/style_improvement_20261003/native_runtime_epoch_v11/native_pedestrian_source_v11.json"
    baseline = json.loads(parent.read_text())
    assert baseline["source_count"] == len(baseline["files"]) == 183
    changed = {"CrowdSim/humanoid/application/configuration.py", "scripts/train_paper_amp.py"}
    added = {"configs/diffusion_amp/amp_style_mixed_state_action_state_prior_pacer_reward.yaml",
             "tests/test_amp_resume_reward_config.py"}
    guard_root = ROOT / "docs/style_improvement_20261003/training_style_root_cause_v1/resume_reward_guard"
    guard = json.loads((guard_root / "staged_change_manifest.json").read_text())
    expected_guard = {row["relative_path"]: row for row in guard["files"]}
    rows, approved, payloads = [], [], {}
    for before in baseline["files"]:
        path = Path(before["path"])
        relative = str(path.relative_to(ROOT))
        assert path.is_file() and not path.is_symlink()
        after = record(path)
        if relative in changed:
            assert after["sha256"] == expected_guard[relative]["production_after"]["sha256"]
            assert before["sha256"] == expected_guard[relative]["production_before"]["sha256"]
            approved.append({"path": relative, "before": before, "after": after, "authorized_change": True})
        else:
            assert (before["bytes"], before["sha256"]) == (after["bytes"], after["sha256"]), relative
        rows.append(after)
        payloads[relative] = path.read_bytes()
    assert len(approved) == 2
    for relative in sorted(added):
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink() and relative not in payloads
        rows.append(record(path))
        payloads[relative] = path.read_bytes()
    assert len(rows) == len(payloads) == 185
    public_receipt = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_experiment_v1/actual_public_profile_CPU.json"
    public = json.loads(public_receipt.read_text())
    profile = ROOT / "configs/diffusion_amp/amp_style_mixed_state_action_state_prior_pacer_reward.yaml"
    assert public["status"] == "passed" and public["new"]["sha256"] == record(profile)["sha256"]
    assert public["training_changed"] == {"amp_reward_weight": [2., .5]}
    assert public["default_checkpoint_changed"] is False
    old_test = guard_root / "test_resume_reward_guard.py"
    new_test = ROOT / "tests/test_amp_resume_reward_config.py"
    assert old_test.read_text().replace("parents[4]", "parents[1]") == new_test.read_text()

    # Verify existing immutable934 training source and six recorded resources.
    # The test bank is hashed as opaque bytes; no sealed34 outputs are accessed.
    isolated_manifest = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_v1/source_manifest.json"
    isolated = json.loads(isolated_manifest.read_text())
    isolated_bundle = isolated_manifest.parent
    source_protection = []
    assert len(isolated["files"]) == 934
    for row in isolated["files"]:
        path = isolated_bundle / row["path"]
        after = record(path)
        assert after["sha256"] == row["sha256"] and after["bytes"] == row["bytes"]
        assert path.stat().st_mode & 0o222 == 0
        source_protection.append({"before": row, "actual": after, "unchanged": True})
    resource_protection = []
    for row in isolated["resources"]:
        path = Path(row["path"])
        after = record(path)
        assert after["sha256"] == row["sha256"] and after["bytes"] == row["bytes"]
        resource_protection.append({"before": row, "actual": after, "unchanged": True})
    assert len(resource_protection) == 6
    selected = ROOT / "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt"
    selected_record = record(selected)
    assert selected_record["sha256"] == "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
    selected_OS_readonly = selected.stat().st_mode & 0o222 == 0
    protection_path = OUT / "protected_fingerprints.json"
    emit(protection_path, {"schema": "native_runtime_epoch.v12.protected_fingerprints.v1",
                          "immutable_934_source_manifest": record(isolated_manifest),
                          "source_count": 934, "source_rows": source_protection,
                          "resource_count": 6, "resources": resource_protection,
                          "resource_total_bytes": sum(row["actual"]["bytes"] for row in resource_protection),
                          "selected_default_checkpoint": selected_record,
                          "selected_default_OS_readonly": selected_OS_readonly,
                          "selected_default_mode": oct(selected.stat().st_mode & 0o777),
                          "data_deserialized": False, "sealed34_outputs_read": False})
    first, second = make_tar(payloads), make_tar(payloads)
    assert first == second
    tar_path = OUT / "native_pedestrian_source_v12.tar.gz"
    with tar_path.open("xb") as stream:
        stream.write(first)
    with tarfile.open(tar_path, "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == 185 and {entry.name for entry in members} == set(payloads)
        for entry in members:
            assert entry.isfile() and archive.extractfile(entry).read() == payloads[entry.name]
    for row in rows:
        assert record(row["path"]) == row  # No concurrent production-source change.
    guard_peer = ROOT / "docs/style_improvement_20261003/resume_reward_guard_peer_review_v1/review_v2.json"
    guard_review = json.loads(guard_peer.read_text())
    assert guard_review["status"] == "passed" and guard_review["findings"] == []
    result = {"schema": "selected_native_runtime_source.v12", "created_utc": datetime.now(timezone.utc).isoformat(),
              "files": sorted(rows, key=lambda row: row["path"]), "source_count": 185,
              "archive": record(tar_path), "parent_manifest": record(parent),
              "parent_manifest_sha256": record(parent)["sha256"],
              "parent_archive": baseline["archive"], "parent_existing_files_changed": 2,
              "parent_existing_files_unchanged": 181, "new_source_files": sorted(added),
              "approved_changes": approved,
              "scope": "parent183 preserved except two approved resume reward guard files; one explicit opt-in profile and one canonical regression test added",
              "change_kind": "training-resume reward identity guard and opt-in PACER AMP reward-factor recipe",
              "fresh_training_algorithm_changed_by_guard": False, "evaluation_algorithm_changed_by_guard": False,
              "fresh_guard_RNG_draws_added": False, "default_checkpoint_changed": False,
              "new_profile_effective_softplus_coefficient": 1., "guard_peer_review": record(guard_peer),
              "selected_default_OS_readonly": selected_OS_readonly,
              "canonical_test_identity": {"snapshot": record(old_test), "production": record(new_test),
                                            "only_delta": "ROOT.parents4 to1 for canonical tests directory"},
              "public_profile_actual_CPU": record(public_receipt),
              "protected_fingerprints": record(protection_path),
              "deterministic_tar_proof": {"builds_compared": 2, "archive_bytes_equal": True,
                                          "archive_members_reverified": 185, "gzip_mtime": 0, "tar_mtime": 0,
                                          "uid": 0, "gid": 0, "mode": "0644", "format": "USTAR",
                                          "symlinks_or_hardlinks": 0},
              "old934_source_changed": False, "known_data_weights_changed": False,
              "SDK_modified_by_freeze": False, "production_sources_modified_by_freeze": False,
              "GPU_or_Kit_started": False, "style_improvement_proven_by_source_freeze": False}
    destination = OUT / "native_pedestrian_source_v12.json"
    emit(destination, result)
    for path in (destination, tar_path, protection_path):
        path.chmod(0o444)
    print(json.dumps({"status": "passed", "manifest": record(destination), "archive": record(tar_path),
                      "protected": record(protection_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
