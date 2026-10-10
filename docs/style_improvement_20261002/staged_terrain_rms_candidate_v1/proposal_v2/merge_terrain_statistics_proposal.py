"""Explicit bounded terrain-statistics merge; dry run by default. Authorized task changes only.

This script has not been executed with --apply by its author. It does not launch
SDK/GPU or touch USD/SDK/root-data files. All target and payload hashes must match
before any repository write. Historical v5/v7 artifacts remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[3]
ALLOWED = ('CrowdSim/humanoid/application/settings.py', 'CrowdSim/humanoid/application/resources.py', 'CrowdSim/humanoid/training/trainer.py', 'CrowdSim/humanoid/runtime/checkpoint_contract.py', 'CrowdSim/humanoid/runtime/amp_inference.py', 'CrowdSim/humanoid/training/terrain_statistics.py', 'configs/diffusion_amp/amp_style_mixed_state_action_terrain_protected.yaml', 'tests/test_task_terrain_statistics.py', 'tests/test_terrain_statistics_integration.py')


def record(path):
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def verify():
    manifest = json.loads((DOC / "merge_manifest.json").read_text())
    rows = manifest["files"]
    if tuple(row["relative_path"] for row in rows) != ALLOWED:
        raise ValueError("Merge manifest differs from the exact bounded path allowlist")
    total = 0
    for row in rows:
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Merge path must be bounded inside this workspace")
        target, source = ROOT / relative, DOC / "proposed" / relative
        if target.resolve() != target or source.resolve() != source:
            raise ValueError("Symlink/redirected merge targets or payloads are not allowed")
        expected = row["before"]
        if expected is None:
            if target.exists():
                raise ValueError("New target already exists: " + str(relative))
        elif record(target) != expected:
            raise ValueError("Before target bytes changed: " + str(relative))
        if record(source) != row["after"] or row["after"]["bytes"] >= 250000:
            raise ValueError("Proposed payload changed or exceeds hard size limit")
        total += row["after"]["bytes"]
    if total != manifest["total_proposed_bytes"] or total >= 500000:
        raise ValueError("Proposed payload total exceeds bounded manifest")
    for guarded in manifest["protected_unchanged_files"]:
        if record(Path(guarded["path"])) != guarded:
            raise ValueError("Protected completed-runtime source changed")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write only after parent approval of this exact manifest")
    args = parser.parse_args(argv)
    manifest = verify()
    if not args.apply:
        print(json.dumps({"status": "verified_dry_run_no_writes", "targets": list(ALLOWED), "bytes": manifest["total_proposed_bytes"]}))
        return 0
    receipt = DOC / "merge_receipt.json"
    if receipt.exists():
        raise ValueError("Merge receipt already exists; no repeated/overwriting merge")
    # Prepare all same-directory temporary files before publishing any target.
    pending = []
    try:
        for row in manifest["files"]:
            target = ROOT / row["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".terrain-statistics-", dir=target.parent)
            temporary = Path(name)
            pending.append((temporary, target, row))
            with os.fdopen(fd, "wb") as stream:
                stream.write(Path(row["after"]["path"]).read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(target.stat().st_mode & 0o777 if target.exists() else 0o644)
        verify()
        for temporary, target, row in pending:
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != row["after"]["sha256"]:
                raise ValueError("Staged file changed before atomic publication")
            os.replace(temporary, target)
        published = [record(ROOT / row["relative_path"]) for row in manifest["files"]]
        if any(record["sha256"] != row["after"]["sha256"] for record, row in zip(published, manifest["files"])):
            raise ValueError("Published target SHA mismatch")
        with receipt.open("x") as stream:
            stream.write(json.dumps({"status": "bounded_merge_applied", "published": published,
                                     "terrain_statistics_merged_by_helper_only": True, "SDK_started": False, "GPU_used": False}, indent=2) + "\n")
        print(json.dumps({"status": "bounded_merge_applied", "files": len(published), "terrain_statistics_merged_by_helper_only": True}))
    finally:
        for temporary, _, _ in pending:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
