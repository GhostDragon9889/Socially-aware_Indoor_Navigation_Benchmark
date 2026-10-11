"""Archived execution evidence must never admit changed live execution code."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from CrowdSim.humanoid.application import historical_recovery as history
from scripts import own_shared_social_recovery_evaluation as owner
from scripts import resume_shared_social_recovery_evaluation as resume


def test_all_snapshots_have_the_original_complete_bytes():
    for relative, (_, size, digest) in history.SNAPSHOTS.items():
        assert history.source_record(history.ROOT / relative) == {
            "path": str(history.ROOT / relative), "bytes": size, "sha256": digest}


@pytest.mark.parametrize("mutation", ["digest", "size", "path", "extra", "bool_size"])
def test_historical_source_record_rejects_unregistered_identity(mutation):
    item = history.source_record(history.ROOT / "src/CrowdSim/humanoid/runtime/social_popart_training.py")
    if mutation == "digest":
        item["sha256"] = "0" * 64
    elif mutation == "size":
        item["bytes"] += 1
    elif mutation == "path":
        item["path"] = str(history.ROOT / "scripts/train_shared_social.py")
    elif mutation == "extra":
        item["unverified"] = True
    else:
        item["bytes"] = True
    with pytest.raises(ValueError, match="source changed"):
        history.verify_sources([item], historical=True)


def test_corrupted_snapshot_rejected_even_when_live_source_exists():
    real_record = history.record

    def changed(path):
        result = real_record(path)
        result["sha256"] = "0" * 64
        return result

    with patch.object(history, "record", side_effect=changed):
        with pytest.raises(ValueError, match="snapshot changed"):
            history.source_record(history.ROOT / "src/CrowdSim/humanoid/runtime/social_popart_training.py")


def test_archived_source_cannot_admit_a_live_owner_plan():
    completed = resume.original.read(history.ROOT / "docs/style_improvement_20261008/shared_social_S3R_evaluation_v2/completion.json")
    with pytest.raises(ValueError, match="source changed"):
        owner.admit(Path(completed["plan"]["path"]))
    owner.admit(Path(completed["plan"]["path"]), historical=True)


@pytest.mark.parametrize("kind", ["owner", "retry"])
@pytest.mark.parametrize("mutation", ["empty", "missing", "duplicate", "reordered"])
def test_mandatory_source_inventory_cannot_be_removed_or_reordered(kind, mutation):
    paths = history.plan_inventory(kind)
    items = [history.source_record(path) for path in paths]
    if mutation == "empty":
        items.clear()
    elif mutation == "missing":
        items.pop()
    elif mutation == "duplicate":
        items.append(items[0])
    else:
        items.reverse()
    with pytest.raises(ValueError, match="complete ordered"):
        history.verify_sources(items, required_paths=paths, historical=True)


def test_real_completed_retry_retains_both_failed_and_successful_executions():
    path = history.ROOT / "docs/style_improvement_20261008/shared_social_S3R_navigation_retry_v1/completion.json"
    bundle, identity = resume.admit_completion(path)
    assert bundle["navigation"]["actual_control_steps"] == 6000
    assert bundle["navigation_CPU_audit"]["actor_controls"] == 24000
    assert identity["sha256"] == "7de1439ae51b94a9e7957c1d5239988fd201881a527c3788887e1ebd9aca5094"
    document = deepcopy(resume.original.read(path))
    document["steps"][0]["actual_exit_code"] = False
    original_read = resume.original.read
    with patch.object(resume.original, "read", side_effect=lambda p: document if Path(p) == path else original_read(p)):
        with pytest.raises(ValueError, match="integer-zero"):
            resume.admit_completion(path)
