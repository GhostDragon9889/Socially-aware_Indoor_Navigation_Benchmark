"""The historical upstream pack is preserved and explicitly excluded from runnable releases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from protomotions.repository import repository_root
from protomotions.utils.pretrained_compat import (
    archived_upstream_reference,
    load_upstream_reference_registry,
    require_supported_checkpoint_config,
)


def test_only_two_branch_specific_upstream_models_are_archived():
    root = repository_root()
    registry = load_upstream_reference_registry()
    assert len(registry["entries"]) == 2
    for row in registry["entries"]:
        assert row["role"] == "archived_upstream_reference"
        assert row["source_context"]["code_context_only"] is True
        assert row["source_context"]["exact_checkpoint_release_commit_verified"] is False
        path = root / row["directory"] / "resolved_configs.pt"
        assert archived_upstream_reference(path) is row or archived_upstream_reference(path) == row
        with pytest.raises(RuntimeError, match="Archived upstream reference"):
            require_supported_checkpoint_config(path)
    for path in ("motion_tracker/smpl", "masked_mimic/smpl", "gpc_prior/soma_bones", "paper_references/trace_public/model.pth"):
        assert archived_upstream_reference(root / "data/pretrained_models" / path) is None


def test_archived_plaintext_sidecars_are_preserved_exactly():
    root = repository_root()
    for row in load_upstream_reference_registry()["entries"]:
        for item in row["files"]:
            if not item["path"].endswith((".py", ".yaml")):
                continue
            value = (root / item["path"]).read_bytes()
            assert len(value) == item["bytes"]
            assert hashlib.sha256(value).hexdigest() == item["sha256"]


@pytest.mark.needs_lfs
def test_archived_binary_models_are_preserved_exactly():
    root = repository_root()
    for row in load_upstream_reference_registry()["entries"]:
        for item in row["files"]:
            if not item["path"].endswith((".ckpt", ".pt")):
                continue
            value = (root / item["path"]).read_bytes()
            assert len(value) == item["bytes"]
            assert hashlib.sha256(value).hexdigest() == item["sha256"]


def test_reference_guard_precedes_pickle_loader(monkeypatch):
    from protomotions.utils import config_utils
    path = repository_root() / load_upstream_reference_registry()["entries"][0]["directory"] / "last.ckpt"
    def unexpected_load(*args, **kwargs):
        pytest.fail("Archived pickle must be rejected before torch.load")
    monkeypatch.setattr(config_utils.torch, "load", unexpected_load)
    with pytest.raises(RuntimeError, match="Archived upstream reference"):
        config_utils.load_resolved_configs_from_checkpoint(path)


@pytest.mark.parametrize("directory", ["/external/assets", "../escape"])
def test_registry_cannot_classify_files_outside_project(tmp_path, directory):
    dest = tmp_path / "data/pretrained_models/upstream_reference_registry.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps({"schema": "social_navigation.upstream_reference_registry.v1", "entries": [
        {"directory": directory, "status": "unsupported_in_current_runtime"}
    ]}))
    with pytest.raises(RuntimeError, match="project-relative"):
        load_upstream_reference_registry(tmp_path)
