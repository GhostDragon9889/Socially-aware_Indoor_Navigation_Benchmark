"""Real Torch mmap I/O path regression, independent of licensed AMASS files.

The tiny tensor packages test file selection, not human motion or physics.
"""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, SCHEMA


def write_motion_package(path, pose):
    frames = 32
    rotations = torch.zeros(frames, 24, 4)
    rotations[..., 3] = 1
    data = {"gts": torch.zeros(frames, 24, 3), "grs": rotations,
            "gvs": torch.zeros(frames, 24, 3), "gavs": torch.zeros(frames, 24, 3),
            "dps": torch.full((frames, 69), pose), "dvs": torch.zeros(frames, 69),
            "motion_num_frames": torch.tensor([frames]), "length_starts": torch.tensor([0])}
    torch.save(data, path)
    return {"path": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clips": [{"motion_id": 0, "diffusion_eligible": False}]}


@pytest.mark.parametrize("absolute", [False, True])
def test_corpus_mmap_selects_manifest_package_when_cwd_contains_a_different_same_name(tmp_path, monkeypatch, absolute):
    manifest_dir = tmp_path / "manifest-bundle"
    unrelated_cwd = tmp_path / "cwd"
    manifest_dir.mkdir()
    unrelated_cwd.mkdir()
    package = manifest_dir / "bank.pt"
    entry = write_motion_package(package, .125)
    decoy = unrelated_cwd / "bank.pt"
    write_motion_package(decoy, .75)
    assert decoy.stat().st_size == package.stat().st_size, "Decoy must pass the old byte-size gate"
    if absolute:
        entry["path"] = str(package)
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(json.dumps({"schema": SCHEMA, "splits": {"train": entry}}))
    monkeypatch.chdir(unrelated_cwd)
    corpus = AMASSCorpus(manifest, seed=42)
    current, history, next_position, _, _ = corpus.sample_reference(8)
    assert current.dof_pos.shape == (8, 69)
    assert history.dof_pos.shape == (80, 69)
    assert torch.equal(next_position, torch.full((8, 69), .125))
    assert corpus.data["dps"].device.type == "cpu"


def test_missing_manifest_relative_file_does_not_fall_back_to_cwd_package(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifest-bundle"
    manifest_dir.mkdir()
    entry = write_motion_package(tmp_path / "bank.pt", .75)
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(json.dumps({"schema": SCHEMA, "splits": {"train": entry}}))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="manifest-bundle.*bank.pt"):
        AMASSCorpus(manifest)
