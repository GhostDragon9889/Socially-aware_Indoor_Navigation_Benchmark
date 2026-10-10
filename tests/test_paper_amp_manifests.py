"""Real tensor package metadata, dataset leakage and strict provenance gates."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from CrowdSim.humanoid.training.manifests import ManifestError, validate_training_manifest


def _write_fixture(tmp_path):
    splits = {}
    for split in ("train", "validation", "test"):
        package = tmp_path / f"{split}.pt"
        source = f"AMASS/{split}/Walking_1.motion"
        data = dict(gts=torch.zeros(2, 24, 3), grs=torch.tensor([0., 0., 0., 1.]).repeat(2, 24, 1),
            dps=torch.zeros(2, 69), dvs=torch.zeros(2, 69),
            length_starts=torch.tensor([0]), motion_num_frames=torch.tensor([2]),
            motion_dt=torch.tensor([1 / 30]), motion_files=(source,))
        torch.save(data, package)
        splits[split] = {"path": package.name, "bytes": package.stat().st_size,
            "sha256": hashlib.sha256(package.read_bytes()).hexdigest(), "clip_count": 1,
            "clips": [{"motion_id": 0, "source_path": source, "frames": 2, "dt": 1 / 30}]}
    path = tmp_path / "manifest.json"
    document = dict(schema="navisaaclab.amass_reconstruction.v1", author_clip_list=False,
        selection={"method": "explicit supplied fixture clips, not an author corpus"}, splits=splits)
    path.write_text(json.dumps(document))
    return path, document


def test_actual_package_integrity_and_explicit_heldout_splits(tmp_path):
    path, _ = _write_fixture(tmp_path)
    report = validate_training_manifest(path, verify_package_hashes=True)
    assert report["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["heldout_count"] == 2
    assert all(resource["sha256_recomputed"] for resource in report["resources"])
    assert report["actual_author_experiment_reproduced"] is False


@pytest.mark.parametrize("problem", ["unknown_id", "wrong_name", "wrong_frames", "wrong_dt", "missing_package", "changed_hash", "duplicate"])
def test_corrupted_or_missing_actual_corpus_records_fail(tmp_path, problem):
    path, document = _write_fixture(tmp_path)
    train = document["splits"]["train"]
    if problem == "unknown_id":
        train["clips"][0]["motion_id"] = 1
    elif problem == "wrong_name":
        train["clips"][0]["source_path"] = "invented_author_id.motion"
    elif problem == "wrong_frames":
        train["clips"][0]["frames"] = 2848
    elif problem == "wrong_dt":
        train["clips"][0]["dt"] = .1
    elif problem == "missing_package":
        train["path"] = "missing.pt"
    elif problem == "changed_hash":
        train["sha256"] = "0" * 64
    else:
        train["clips"] *= 2
        train["clip_count"] = 2
    path.write_text(json.dumps(document))
    with pytest.raises(ManifestError):
        validate_training_manifest(path, verify_package_hashes=True)


def test_original_motion_source_leakage_fails_even_with_valid_package_hashes(tmp_path):
    path, document = _write_fixture(tmp_path)
    validation = document["splits"]["validation"]
    package = tmp_path / validation["path"]
    data = torch.load(package, weights_only=True)
    train_source = document["splits"]["train"]["clips"][0]["source_path"]
    data["motion_files"] = (train_source,)
    torch.save(data, package)
    validation["clips"][0]["source_path"] = train_source
    validation["bytes"] = package.stat().st_size
    validation["sha256"] = hashlib.sha256(package.read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(ManifestError, match="leakage"):
        validate_training_manifest(path, verify_package_hashes=True)


def test_strict_mode_rejects_user_supplied_derived_clips_instead_of_filling_2848(tmp_path):
    path, _ = _write_fixture(tmp_path)
    with pytest.raises(ManifestError, match="actual author 2,848"):
        validate_training_manifest(path, mode="strict_author_experiment")


@pytest.mark.needs_lfs
def test_actual_root_derived_corpus_contract_with_truthful_counts():
    path = Path(__file__).resolve().parents[1] / "data/diffusion_amp/amass_manifest.json"
    if not path.is_file():
        pytest.skip("The supplied reconstructed AMASS corpus has not been prepared")
    report = validate_training_manifest(path)
    assert report["counts"] == {"train": 319, "validation": 22, "test": 34}
    assert report["author_clip_list"] is False
    assert not any(resource["sha256_recomputed"] for resource in report["resources"])
    assert report["heldout_count"] == 56
