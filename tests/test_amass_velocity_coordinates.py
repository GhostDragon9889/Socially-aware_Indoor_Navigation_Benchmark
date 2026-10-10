"""Independent SO(3) and real-bank checks for explicit expert velocity frames."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, SCHEMA


def make_bank(tmp_path, *, entry_frame=None, bank_frame=None, include_local=True, dtype=torch.float64):
    frame_count = 6
    angles = np.array([.4, .6, -.2]) + np.arange(frame_count*23*3).reshape(frame_count, 23, 3)*.001
    rotations = Rotation.from_rotvec(angles.reshape(-1, 3))
    child_velocity = np.arange(frame_count*69).reshape(frame_count, 23, 3)/100 + [.3, -.7, 1.1]
    parent_velocity = np.einsum("nij,nj->ni", rotations.as_matrix(), child_velocity.reshape(-1, 3)).reshape(frame_count, 69)
    local = np.concatenate((np.broadcast_to([0., 0., 0., 1.], (frame_count, 1, 4)),
                            rotations.as_quat().reshape(frame_count, 23, 4)), axis=1)
    stored_velocity = child_velocity.reshape(frame_count, 69) if bank_frame == "child_joint" else parent_velocity
    data = {"gts": torch.zeros(frame_count, 24, 3, dtype=dtype),
        "grs": torch.tensor(np.broadcast_to([0., 0., 0., 1.], (frame_count, 24, 4)).copy(), dtype=dtype),
        "gvs": torch.arange(frame_count*72, dtype=dtype).reshape(frame_count, 24, 3),
        "gavs": torch.arange(frame_count*72, dtype=dtype).reshape(frame_count, 24, 3)/10,
        "dps": torch.tensor(angles.reshape(frame_count, 69), dtype=dtype),
        "dvs": torch.tensor(stored_velocity, dtype=dtype),
        "motion_files": ["train/synthetic.motion"], "motion_num_frames": torch.tensor([frame_count]),
        "motion_dt": torch.tensor([1/30]), "length_starts": torch.tensor([0])}
    if include_local:
        data["lrs"] = torch.tensor(local, dtype=dtype)
    if bank_frame is not None:
        data["dof_velocity_frame"] = bank_frame
    bank = tmp_path/"bank.pt"
    torch.save(data, bank)
    entry = {"path": str(bank), "bytes": bank.stat().st_size,
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(), "clips": [{"motion_id": 0, "frames": frame_count,
        "source_path": "train/synthetic.motion", "dt": 1/30, "diffusion_eligible": False}]}
    if entry_frame is not None:
        entry["dof_velocity_frame"] = entry_frame
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"schema": SCHEMA, "splits": {"train": entry}}))
    return manifest, data, torch.tensor(child_velocity.reshape(frame_count, 69), dtype=dtype)


def test_default_preserves_all_legacy_state_tensors_bitwise_and_records_assumption(tmp_path):
    manifest, data, _ = make_bank(tmp_path)
    corpus = AMASSCorpus(manifest)
    indices = torch.tensor([0, 2, 4])
    state = corpus.state_at(indices)
    for field, bank_field in (("rigid_body_pos", "gts"), ("rigid_body_rot", "grs"),
            ("rigid_body_vel", "gvs"), ("rigid_body_ang_vel", "gavs"), ("dof_pos", "dps"), ("dof_vel", "dvs")):
        assert torch.equal(getattr(state, field), data[bank_field][indices])
    metadata = corpus.provenance()["dof_velocity"]
    assert metadata["requested_mode"] == "legacy_stored"
    assert metadata["stored_frame"] == metadata["output_frame"] == "parent_joint"
    assert metadata["stored_frame_assumed"] is True
    assert metadata["conversion"] == "none"


@pytest.mark.parametrize("indices", [torch.tensor(1), torch.tensor([0, 3, 5]), torch.tensor([[0, 2], [1, 3]])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_local_rotation_matches_independent_matrix_so3_and_keeps_leading_dimensions(tmp_path, indices, dtype):
    manifest, data, expected = make_bank(tmp_path, dtype=dtype)
    corpus = AMASSCorpus(manifest, dof_velocity_frame="child_joint")
    state = corpus.state_at(indices)
    assert state.dof_vel.shape == (*indices.shape, 69)
    assert state.dof_vel.dtype == dtype
    assert state.dof_vel.device.type == "cpu"
    torch.testing.assert_close(state.dof_vel, expected[indices], atol=2e-6, rtol=2e-6)
    for field, bank_field in (("dof_pos", "dps"), ("rigid_body_vel", "gvs"), ("rigid_body_ang_vel", "gavs")):
        assert torch.equal(getattr(state, field), data[bank_field][indices])
    assert torch.equal(corpus.data["dvs"], data["dvs"])
    assert corpus.provenance()["dof_velocity"]["output_frame"] == "child_joint"


def test_native_child_velocity_is_not_principal_expmap_component_derivative(tmp_path):
    manifest, data, expected = make_bank(tmp_path)
    corpus = AMASSCorpus(manifest, dof_velocity_frame="child_joint")
    first = Rotation.from_rotvec(data["dps"][0, :3].numpy())
    native = expected[0, :3].numpy()
    next_rotation = first*Rotation.from_rotvec(native/30)
    public_pacer_velocity = (first.inv()*next_rotation).as_rotvec()*30
    torch.testing.assert_close(corpus.state_at(torch.tensor(0)).dof_vel[:3],
        torch.tensor(public_pacer_velocity), atol=1e-12, rtol=0)
    derivative = (next_rotation.as_rotvec()-first.as_rotvec())*30
    assert np.linalg.norm(derivative-native) > .3


def test_declared_child_bank_is_not_converted_twice_and_needs_no_lrs(tmp_path):
    manifest, data, _ = make_bank(tmp_path, entry_frame="child_joint", bank_frame="child_joint", include_local=False)
    explicit = AMASSCorpus(manifest, dof_velocity_frame="child_joint")
    legacy = AMASSCorpus(manifest)
    for corpus in (explicit, legacy):
        assert torch.equal(corpus.state_at(torch.tensor([1, 2])).dof_vel, data["dvs"][[1, 2]])
        assert corpus.provenance()["dof_velocity"]["conversion"] == "none"
        assert corpus.provenance()["dof_velocity"]["stored_frame_assumed"] is False


def test_missing_lrs_fails_only_when_parent_to_child_conversion_explicitly_requested(tmp_path):
    manifest, data, _ = make_bank(tmp_path, include_local=False)
    assert torch.equal(AMASSCorpus(manifest).state_at(torch.tensor(0)).dof_vel, data["dvs"][0])
    with pytest.raises(ValueError, match="actual local XYZW"):
        AMASSCorpus(manifest, dof_velocity_frame="child_joint")


@pytest.mark.parametrize("source", ["entry", "bank"])
def test_explicit_null_storage_declaration_is_not_treated_as_missing(tmp_path, source):
    manifest, data, _ = make_bank(tmp_path)
    document = json.loads(manifest.read_text())
    if source == "entry":
        document["splits"]["train"]["dof_velocity_frame"] = None
    else:
        data["dof_velocity_frame"] = None
        bank = Path(document["splits"]["train"]["path"])
        torch.save(data, bank)
        document["splits"]["train"]["bytes"] = bank.stat().st_size
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Unknown"):
        AMASSCorpus(manifest)


@pytest.mark.parametrize("entry_frame,bank_frame", [
    ("parent_joint", "child_joint"), ("child_joint", "parent_joint"),
    ("unknown", None), (None, "unknown"), ("legacy_stored", None), (None, "legacy_stored"),
])
def test_contradictory_or_unknown_storage_declarations_fail_closed(tmp_path, entry_frame, bank_frame):
    manifest, _, _ = make_bank(tmp_path, entry_frame=entry_frame, bank_frame=bank_frame)
    with pytest.raises(ValueError, match="disagree|Unknown"):
        AMASSCorpus(manifest)


@pytest.mark.parametrize("choice", [None, True, "parent_joint", "unknown", []])
def test_unknown_output_mode_is_rejected(tmp_path, choice):
    manifest, _, _ = make_bank(tmp_path)
    with pytest.raises(ValueError, match="legacy_stored or child_joint"):
        AMASSCorpus(manifest, dof_velocity_frame=choice)


@pytest.mark.parametrize("failure", ["nan", "zero", "doubled"])
def test_nonunit_or_nonfinite_local_rotations_rejected_on_selected_conversion(tmp_path, failure):
    manifest, _, _ = make_bank(tmp_path)
    corpus = AMASSCorpus(manifest, dof_velocity_frame="child_joint")
    local = corpus.data["lrs"].clone()
    local[1, 3] = float("nan") if failure == "nan" else 0 if failure == "zero" else local[1, 3]*2
    corpus.data["lrs"] = local
    with pytest.raises(ValueError, match="unit finite local"):
        corpus.state_at(torch.tensor([1]))


@pytest.mark.needs_lfs
def test_real_v6_conversion_matches_independent_scipy_and_legacy_stored_remains_bitwise():
    path = Path(__file__).resolve().parents[1]/"data/diffusion_amp/amass_style_2848/manifest_labeled_v6.json"
    if not path.exists():
        pytest.skip("Local licensed curated V6 bank is not supplied")
    legacy = AMASSCorpus(path)
    converted = AMASSCorpus(path, dof_velocity_frame="child_joint")
    starts = legacy.data["length_starts"][legacy.ids[:16]]
    indices = starts[:, None]+torch.arange(10)
    old, new = legacy.state_at(indices), converted.state_at(indices)
    assert torch.equal(old.dof_vel, legacy.data["dvs"][indices])
    rotation = Rotation.from_quat(legacy.data["lrs"][indices][..., 1:, :].numpy().reshape(-1, 4))
    expected = rotation.inv().apply(old.dof_vel.numpy().reshape(-1, 3)).reshape(new.dof_vel.shape)
    np.testing.assert_allclose(new.dof_vel.numpy(), expected, rtol=1e-5, atol=2e-6)
    assert float((old.dof_vel-new.dof_vel).abs().max()) > .1
    assert torch.equal(old.rigid_body_ang_vel, new.rigid_body_ang_vel)


@pytest.mark.needs_lfs
def test_real_v7_child_declaration_prevents_double_conversion():
    path = Path(__file__).resolve().parents[1]/"data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
    if not path.exists():
        pytest.skip("Local licensed curated V7 bank is not supplied")
    corpus = AMASSCorpus(path, dof_velocity_frame="child_joint")
    indices = corpus.data["length_starts"][corpus.ids[:16]]+7
    assert torch.equal(corpus.state_at(indices).dof_vel, corpus.data["dvs"][indices])
    provenance = corpus.provenance()["dof_velocity"]
    assert provenance["storage_declarations"] == {"entry": "child_joint", "bank": "child_joint"}
    assert provenance["stored_frame"] == provenance["output_frame"] == "child_joint"
    assert provenance["conversion"] == "none"
