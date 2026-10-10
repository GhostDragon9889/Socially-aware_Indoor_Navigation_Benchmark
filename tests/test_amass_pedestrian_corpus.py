"""Real mmap AMASS windows, checkpoint RNG continuation and expert PD mapping.

These tests require the locally supplied AMASS packages. They neither construct
replacement motion tensors nor import IsaacSim/IsaacLab. License restricted
packages are intentionally absent from public CI and distribution artifacts.
"""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, SCHEMA
from CrowdSim.humanoid.amp import action_to_pd, build_amp_observation
from CrowdSim.humanoid.amp.observations import PACER_DOF_NAMES, PacerSMPLMapping
from protomotions.utils.rotations import quat_rotate


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/diffusion_amp/amass_manifest.json"
SPLITS = ("train", "validation", "test")
pytestmark = pytest.mark.needs_lfs


@pytest.fixture(scope="module")
def corpora():
    if not MANIFEST.is_file():
        pytest.skip("Prepare the supplied reconstructed AMASS manifest first")
    manifest = json.loads(MANIFEST.read_text())
    missing = [entry["path"] for entry in manifest["splits"].values()
               if not Path(entry["path"]).is_file()]
    if missing:
        pytest.skip(f"Real AMASS packages are unavailable: {missing}")
    return {split: AMASSCorpus(MANIFEST, split, seed=173) for split in SPLITS}


def _assert_planar_matches_bank(corpus, actual, indices):
    """Check channels against real positions/velocities and rotated forward axes."""
    torch.testing.assert_close(actual[..., :2], corpus.data["gts"][indices, 0, :2], rtol=0, atol=0)
    torch.testing.assert_close(actual[..., 2:4], corpus.data["gvs"][indices, 0, :2], rtol=0, atol=0)
    rotations = corpus.data["grs"][indices, 0]
    forward = torch.zeros_like(rotations[..., :3])
    forward[..., 0] = 1
    rotated = quat_rotate(rotations.reshape(-1, 4), forward.reshape(-1, 3), w_last=True)
    expected_yaw = torch.atan2(rotated[:, 1], rotated[:, 0]).reshape(indices.shape)
    torch.testing.assert_close(actual[..., 4], expected_yaw, rtol=0, atol=1e-5)
    assert torch.isfinite(actual).all()


def _assert_state_matches_bank(state, data, indices):
    for name, key in (("rigid_body_pos", "gts"), ("rigid_body_rot", "grs"),
                      ("rigid_body_vel", "gvs"), ("rigid_body_ang_vel", "gavs"),
                      ("dof_pos", "dps"), ("dof_vel", "dvs")):
        torch.testing.assert_close(getattr(state, name), data[key][indices], rtol=0, atol=0)
    for name, key in (("root_pos", "gts"), ("root_rot", "grs"),
                      ("root_vel", "gvs"), ("root_ang_vel", "gavs")):
        torch.testing.assert_close(getattr(state, name), data[key][indices, 0], rtol=0, atol=0)


def test_real_split_manifest_matches_all_selected_clip_metadata(corpora):
    manifest = corpora["train"].manifest
    assert manifest["schema"] == SCHEMA
    assert manifest["author_clip_list"] is False
    paths = []
    for split, corpus in corpora.items():
        entry, data = corpus.entry, corpus.data
        assert corpus.device.type == "cpu"
        assert Path(entry["path"]).stat().st_size == entry["bytes"]
        assert len(entry["sha256"]) == 64  # Large banks are not rehashed in every test run.
        assert len(corpus.ids) == entry["clip_count"] == len(entry["clips"])
        assert len(corpus.diffusion_ids) == entry["diffusion_clip_count"]
        assert len(set(corpus.ids.tolist())) == len(corpus.ids)
        frames = data["dps"].shape[0]
        assert data["gts"].shape == data["gvs"].shape == data["gavs"].shape == (frames, 24, 3)
        assert data["grs"].shape == (frames, 24, 4)
        assert data["dps"].shape == data["dvs"].shape == (frames, 69)
        assert all(data[key].device.type == "cpu" for key in ("gts", "grs", "dps"))
        starts = data["length_starts"]
        assert (starts[1:] > starts[:-1]).all()
        split_paths = set()
        for record in entry["clips"]:
            index = record["motion_id"]
            assert data["motion_files"][index] == record["source_path"]
            assert int(data["motion_num_frames"][index]) == record["frames"]
            assert float(data["motion_dt"][index]) == pytest.approx(record["dt"], abs=1e-8)
            assert record["dt"] == pytest.approx(1 / 30, abs=1e-5)
            assert record["diffusion_eligible"] == (record["frames"] >= 247)
            assert int(starts[index]) + record["frames"] <= frames
            split_paths.add(record["source_path"])
        assert len(split_paths) == entry["clip_count"], split
        paths.append(split_paths)
        provenance = corpus.provenance()
        assert provenance["author_clip_list"] is False
        assert provenance["source_sha256"] == entry["sha256"]
        assert provenance["split"] == split
    assert all(not left.intersection(right) for i, left in enumerate(paths) for right in paths[i + 1:])


@pytest.mark.parametrize("split", SPLITS)
def test_real_diffusion_all_31_past_and_52_future_points_stay_in_clip(corpora, split):
    corpus = corpora[split]
    corpus.generator.manual_seed(173)
    batch = corpus.diffusion_batch(2048)
    assert batch["history"].shape == (2048, 31, 5)
    assert batch["future"].shape == (2048, 52, 5)
    assert batch["states"].shape == (2048, 5)
    ids, frame = batch["motion_ids"], batch["frame_indices"]
    assert set(ids.tolist()).issubset(set(corpus.diffusion_ids.tolist()))
    # The deterministic real draw covers almost all eligible clips, not one mock clip.
    assert ids.unique().numel() >= .9 * len(corpus.diffusion_ids)
    starts = corpus.data["length_starts"][ids]
    counts = corpus.data["motion_num_frames"][ids]
    history_frames = frame[:, None] - torch.arange(30, -1, -1) * 3
    future_frames = frame[:, None] + torch.arange(1, 53) * 3
    all_frames = torch.cat((history_frames, future_frames), 1)
    assert (all_frames >= 0).all()
    assert (all_frames < counts[:, None]).all()
    assert torch.equal(history_frames[:, -1], frame)
    assert (future_frames[:, 0] - frame == 3).all()
    assert (future_frames[:, -1] - frame == 156).all()
    global_frames = starts[:, None] + all_frames
    # Independent global-boundary lookup catches both an off-by-one and adjacent-clip leakage.
    owning_clips = torch.bucketize(global_frames, corpus.data["length_starts"], right=True) - 1
    assert torch.equal(owning_clips, ids[:, None].expand_as(owning_clips))
    _assert_planar_matches_bank(corpus, batch["history"], starts[:, None] + history_frames)
    _assert_planar_matches_bank(corpus, batch["future"], starts[:, None] + future_frames)
    assert torch.equal(batch["states"], batch["history"][:, -1])


@pytest.mark.parametrize("split", SPLITS)
def test_real_expert_history_is_t_through_t_minus_9_and_action_is_t_plus_1(corpora, split):
    corpus = corpora[split]
    corpus.generator.manual_seed(719)
    current, history, next_position, ids, frame = corpus.sample_reference(512)
    starts = corpus.data["length_starts"][ids]
    counts = corpus.data["motion_num_frames"][ids]
    assert (frame >= 9).all()
    assert (frame + 1 < counts).all()
    indices = starts + frame
    history_indices = indices[:, None] - torch.arange(10)
    assert (history_indices >= starts[:, None]).all()
    _assert_state_matches_bank(current, corpus.data, indices)
    _assert_state_matches_bank(history, corpus.data, history_indices.flatten())
    torch.testing.assert_close(next_position, corpus.data["dps"][indices + 1], rtol=0, atol=0)
    assert history.rigid_body_pos.shape == (5120, 24, 3)
    features = build_amp_observation(history).reshape(512, 10, 195)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[:, 0], build_amp_observation(current), rtol=0, atol=0)


@pytest.mark.parametrize("split", SPLITS)
def test_real_corpus_rng_checkpoint_continues_mixed_sampling_exactly(corpora, split, tmp_path):
    corpus = corpora[split]
    corpus.generator.manual_seed(337)
    corpus.diffusion_batch(13)
    corpus.sample_reference(7)
    checkpoint = tmp_path / f"{split}_rng.pt"
    torch.save({"corpus_rng": corpus.generator.get_state()}, checkpoint)
    expected_diffusion = corpus.diffusion_batch(97)
    expected_reference = corpus.sample_reference(53)
    expected_next_rng = corpus.generator.get_state().clone()
    corpus.generator.manual_seed(999)  # A fresh process's seed must not change continuation.
    corpus.generator.set_state(torch.load(checkpoint, weights_only=True)["corpus_rng"])
    actual_diffusion = corpus.diffusion_batch(97)
    actual_reference = corpus.sample_reference(53)
    for key, expected in expected_diffusion.items():
        torch.testing.assert_close(actual_diffusion[key], expected, rtol=0, atol=0)
    for actual, expected in zip(actual_reference, expected_reference):
        if isinstance(actual, SimpleNamespace):
            for key in vars(actual):
                torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(corpus.generator.get_state(), expected_next_rng)


def _production_expert_batch():
    """Execute the actual small method without importing the simulator module.

    The function body is compiled verbatim from the working tree. This avoids a
    second implementation of the inverse contract and requires no Kit startup.
    """
    path = ROOT / "CrowdSim/humanoid/amp_world.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    world = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AMPWorld")
    method = next(node for node in world.body if isinstance(node, ast.FunctionDef) and node.name == "expert_batch")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch, "action_to_pd": action_to_pd,
                 "build_amp_observation": build_amp_observation}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["expert_batch"]


def test_real_next_frame_expert_inverse_round_trips_physical_dof_permutation(corpora):
    corpus = corpora["train"]
    xml = ET.parse(ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml").getroot()
    joints = [joint for joint in xml.findall(".//worldbody//joint") if joint.get("type") == "hinge"]
    names = tuple(joint.attrib["name"] for joint in joints)
    assert names == PACER_DOF_NAMES
    limits = torch.tensor([[math.radians(float(value)) for value in joint.attrib["range"].split()]
                           for joint in joints])
    lower, upper = limits.unbind(1)
    world = SimpleNamespace(corpus=corpus, history_frames=10, device=torch.device("cpu"),
                            lower=lower, upper=upper, mapping=PacerSMPLMapping(dof_names=names))
    corpus.generator.manual_seed(917)
    rng = corpus.generator.get_state()
    _, _, next_position, _, _ = corpus.sample_reference(128)
    corpus.generator.set_state(rng)
    features, canonical = _production_expert_batch()(world, 128)
    assert features.shape == (128, 10, 195)
    assert canonical.shape == (128, 69)
    assert torch.isfinite(features).all() and torch.isfinite(canonical).all()
    assert (canonical.abs() <= 1).all()
    # Independent public contract: symmetric pi cap; two knee_y scales are five.
    cap = torch.full((69,), math.pi)
    for side in ("L", "R"):
        cap[names.index(f"{side}_Knee_y")] = 5
    attainable = next_position.clamp(-cap, cap)
    frozen = [i for i, name in enumerate(names)
              if name.rsplit("_", 1)[0] in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}]
    attainable[:, frozen] = 0
    assert canonical[:, frozen].count_nonzero() == 0
    torch.testing.assert_close(action_to_pd(canonical, lower, upper), attainable, rtol=1e-6, atol=1e-6)
    permutation = torch.randperm(69, generator=torch.Generator().manual_seed(241))
    physical_names = tuple(names[index] for index in permutation)
    physical = action_to_pd(canonical, lower[permutation], upper[permutation], dof_names=physical_names)
    torch.testing.assert_close(physical, attainable[:, permutation], rtol=1e-6, atol=1e-6)
    inverse = PacerSMPLMapping(dof_names=physical_names).dof_indices
    torch.testing.assert_close(physical[:, inverse], attainable, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(physical, attainable), "Permutation fixture must exercise a real ordering change"
