"""Teacher and deployment-schema gates; no synthetic scene quality claims."""
from pathlib import Path

import pytest
import torch

from CrowdSim.humanoid.application.settings import TrainingConfig
from CrowdSim.humanoid.training.pacer_finetune import AnchoredPacerTrainer
from CrowdSim.humanoid.amp.controller import NO_SHAPE_SHA256, PacerAMPController

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"


def make_trainer():
    if not PUBLIC.is_file():
        pytest.skip("Verified public PACER resource is unavailable")
    config = TrainingConfig(discriminator_input="state", amp_normalization="per_history_coordinate",
                            warm_start_discriminator=True, effective_action_likelihood=True)
    trainer = AnchoredPacerTrainer(config, provenance={"finetune": {"anchor_coefficient": 1.}})
    limits = torch.full((69,), 1.)
    trainer.warm_start_pacer(PUBLIC, -limits, limits)
    trainer.bind_teacher(PUBLIC, -limits, limits)
    return trainer


def test_teacher_kl_is_zero_at_initialization_and_masks_inactive_dofs():
    trainer = make_trainer()
    own, task = torch.randn(3, 357), torch.randn(3, 1044)
    distribution, _ = trainer.model.policy(own, task)
    assert trainer.policy_regularization(distribution, own, task).item() == 0
    with torch.no_grad():
        trainer.model.actor[4].bias[trainer.action_mask == 0] += .02
    distribution, _ = trainer.model.policy(own, task)
    assert trainer.policy_regularization(distribution, own, task).item() == 0
    with torch.no_grad():
        trainer.model.actor[4].bias[trainer.action_mask == 1] += .01
    distribution, _ = trainer.model.policy(own, task)
    loss = trainer.policy_regularization(distribution, own, task)
    assert loss.item() > 0
    loss.backward()
    assert trainer.model.actor[4].bias.grad[trainer.action_mask == 1].abs().sum() > 0
    assert all(p.grad is None for p in trainer.teacher.parameters())
    assert trainer.freeze_normalizers


def test_untrained_checkpoint_cannot_be_exported_as_fine_tuned(tmp_path):
    trainer = make_trainer()
    with pytest.raises(ValueError, match="untrained"):
        trainer.export(tmp_path / "forbidden.pth", coverage=set(), manifest_sha256="0"*64)
    assert not (tmp_path / "forbidden.pth").exists()


def test_invalid_local_provenance_is_rejected_even_with_correct_file_digest(tmp_path):
    import hashlib
    state = torch.load(PUBLIC, weights_only=True, map_location="cpu")
    state["local_finetune"] = {"schema": "crowdsim.pacer_amass_finetune.v1", "parent_sha256": NO_SHAPE_SHA256,
        "train_clip_count": 2848, "normalizers_frozen": True, "all_train_clips_seen": False,
        "updates": 1, "physical_transitions": 1, "amp_sample_offsets_s": [i*.5 for i in range(10)]}
    path = tmp_path / "invalid.pth"
    torch.save(state, path)
    with pytest.raises(ValueError, match="provenance"):
        PacerAMPController.from_checkpoint(path, -torch.ones(69), torch.ones(69),
                                           expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
