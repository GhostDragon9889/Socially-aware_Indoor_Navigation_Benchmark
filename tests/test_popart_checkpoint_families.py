"""Full actor admission across actual v1 weights and current v2 CPU fixtures."""
from copy import deepcopy
import hashlib
import uuid

import pytest
import torch

from CrowdSim.humanoid.runtime import social_popart_training as training
from CrowdSim.humanoid.runtime import popart_checkpoint_family as family
from CrowdSim.humanoid.runtime.social_popart_inference import FrozenSocialPopArtCheckpointInference


@pytest.fixture(scope="module")
def current_checkpoint(tmp_path_factory):
    torch.set_num_threads(4)
    module = training.load_trainer("_family_test_" + uuid.uuid4().hex)
    config = module.SocialTrainingConfig(actor_count=4, grid_center="root", grid_heading="root",
        height_reference="static_center_ground_z", include_ego=False, update_epochs=1,
        minibatch_size=4, discriminator_updates_per_update=1,
        discriminator_batch_size=4, discriminator_replay_capacity=4)
    trainer = module.SocialAMPTrainer.prepare(config, device="cpu")
    self_obs, task = torch.zeros(1, 4, 357), torch.zeros(1, 4, 12308)
    sampled = {key: value[None] for key, value in trainer.act(self_obs[0], task[0]).items()}
    rollout = dict(self_obs=self_obs, task_obs=task, amp_history=torch.zeros(1, 4, 10, 195),
        **sampled, rewards=torch.full((1, 4), .5), next_values=sampled["values"].clone(),
        terminated=torch.zeros(1, 4, dtype=torch.bool), truncated=torch.zeros(1, 4, dtype=torch.bool))
    trainer.update(rollout, torch.zeros(4, 10, 195), torch.zeros(4, 69))
    path = tmp_path_factory.mktemp("popart_family") / "step_000001.pt"
    trainer.save(path)
    path.chmod(0o444)
    del trainer
    return module, path






@pytest.mark.parametrize("mutation", ["unknown_format", "crossed_binding", "source_sha", "derivation", "extra_key"])
def test_historical_metadata_tampering_never_admits_current_optimizer_resume(current_checkpoint, mutation):
    module, current = current_checkpoint
    state = torch.load(current, weights_only=True, map_location="cpu", mmap=True)
    config = module.SocialAMPTrainer._admit_metadata(state)
    state["format"] = training.LEGACY_FORMAT
    state["social_training_contract"] = family.legacy_contract(module, config)
    validator = family.select_inference_definitions(module, state)
    validator.SocialAMPTrainer._admit_metadata(state)
    assert not hasattr(validator.SocialAMPTrainer, "load")
    with pytest.raises(ValueError, match="schema"):
        module.SocialAMPTrainer._admit_metadata(state)
    if mutation == "unknown_format":
        state["format"] += ".unknown"
    elif mutation == "crossed_binding":
        state["social_training_contract"]["value_normalization"] = training.describe_runtime()
    elif mutation == "source_sha":
        state["social_training_contract"]["sources"][0]["sha256"] = "0" * 64
    elif mutation == "derivation":
        state["social_training_contract"]["derivations"]["update"]["inverse_bytes_equal"] = False
    else:
        state["unexpected"] = True
    with pytest.raises(ValueError):
        family.select_inference_definitions(module, state).SocialAMPTrainer._admit_metadata(state)


def test_inference_rejects_a_different_inode_digest_before_loading(current_checkpoint):
    _, path = current_checkpoint
    with pytest.raises(ValueError, match="SHA256"):
        FrozenSocialPopArtCheckpointInference.from_checkpoint(path, expected_sha256="0" * 64)


def test_complete_binding_comparison_keeps_types_hashes_and_extra_fields():
    original = training.describe()
    for field, value in (("checkpoint_format", "unknown"), ("epsilon", "1e-5"), ("Adam_state_modified", 0)):
        changed = deepcopy(original)
        changed[field] = value
        with pytest.raises(ValueError):
            family.admit_binding(changed)
    with pytest.raises(ValueError):
        family.admit_binding({**original, "extra": True})
