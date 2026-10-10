"""Read-only CPU proof for a proposed PACER state-action initialization.

This changes only an in-memory fresh trainer and does no optimizer step,
simulation, physical rollout, production-file edit, or checkpoint save.
Synthetic inputs demonstrate function/gradient contracts, not motion quality.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
import yaml

from CrowdSim.humanoid.amp.controller import PacerAMPController
from CrowdSim.humanoid.application.settings import TrainingConfig
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer

torch.set_num_threads(4)
torch.set_num_interop_threads(4)
assert not torch.cuda.is_initialized()
profile = yaml.safe_load((ROOT / "configs/diffusion_amp/amp_style_mixed_state_action_terrain_protected.yaml").read_text())
config = TrainingConfig(**profile["training"])
checkpoint = ROOT / profile["initialization"]["checkpoint"]
reference = PacerAMPController.from_checkpoint(checkpoint, -torch.ones(69), torch.ones(69), device="cpu")
trainer = PaperAMPTrainer(config, device="cpu")
trainer.warm_start_pacer(checkpoint, -torch.ones(69), torch.ones(69))
assert trainer.model.discriminator[0].weight.shape == (1024, 2019)
assert trainer.amp_normalizer.mean.shape == (1950,)

def tensor_digest(values):
    digest = hashlib.sha256()
    for name, tensor in values.items():
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()

protected_modules = {name: tensor_digest(getattr(trainer.model, name).state_dict())
                     for name in ("actor", "critic", "task_encoder")}
protected_moments = {name: tensor_digest(getattr(trainer, name).state_dict())
                     for name in ("self_normalizer", "task_normalizer", "value_normalizer")}

# Proposed transfer, in memory only. The production bool remains False.
with torch.no_grad():
    first = trainer.model.discriminator[0]
    first.weight.zero_()
    first.weight[:, :1950].copy_(reference.network._disc_mlp[0].weight)
    first.bias.copy_(reference.network._disc_mlp[0].bias)
    trainer.model.discriminator[2].load_state_dict(reference.network._disc_mlp[2].state_dict(), strict=True)
    trainer.model.discriminator[4].load_state_dict(reference.network._disc_logits.state_dict(), strict=True)
    amp = reference.discriminator_normalizer
    trainer.amp_normalizer.mean.copy_(amp.running_mean)
    trainer.amp_normalizer.variance.copy_(amp.running_var)
    trainer.amp_normalizer.count.copy_(amp.count)
    trainer.amp_normalizer.count.clamp_max_(config.warm_start_normalizer_count_cap)

generator = torch.Generator().manual_seed(20261003)
mean = amp.running_mean.float()
scale = (amp.running_var.float() + 1e-5).sqrt()
raw = (mean + scale * torch.randn(64, 1950, generator=generator)).reshape(64, 10, 195)
zeros = torch.zeros(64, 69)
random_actions = (torch.rand(64, 69, generator=generator) * 2 - 1) * trainer.action_mask
extreme_actions = torch.ones(64, 69) * trainer.action_mask
with torch.no_grad():
    public_logits = reference.eval_discriminator(raw).squeeze(-1)
    normalized, _ = trainer._amp_inputs(raw, zeros)
    checks = {}
    for label, actions in (("zero", zeros), ("random", random_actions), ("extreme", extreme_actions)):
        logits = trainer.model.disc_logits(normalized, actions)
        checks[label] = float((logits - public_logits).abs().max())
        torch.testing.assert_close(logits, public_logits, atol=1e-5, rtol=1e-5)
    action_change_error = float((trainer.model.disc_logits(normalized, random_actions) -
                                trainer.model.disc_logits(normalized, zeros)).abs().max())
    assert action_change_error == 0
    normalized_error = float((normalized.flatten(1) - amp(raw.flatten(1))).abs().max())
    assert normalized_error == 0
    wrong_rms_logits = trainer.model.disc_logits(raw.clamp(-5, 5), zeros)
    wrong_order, _ = trainer._amp_inputs(raw.flip(1), zeros)
    wrong_order_logits = trainer.model.disc_logits(wrong_order, zeros)

# One backward pass through exactly the production loss equation, no step.
agent_h, agent_a = normalized[:32].detach(), random_actions[:32].detach()
expert_h, expert_a = normalized[32:].detach(), random_actions[32:].detach()
negative = trainer.model.disc_logits(agent_h, agent_a)
expert_features = torch.cat((expert_h.flatten(1), expert_a), -1).requires_grad_(True)
positive = trainer.model.discriminator(expert_features).squeeze(-1)
classification = .5 * (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive)) +
                       F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))
input_gradient = torch.autograd.grad(positive.sum(), expert_features, create_graph=True)[0]
penalty = input_gradient.square().sum(-1).mean()
logit_regularization = trainer.model.discriminator[-1].weight.square().sum()
l2 = sum(module.weight.square().sum() for module in trainer.model.discriminator if isinstance(module, torch.nn.Linear))
loss = config.discriminator_coefficient * (classification + config.discriminator_gradient_penalty * penalty +
       config.discriminator_logit_regularization * logit_regularization + config.discriminator_weight_decay * l2)
loss.backward()
action_gradient = trainer.model.discriminator[0].weight.grad[:, 1950:]
active_gradient = action_gradient[:, trainer.action_mask.bool()]
frozen_gradient = action_gradient[:, ~trainer.action_mask.bool()]
assert torch.isfinite(active_gradient).all() and torch.count_nonzero(active_gradient) > 0
assert torch.count_nonzero(frozen_gradient) == 0
assert torch.count_nonzero(trainer.model.discriminator[0].weight[:, 1950:]) == 0
assert all(tensor_digest(getattr(trainer.model, name).state_dict()) == digest
           for name, digest in protected_modules.items())
assert all(tensor_digest(getattr(trainer, name).state_dict()) == digest
           for name, digest in protected_moments.items())
trainer.terrain_statistics_policy.validate_checkpoint(trainer.task_normalizer, update_count=0, physical_transition_count=0)
assert not trainer.optimizer.state and not trainer.discriminator_optimizer.state
assert trainer.policy_replay.size == 0
assert not torch.cuda.is_initialized()
assert not any(name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules)

result = {
    "scope": "synthetic CPU forward and backward proof only; no optimizer/simulation/checkpoint writes",
    "checkpoint_sha256": reference.checkpoint_sha256,
    "source_shapes": {key: list(value.shape) for key, value in reference.network.state_dict().items()
                      if key.startswith("_disc")},
    "amp_moments": {"shape": list(amp.running_mean.shape), "dtype": str(amp.running_mean.dtype),
        "public_count": float(amp.count), "candidate_count_after_existing_cap": float(trainer.amp_normalizer.count),
        "mean_min": float(amp.running_mean.min()), "mean_max": float(amp.running_mean.max()),
        "variance_min": float(amp.running_var.min()), "variance_max": float(amp.running_var.max())},
    "history_normalization_max_abs_difference": normalized_error,
    "candidate_vs_public_logit_max_abs_difference": checks,
    "candidate_logit_action_change_max_abs_difference": action_change_error,
    "counterexample_zero_unit_rms_logit_max_abs_difference": float((wrong_rms_logits - public_logits).abs().max()),
    "counterexample_reversed_history_logit_max_abs_difference": float((wrong_order_logits - public_logits).abs().max()),
    "initial_action_input_gradient_max_abs": float(input_gradient[:, 1950:].detach().abs().max()),
    "action_column_parameter_gradient": {"active_coordinates": int(trainer.action_mask.sum()),
        "active_nonzero_entries": int(torch.count_nonzero(active_gradient)),
        "active_gradient_norm": float(active_gradient.norm()),
        "active_nonzero_columns": int((active_gradient.abs().sum(0) > 0).sum()),
        "frozen_coordinates": int((trainer.action_mask == 0).sum()),
        "frozen_nonzero_entries": int(torch.count_nonzero(frozen_gradient))},
    "actor_critic_task_and_other_moments_unchanged": True,
    "protected_terrain_checkpoint_gate_passed": True,
    "optimizer_steps": 0, "replay_size": trainer.policy_replay.size,
    "update_count": trainer.update_count, "physical_transition_count": trainer.physical_transition_count,
    "cuda_initialized": False, "kit_imported": False,
    "motion_quality_evidence": False,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
