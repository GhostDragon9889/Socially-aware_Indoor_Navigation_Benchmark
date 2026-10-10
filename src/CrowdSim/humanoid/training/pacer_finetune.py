"""Conservative PPO+AMP adaptation of public PACER to the fixed real AMASS bank.

The frozen teacher only regularizes actions on actual simulator observations.
It supplies neither rewards nor future mocap poses. All four public moments
remain frozen so exported inference uses exactly the training observation map.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

from .trainer import PaperAMPTrainer
from ..amp.controller import NO_SHAPE_SHA256, PacerAMPController
from ..amp.network import PacerAMPNetwork


class AnchoredPacerTrainer(PaperAMPTrainer):
    def __init__(self, config, *, device="cpu", provenance=None):
        super().__init__(config, device=device, provenance=provenance)
        if (config.discriminator_input != "state" or config.amp_normalization != "per_history_coordinate"
                or not config.warm_start_discriminator or not config.effective_action_likelihood):
            raise ValueError("Fine tuning requires the complete public state-only AMP contract")
        self.freeze_normalizers = True
        self.teacher = None
        self.anchor_coefficient = float(self.provenance["finetune"]["anchor_coefficient"])
        if not 0 < self.anchor_coefficient < 100:
            raise ValueError("A finite positive teacher coefficient is required")

    def bind_teacher(self, source, lower, upper):
        reference = PacerAMPController.from_checkpoint(source, lower, upper, device=self.device)
        self.teacher = nn.ModuleDict({"task": copy.deepcopy(reference.network._task_mlp),
                                      "actor": copy.deepcopy(reference.network.actor_mlp),
                                      "mean": copy.deepcopy(reference.network.mu)})
        self.teacher.requires_grad_(False).eval()
        self.teacher_log_std = reference.network.sigma.detach().clone()
        if not torch.equal(self.model.log_std, self.teacher_log_std):
            raise ValueError("Anchor requires unchanged public Gaussian variance")

    def policy_regularization(self, distribution, self_obs, task_obs):
        if self.teacher is None:
            raise RuntimeError("Public frozen teacher must be bound before any optimizer update")
        with torch.no_grad():
            features = torch.cat((self_obs, self.teacher["task"](task_obs)), -1)
            teacher_mean = self.teacher["mean"](self.teacher["actor"](features))
        # Exact KL of equal-variance Gaussians, on physically active coordinates.
        kl = .5 * ((distribution.mean - teacher_mean) / self.teacher_log_std.exp()).square()
        return self.anchor_coefficient * (kl * self.action_mask).sum(-1).mean()

    @torch.no_grad()
    def export(self, path, *, coverage, manifest_sha256):
        if self.update_count < 1 or self.physical_transition_count < 1:
            raise ValueError("Cannot export an untrained initialization as fine tuned")
        network = PacerAMPNetwork().to(self.device)
        network._task_mlp.load_state_dict(self.model.task_encoder.state_dict())
        for source, trunk, head in ((self.model.actor, network.actor_mlp, network.mu),
                                    (self.model.critic, network.critic_mlp, network.value),
                                    (self.model.discriminator, network._disc_mlp, network._disc_logits)):
            for index in (0, 2):
                trunk[index].load_state_dict(source[index].state_dict())
            head.load_state_dict(source[4].state_dict())
        network.sigma.copy_(self.model.log_std)
        if self.self_normalizer.count != self.task_normalizer.count:
            raise ValueError("Public combined observation RMS requires identical counts")
        def rms(source):
            return {"running_mean": source.mean.cpu(), "running_var": source.variance.cpu(),
                    "count": source.count.cpu()}
        observation = {"running_mean": torch.cat((self.self_normalizer.mean, self.task_normalizer.mean)).cpu(),
                       "running_var": torch.cat((self.self_normalizer.variance, self.task_normalizer.variance)).cpu(),
                       "count": self.self_normalizer.count.cpu()}
        metadata = {"schema": "crowdsim.pacer_amass_finetune.v1", "parent_sha256": NO_SHAPE_SHA256,
            "manifest_sha256": manifest_sha256, "train_clip_count": 2848,
            "expert_unique_motion_ids": sorted(coverage), "all_train_clips_seen": len(coverage) == 2848,
            "updates": self.update_count, "physical_transitions": self.physical_transition_count,
            "normalizers_frozen": True, "amp_sample_offsets_s": [i*.5 for i in range(10)],
            "anchor_coefficient": self.anchor_coefficient, "author_experiment_reproduction": False,
            "training_provenance": self.provenance}
        state = {"model": {"a2c_network." + k: v.cpu() for k, v in network.state_dict().items()},
                 "running_mean_std": observation, "reward_mean_std": rms(self.value_normalizer),
                 "amp_input_mean_std": rms(self.amp_normalizer), "local_finetune": metadata}
        # Round-trip equality for every layer. This is not a quality claim.
        probe_self = torch.randn(4, 357, device=self.device)
        probe_task = torch.randn(4, 1044, device=self.device)
        distribution, values = self.model.policy(probe_self, probe_task)
        probe = torch.cat((probe_self, probe_task), -1)
        if not (torch.equal(distribution.mean, network.eval_actor(probe)[0])
                and torch.equal(values, network.eval_critic(probe).squeeze(-1))):
            raise RuntimeError("Exported public-schema policy is not exactly equivalent")
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size, "metadata": metadata, "inference_equivalence": True}
