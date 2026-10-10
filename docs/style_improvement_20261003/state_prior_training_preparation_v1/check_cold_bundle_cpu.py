"""CPU cold-state check of an actual new immutable bundle and its owned imports.

Only in-memory synthetic forwards/backward are performed. No optimizer step,
simulation, data rewrite, or checkpoint save is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-report", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.absolute()
    args.import_report = args.import_report.absolute()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU cold gate requires CUDA_VISIBLE_DEVICES=''"
    assert not args.output.exists(), "Exclusive cold gate report required"
    bundle = args.bundle.absolute()
    spec = importlib.util.spec_from_file_location("state_prior_owned_launcher", bundle / "launch_isolated.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    manifest = launcher.verify()
    ledger = launcher.ImportLedger(args.import_report)
    launcher.activate(ledger)
    status = "failed"
    try:
        import torch
        import yaml
        from torch.nn import functional as F
        from CrowdSim.humanoid.amp.controller import PacerAMPController
        from CrowdSim.humanoid.application.settings import TrainingConfig
        from CrowdSim.humanoid.application.state_prior_initialization import STRATEGY
        from CrowdSim.humanoid.training.trainer import PaperAMPTrainer

        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        assert not torch.cuda.is_initialized()
        profile_path = bundle / "code/configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml"
        profile = yaml.safe_load(profile_path.read_text())
        config = TrainingConfig(**profile["training"])
        assert config.warm_start_action_discriminator_from_state_prior is True
        assert config.warm_start_discriminator is False and config.discriminator_input == "state_action"
        assert config.task_terrain_statistics_mode == "preserve_declared_suffix"
        public = profile["initialization"]["checkpoint"]
        trainer = PaperAMPTrainer(config, device="cpu")
        initialization = trainer.warm_start_pacer(public, -torch.ones(69), torch.ones(69))
        reference = PacerAMPController.from_checkpoint(public, -torch.ones(69), torch.ones(69), device="cpu")
        first = trainer.model.discriminator[0]
        assert tuple(first.weight.shape) == (1024, 2019)
        assert torch.equal(first.weight[:, :1950], reference.network._disc_mlp[0].weight)
        assert torch.count_nonzero(first.weight[:, 1950:]) == 0
        assert torch.equal(first.bias, reference.network._disc_mlp[0].bias)
        for destination, source in ((trainer.model.discriminator[2], reference.network._disc_mlp[2]),
                                    (trainer.model.discriminator[4], reference.network._disc_logits)):
            assert all(torch.equal(tensor, source.state_dict()[name]) for name, tensor in destination.state_dict().items())
        amp = reference.discriminator_normalizer
        assert torch.equal(trainer.amp_normalizer.mean, amp.running_mean)
        assert torch.equal(trainer.amp_normalizer.variance, amp.running_var)
        assert trainer.amp_normalizer.mean.dtype == torch.float64
        assert float(amp.count) == 15482880001. and float(trainer.amp_normalizer.count) == 200000.
        assert initialization["normalizer_counts_before_cap"]["amp_normalizer"] == 15482880001.
        assert initialization["discriminator_initialization"] == STRATEGY
        assert initialization["published_state_only_discriminator_transferred"] is True
        assert initialization["published_amp_history_moments_transferred"] is True
        assert trainer.update_count == trainer.physical_transition_count == 0
        assert not trainer.optimizer.state and not trainer.discriminator_optimizer.state
        assert trainer.policy_replay.size == 0
        trainer.terrain_statistics_policy.validate_checkpoint(trainer.task_normalizer, update_count=0, physical_transition_count=0)
        generator = torch.Generator().manual_seed(20261003)
        raw = (amp.running_mean.float() + (amp.running_var.float() + 1e-5).sqrt() *
               torch.randn(64, 1950, generator=generator)).reshape(64, 10, 195)
        zero = torch.zeros(64, 69)
        random = (torch.rand(64, 69, generator=generator) * 2 - 1) * trainer.action_mask
        extreme = torch.ones(64, 69) * trainer.action_mask
        normalized, _ = trainer._amp_inputs(raw, zero)
        assert torch.equal(normalized.flatten(1), amp(raw.flatten(1)))
        errors = {}
        with torch.no_grad():
            public_logits = reference.eval_discriminator(raw).squeeze(-1)
            for label, actions in (("zero", zero), ("random", random), ("extreme", extreme)):
                actual = trainer.model.disc_logits(normalized, actions)
                torch.testing.assert_close(actual, public_logits, atol=1e-5, rtol=1e-5)
                errors[label] = float((actual - public_logits).abs().max())
        negative = trainer.model.disc_logits(normalized[:32].detach(), random[:32])
        features = torch.cat((normalized[32:].flatten(1).detach(), random[32:]), -1).requires_grad_(True)
        positive = trainer.model.discriminator(features).squeeze(-1)
        classification = .5 * (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive)) +
                               F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))
        gradient = torch.autograd.grad(positive.sum(), features, create_graph=True)[0]
        penalty = gradient.square().sum(-1).mean()
        logit_reg = trainer.model.discriminator[-1].weight.square().sum()
        l2 = sum(module.weight.square().sum() for module in trainer.model.discriminator if isinstance(module, torch.nn.Linear))
        loss = config.discriminator_coefficient * (classification + config.discriminator_gradient_penalty * penalty +
               config.discriminator_logit_regularization * logit_reg + config.discriminator_weight_decay * l2)
        loss.backward()
        action_gradient = first.weight.grad[:, 1950:]
        active = action_gradient[:, trainer.action_mask.bool()]
        frozen = action_gradient[:, ~trainer.action_mask.bool()]
        assert torch.isfinite(active).all() and (active.abs().sum(0) > 0).all()
        assert torch.count_nonzero(frozen) == 0 and torch.count_nonzero(gradient[:, 1950:]) == 0
        assert torch.count_nonzero(first.weight[:, 1950:]) == 0
        assert not torch.cuda.is_initialized()
        assert not any(name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules)
        report = {"schema": "public_state_prior.actual_bundle_cold_CPU.v1", "status": "passed", "future_effect": False,
            "bundle": str(bundle), "source_manifest_sha256": digest(bundle / "source_manifest.json"),
            "profile_sha256": digest(profile_path), "copied_regular_files": len(manifest["files"]),
            "public_checkpoint_sha256": reference.checkpoint_sha256, "D_input": 2019, "history_RMS_size": 1950,
            "public_AMP_count": 15482880001., "cold_AMP_count": 200000., "initialization": initialization,
            "cold_action_columns_nonzero": 0, "logit_max_abs_errors": errors, "active_action_gradient_columns": int(active.shape[1]),
            "active_gradient_norm": float(active.norm()), "frozen_action_gradient_nonzero": 0,
            "optimizer_steps": 0, "updates": 0, "physical_transitions": 0, "replay_size": 0,
            "CUDA_initialized": False, "Kit_imported": False, "motion_quality_evidence": False}
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        status = "passed"
    finally:
        ledger.finish(status)


if __name__ == "__main__":
    main()
