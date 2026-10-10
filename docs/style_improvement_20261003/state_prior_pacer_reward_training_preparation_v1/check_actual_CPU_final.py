"""Check the real isolated loader, changed weight, resume guard and cold invariance."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def record(path):
    path = Path(path).absolute()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(b)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-report", type=Path, required=True)
    args = parser.parse_args()
    args.bundle = args.bundle.absolute()
    args.output = args.output.absolute()
    args.import_report = args.import_report.absolute()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("pacer_reward_owned_launcher", args.bundle / "launch_isolated.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    manifest = launcher.verify()
    ledger = launcher.ImportLedger(args.import_report)
    launcher.activate(ledger)
    status = "failed"
    try:
        from CrowdSim.humanoid.application.configuration import load_amp_profile, validate_resume_reward_config, ConfigurationError
        from CrowdSim.humanoid.application.settings import TrainingConfig
        profile = load_amp_profile(manifest["profile"]["path"])
        config = TrainingConfig(**deepcopy(profile["training"]))
        config.validate()
        assert config.amp_reward_weight == .5 and profile["reward"]["amp_weight"] == .5
        assert config.amp_reward_scale == 2. and config.expert_sampling is None
        assert config.warm_start_action_discriminator_from_state_prior is True
        assert config.task_terrain_statistics_mode == "preserve_declared_suffix"
        assert not any(k == "torch" or k == "isaacsim" or k.startswith(("isaaclab", "omni")) for k in sys.modules)
        validate_resume_reward_config(profile["training"], config)
        old_values = deepcopy(profile["training"])
        old_values["amp_reward_weight"] = 2.
        old_config = TrainingConfig(**old_values)
        rejected = False
        try:
            validate_resume_reward_config(profile["training"], old_config)
        except ConfigurationError:
            rejected = True
        assert rejected
        import torch
        from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        new = PaperAMPTrainer(config, device="cpu")
        new.warm_start_pacer(profile["initialization"]["checkpoint"], -torch.ones(69), torch.ones(69))
        new_rng = torch.get_rng_state().clone()
        old = PaperAMPTrainer(old_config, device="cpu")
        old.warm_start_pacer(profile["initialization"]["checkpoint"], -torch.ones(69), torch.ones(69))
        assert torch.equal(new_rng, torch.get_rng_state())
        assert asdict(new.config) == {**asdict(old.config), "amp_reward_weight": .5}
        assert new.model.state_dict().keys() == old.model.state_dict().keys()
        assert all(torch.equal(x, old.model.state_dict()[k]) for k, x in new.model.state_dict().items())
        for name in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"):
            a, b = getattr(new, name), getattr(old, name)
            assert all(torch.equal(x, b.state_dict()[k]) for k, x in a.state_dict().items())
        for trainer in (new, old):
            assert not trainer.optimizer.state and not trainer.discriminator_optimizer.state
            assert trainer.update_count == trainer.physical_transition_count == trainer.policy_replay.size == 0
            assert float(trainer.amp_normalizer.count) == 200000.
            assert torch.count_nonzero(trainer.model.discriminator[0].weight[:, 1950:]) == 0
            trainer.terrain_statistics_policy.validate_checkpoint(trainer.task_normalizer, update_count=0, physical_transition_count=0)
        g = torch.Generator().manual_seed(20261003)
        own, task = torch.randn(2, 357, generator=g), torch.randn(2, 1044, generator=g)
        a, b = new.act(own, task, deterministic=True), old.act(own, task, deterministic=True)
        assert torch.equal(a["actions"], b["actions"]) and torch.equal(a["applied_actions"], b["applied_actions"])
        h = torch.randn(2, 10, 195, generator=g)
        raw_a, raw_b = new.amp_reward(h, a["applied_actions"]), old.amp_reward(h, a["applied_actions"])
        assert torch.equal(raw_a, raw_b)
        assert torch.equal(new.config.amp_reward_weight * raw_a, (old.config.amp_reward_weight * raw_b) * .25)
        assert not torch.cuda.is_initialized()
        assert not any(k == "isaacsim" or k.startswith(("isaaclab", "omni")) for k in sys.modules)
        result = {"schema": "pacer_reward.actual_preparation_admission.v1", "status": "passed", "CPU_only": True, "GPU_used": False, "sealed34_outputs_read": False,
                  "source_manifest": record(args.bundle / "source_manifest.json"), "profile": record(manifest["profile"]["path"]),
                  "copied_regular_files": len(manifest["files"]), "reward_contract_validated": True,
                  "loaded_training_amp_reward_weight": .5, "loaded_reward_amp_weight_declaration": .5,
                  "loaded_amp_reward_scale": 2., "effective_softplus_coefficient": 1.,
                  "old_weight2_resume_rejected": True, "same_weight_half_resume_accepted": True,
                  "actual_owned_loader": True, "SDK_free_config_loader": True,
                  "cold_models_all_tensors_equal_except_config_weight": True, "cold_normalizers_all_tensors_equal": True,
                  "cold_constructor_and_warm_start_Torch_RNG_equal": True, "cold_actor_actions_equal": True,
                  "raw_AMP_reward_same": True, "weighted_AMP_reward_is_quarter": True, "reward_contract": {"amp_reward_scale": 2.0, "amp_reward_weight": 0.5, "task_reward_weight": 0.5, "direction_mode": "backwards_penalty", "amp_probability_epsilon": 0.0001},
                  "both_cold_optimizer_empty": True, "cold_updates": 0, "cold_transitions": 0, "cold_replay": 0,
                  "public_coupled_D_and_AMP_RMS_transfer": True, "protected_public_terrain_moments": True,
                  "CUDA_initialized": False, "SDK_started": False, "Kit_started": False,
                  "future_effect": False, "training_executed": False, "quality_claim": False,
                  "checks": [{"name": x, "passed": True} for x in ("real_loader", "two_key_profile_delta", "old_resume_rejection", "cold_model_identity", "cold_moments_identity", "cold_RNG_identity", "weighted_reward_quarter")]}
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        status = "passed"
    finally:
        ledger.finish(status)


if __name__ == "__main__":
    main()
