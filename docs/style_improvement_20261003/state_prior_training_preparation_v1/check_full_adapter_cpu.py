"""Check the retained old full-network adapter against a private cold fixture.

The temporary checkpoint contains an initialized CPU trainer, no physical
runner or optimizer steps. This proves interface compatibility, not training
resume, real motion, or terminal admission.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-report", type=Path, required=True)
    args = parser.parse_args()
    args.output, args.import_report = args.output.absolute(), args.import_report.absolute()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output.exists()
    bundle = args.bundle.absolute()
    spec = importlib.util.spec_from_file_location("full_adapter_owned_launcher", bundle / "launch_isolated.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.verify()
    ledger = launcher.ImportLedger(args.import_report)
    launcher.activate(ledger)
    status = "failed"
    try:
        import torch
        import yaml
        from CrowdSim.humanoid.amp.controller import PacerAMPController
        from CrowdSim.humanoid.application.settings import TrainingConfig
        from CrowdSim.humanoid.runtime.amp_inference import ReconstructionAMPInference
        from CrowdSim.humanoid.training.trainer import PaperAMPTrainer

        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        profile = yaml.safe_load((bundle / "code/configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml").read_text())
        trainer = PaperAMPTrainer(TrainingConfig(**profile["training"]), device="cpu")
        trainer.warm_start_pacer(profile["initialization"]["checkpoint"], -torch.ones(69), torch.ones(69))
        generator = torch.Generator().manual_seed(20261003)
        own_obs = torch.randn(3, 357, generator=generator)
        task_obs = torch.randn(3, 1044, generator=generator)
        expected = trainer.act(own_obs, task_obs, deterministic=True)
        with tempfile.TemporaryDirectory(prefix="state_prior_full_adapter_CPU_", dir="/tmp") as temporary:
            checkpoint = Path(temporary) / "private_cold_fixture.pt"
            trainer.save(checkpoint)
            checksum = launcher.digest(checkpoint)
            original_loader = PacerAMPController.from_checkpoint
            def forbidden_public_read(*_args, **_kwargs):
                raise AssertionError("Resume/inference must not reopen public PACER")
            PacerAMPController.from_checkpoint = forbidden_public_read
            try:
                rng_before = torch.get_rng_state().clone()
                inference = ReconstructionAMPInference.from_checkpoint(checkpoint, device="cpu", expected_sha256=checksum)
                assert torch.equal(rng_before, torch.get_rng_state())
                assert hasattr(inference.model, "critic") and tuple(inference.model.discriminator[0].weight.shape) == (1024, 2019)
                actual = inference.act(own_obs, task_obs)
                for name in ("actions", "applied_actions"):
                    assert torch.equal(expected[name], actual[name])
                resumed, runner_state = PaperAMPTrainer.load(checkpoint, device="cpu", restore_rng=False)
                assert runner_state is None
                assert resumed.config.warm_start_action_discriminator_from_state_prior is True
                assert resumed.provenance["initialization"] == trainer.provenance["initialization"]
                assert torch.equal(resumed.model.discriminator[0].weight, trainer.model.discriminator[0].weight)
                assert torch.equal(resumed.amp_normalizer.mean, trainer.amp_normalizer.mean)
            finally:
                PacerAMPController.from_checkpoint = original_loader
        assert not torch.cuda.is_initialized()
        with args.output.open("x") as stream:
            json.dump({"schema": "public_state_prior.old_full_adapter_CPU.v1", "status": "passed", "future_effect": False,
                "source_manifest_sha256": launcher.digest(bundle / "source_manifest.json"), "private_cold_fixture_only": True,
                "retained_full_model_critic_and_D2019": True, "raw_applied_actions_exact": True,
                "inference_constructor_caller_CPU_RNG_unchanged": True, "resume_and_inference_public_reads": 0,
                "cold_resume_D_and_RMS_exact": True, "optimizer_steps": 0, "updates": 0, "physical_transitions": 0,
                "GPU_started": False, "Kit_started": False, "physical_runner_resume_proven": False,
                "motion_quality_evidence": False}, stream, indent=2, allow_nan=False)
            stream.write("\n")
        status = "passed"
    finally:
        ledger.finish(status)


if __name__ == "__main__":
    main()
