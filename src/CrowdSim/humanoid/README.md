# Humanoid trajectory and AMP runtime

The current benchmark uses TRACE trajectory generation and PACER AMP control in four indoor scenes. The selected weights, scene profiles and resource identities are declared in `configs/reproduction/four_scene/` and `configs/diffusion_amp/four_scene_speed_20261010/`.

Configure the external SDK or install the pinned SDK through uv using the [installation guide](../../../docs/guides/installation.en.md). Run commands through `python3 scripts/run_external_python.py` from the repository root.

```bash
python3 scripts/run_external_python.py -m social_navigation preflight --scene YOHO
python3 scripts/run_external_python.py -m social_navigation simulate --scene YOHO --steps 1800
python3 scripts/run_external_python.py -m social_navigation train-amp --help
python3 scripts/run_external_python.py scripts/train_paper_amp.py --help
python3 scripts/run_external_python.py scripts/train_paper_diffusion.py --help
python3 scripts/run_external_python.py scripts/train_shared_social.py --check-sources
```

See the [training, evaluation and recording guide](../../../docs/guides/usage.en.md) for complete commands. AMP training retains its expert observations, PPO/discriminator updates and checkpoint contracts. Research training requires an explicit compatible diffusion checkpoint when the selected trajectory source is diffusion; old experiment outputs are no longer supplied as defaults.

Fixed S3/episode historical replay, owner completion and comparison tools have been retired. General training, evaluation, source integrity and mathematical regression tests remain. Fresh grouped variants can be inspected with `train_shared_social_episode_reference.py --check-sources` and `train_shared_social_minimum_horizon.py --check-sources`; they no longer depend on a completed historical S3 run. Their profiles are in `configs/training/social_amp/`.

[Validation status and hardware/physical quality gaps](../../../docs/validation.md).
