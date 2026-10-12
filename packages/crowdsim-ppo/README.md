# PointGoal PPO package metadata

The baseline implementation lives in `src/crowdsim_ppo` and is included in the root project. This metadata describes the minimal library distribution; benchmark runtime setup is documented in [the installation guide](../../docs/guides/installation.en.md).

Configure the external SDK or install the pinned SDK through uv, then run from the repository root:

```bash
python3 scripts/run_external_python.py -m social_navigation baseline-train --help
python3 scripts/run_external_python.py -m social_navigation baseline-evaluate --help
```

Pass `--config` for a custom PointGoal configuration. This baseline uses tensor-based environments and does not start Isaac Sim.
