# Installation

[中文](installation.zh-CN.md) · [Guide index](README.md)

Follow [reproduction from a Git clone](../../CLONE_REPRODUCTION.en.md) for the current physical pedestrian pipeline. The project bootstrap installs the pinned SDK, restores Git LFS resources and rebuilds the fixed AMASS banks. Sibling-directory layouts and chat workspace code are not required.

```bash
python3 scripts/bootstrap_reproduction.py install
export NAV_PY="$PWD/.runtime/IsaacLab/.venv/bin/python"
export PYTHONPATH="$PWD/src:$PWD"
"$NAV_PY" scripts/manage_release_assets.py fetch
"$NAV_PY" scripts/manage_release_assets.py prepare --check-usd
```

Matching existing SDK environments can use `scripts/bootstrap_reproduction.py prepare-native`. See [dependencies](dependencies.en.md) for system requirements, versions and recorded overrides, and [usage](usage.en.md) for training/evaluation/recording. Complete licensed AMASS/SMPL, upstream mesh LFS and EULA instructions are in the clone guide.

CPU-only source inspection: `python3 scripts/audit_checkout.py`. Independent PointGoal PPO uses `configs/crowdsim_ppo/` and `src/crowdsim_ppo/`; its CPU evaluation does not establish full 3D pedestrian performance.
