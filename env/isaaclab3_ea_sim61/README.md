# IsaacLab 3.0-EA / Isaac Sim 6.1

This directory holds the pinned SDK manifest, uv lock, provider versions and exact project additions lock. It is the input to `scripts/bootstrap_reproduction.py install`, not an independent project overlay environment.

Use `bootstrap_reproduction.py configure --lab-root ../IsaacLab` for an existing SDK, or `bootstrap_reproduction.py install` for a new uv-managed SDK under `.runtime/IsaacLab61`. Run project commands through `scripts/run_external_python.py`.

See [installation](../../docs/guides/installation.en.md) and [中文](../../docs/guides/installation.zh-CN.md). The SDK owns Torch, NumPy, Warp and USD; project additions must not replace them.
