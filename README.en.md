# Socially-aware Indoor Navigation Benchmark

[中文](README.md)

Indoor social navigation research with TRACE trajectory diffusion, physical PACER/AMP pedestrians and PPO. The supported runtime is IsaacLab 3.0-EA / Isaac Sim 6.1.

## Setup

Configure an existing SDK:

```bash
python3 scripts/bootstrap_reproduction.py configure --lab-root ../IsaacLab
```

Or install the pinned SDK with uv:

```bash
python3 scripts/bootstrap_reproduction.py install
```

Run with the SDK interpreter:

```bash
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
python3 scripts/run_external_python.py -m social_navigation preflight --scene YOHO
```

See [installation](docs/guides/installation.en.md), [resource preparation](CLONE_REPRODUCTION.en.md), [usage](docs/guides/usage.en.md), [layout](docs/structure.md), [verification and cache maintenance](docs/verification.md) and [validation limits](docs/validation.md). Code belongs in `src`, maintained configuration in `configs`, entry points in `scripts`, and regression tests in `tests`. Downloaded resources and generated results are separate from source.

The implementation builds on [ProtoMotions](https://github.com/NVlabs/ProtoMotions), [TRACE](https://github.com/nv-tlabs/trace) and [PACER](https://github.com/nv-tlabs/pacer). See [LICENSE.md](LICENSE.md) and `legal/`. SMPL/AMASS have separate licensing terms. Full code coverage, long-term policy convergence and every scene's physical quality have not been established.

ProtoMotions references: [Pretrained models](docs/source/getting_started/pretrained_models.rst) · [GPC and PEFT guide](docs/source/user_guide/gpc.rst).
Robot definitions and assets: `src/protomotions/data/assets/mjcf/`.
