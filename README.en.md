# Socially-aware_Indoor_Navigation_Benchmark

[简体中文](README.md) · **English** · [Documentation](docs/guides/README.md)

An indoor social navigation research project using Isaac Sim 6 / IsaacLab 3, TRACE trajectory diffusion and physical PACER/AMP pedestrians. Core code lives under `src/`; ProtoMotions and CrowdSim import names and upstream attribution are retained.

## Documentation

| Topic | English | 中文 |
| --- | --- | --- |
| Dependencies and compatibility | [Dependencies](docs/guides/dependencies.en.md) | [依赖](docs/guides/dependencies.zh-CN.md) |
| Installation and troubleshooting | [Installation](docs/guides/installation.en.md) | [安装](docs/guides/installation.zh-CN.md) |
| Assets, checkpoints and settings | [Configuration](docs/guides/configuration.en.md) | [配置](docs/guides/configuration.zh-CN.md) |
| Fine-tuning, scene tests and video | [Usage](docs/guides/usage.en.md) | [使用](docs/guides/usage.zh-CN.md) |
| Published models, robot and four scenes | [Asset release](ASSET_RELEASE.md#english-instructions) | [资源发布](ASSET_RELEASE.md) |

## Available local results

- AMP controllers fine-tuned on genuine local AMASS motions, with scene routes and camera configurations.
- One continuous 60-second video for each of YOHO, NewMetro, SecondFloor and ThirdFloor, plus physical, posture, speed, spatial stasis and recording checks.
- Actual process exits, source snapshots, weight/resource hashes and complete video decoding records.
- A standalone PointGoal PPO baseline that runs without a simulator.

See the [local delivery index](docs/four_scene_speed_final_delivery_20261010.json) and [four-scene report (Chinese)](docs/FOUR_SCENE_SPEED_20261010_ZH.md). Results cover bounded local scenes, not the complete author protocol or verified paper-level performance. Robot social navigation and full-scene route coverage still require independent experiments.

## Structure

```text
src/
  social_navigation/  Unified commands and checkout discovery
  CrowdSim/           Crowds, trajectory diffusion, AMP and evaluation
  protomotions/       Simulator backends, motion libraries, training/inference
  crowdsim_ppo/       Standalone PointGoal PPO baseline
  deployment/         Deployment and motion processing
configs/              Experiments, scenes, routes, cameras and navigation
data/                 Locally managed AMASS, models and assets
resources/releases/   Private LFS models, robot, four scenes and release manifests
scripts/              Process entry points for training, simulation and audits
tools/                Asset conversion and motion retargeting
third_party/          Pinned reference repositories
packages/             Standalone baseline packaging
env/                  IsaacLab 3 / Isaac Sim 6 environment snapshots
tests/                Module and native execution contract tests
docs/                 Bilingual guides, provenance and experiment records
output/               Generated training, simulation and video artifacts
.vscode/              Interpreter, debugging, tasks and extension settings
```

See the [detailed structure (Chinese)](docs/PROJECT_STRUCTURE_ZH.md).

## Source and resources

Private repository: [GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark](https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark). Access requires an authorized GitHub account.

All 1,017 original upstream paths are mapped into the reorganized project, with 235 previously omitted files restored and the original 74-commit ProtoMotions history retained. Original LFS references download from NVlabs/ProtoMotions; approximately 5.3 GB of payloads are not mirrored into this private repository. Existing adaptations remain in the merged source. See the [merge record (Chinese)](docs/UPSTREAM_MERGE_20261010_ZH.md).

The selected fine-tuned controllers, training parameters, EVT-EXP robot and four-scene assets are packaged in the [private LFS release](ASSET_RELEASE.md#english-instructions). Raw AMASS motions, SMPL parameters, SDK workspaces and videos remain separately prepared. [Configuration](docs/guides/configuration.en.md) explains cross-machine requirements.

## Quick start with an existing environment

Use the pinned Python 3.12 environment. Root `.venv` can link to the adjacent `IsaacLab-Protomotions/.venv`.

```bash
cd /absolute/path/to/Socially-aware_Indoor_Navigation_Benchmark
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m social_navigation status
.venv/bin/python -m social_navigation preflight --scene YOHO
.venv/bin/python -m social_navigation simulate --scene YOHO --steps 1800
```

Preflight checks resources and native scene configuration. Simulation uses the owning parent launcher and a fresh output directory. 1,800 controls at 30Hz represent 60 physical seconds. Run native jobs serially. For a new machine, follow [installation](docs/guides/installation.en.md) before these commands.

## Training, evaluation and recording

```bash
.venv/bin/python -m social_navigation train-amp --help
.venv/bin/python -m social_navigation evaluate-physics --help
.venv/bin/python -m social_navigation evaluate-posture --help
.venv/bin/python -m social_navigation evaluate-speed --help
.venv/bin/python -m social_navigation evaluate-stasis --help
.venv/bin/python -m social_navigation audit-speed --help
.venv/bin/python -m social_navigation audit-feet --help
.venv/bin/python -m social_navigation audit-camera --help
.venv/bin/python -m social_navigation record --help
```

Tools retain quality gates, actual child exits and resource provenance. Use new output paths and follow the complete [usage workflow](docs/guides/usage.en.md); a new checkpoint requires independent admission before recording.

## VS Code

Open `Socially-aware_Indoor_Navigation_Benchmark.code-workspace`. Recommended extensions are Python, Pylance, Python Debugger and Ruff; the default interpreter is `.venv/bin/python`. Tasks and debug configurations expose status, preflight, baseline and native entry points.

## Upstream projects and licenses

The project includes [ProtoMotions](https://github.com/NVlabs/ProtoMotions), [TRACE / Trace and Pace](https://github.com/nv-tlabs/trace), PACER and NavIsaacLab-related implementation/adaptations. The [original ProtoMotions README](docs/reference/PROTOMOTIONS_README.md) is preserved.

Retain [LICENSE.md](LICENSE.md), `legal/`, and third-party license files. AMASS, SMPL/SMPL-H, weights and scene assets follow their own terms; body-model assets are excluded from built wheel/sdist distributions.
