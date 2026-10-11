# Socially-aware_Indoor_Navigation_Benchmark

The latest [verification report and remaining limits](docs/VERIFICATION_REMAINING_20261011_ZH.md) records the complete unfiltered regression, 256-actor training and the PhysX exit-crash fix. Statement coverage is 64.37%; complete feature and model-quality acceptance remains open.

**Reproduce from a clone:** [complete instructions](CLONE_REPRODUCTION.en.md) for SDK bootstrap, resource restoration, fixed AMASS rebuild, four-scene evaluation and recording.

[简体中文](README.md) · **English** · [Documentation](docs/guides/README.md)

The default migration target is **Isaac Sim 6.1.0.0 / IsaacLab 3.0 EA**, using TRACE trajectory diffusion and physical PACER/AMP pedestrians. See [runtime instructions](env/isaaclab3_ea_sim61/README.md). Core code lives under `src/`; ProtoMotions and CrowdSim import names and upstream attribution are retained.

## Documentation

| Topic | English | 中文 |
| --- | --- | --- |
| Dependencies and compatibility | [Dependencies](docs/guides/dependencies.en.md) | [依赖](docs/guides/dependencies.zh-CN.md) |
| Installation and troubleshooting | [Installation](docs/guides/installation.en.md) | [安装](docs/guides/installation.zh-CN.md) |
| Assets, checkpoints and settings | [Configuration](docs/guides/configuration.en.md) | [配置](docs/guides/configuration.zh-CN.md) |
| Fine-tuning, scene tests and video | [Usage](docs/guides/usage.en.md) | [使用](docs/guides/usage.zh-CN.md) |
| Published models, robot and four scenes | [Asset release](ASSET_RELEASE.md#english-instructions) | [资源发布](ASSET_RELEASE.md) |
| Verified local resources and dependencies | [2026-10-10 audit (Chinese)](docs/RESOURCE_DEPENDENCY_MAP_20261010_ZH.md) | [代码结构](docs/codebase/STRUCTURE.md) |

## Available local results

The Sim6.1 runtime has separately recorded dependency checks, the current full regression suite, real CUDA tests, AMASS AMP training smoke, three-frame RTX capture and four60-second physical scene runs. See the [migration audit](docs/ISAACLAB3_EA_SIM61_MIGRATION_20261011_EN.md). The four60-second videos below retain their historical Sim6.0 identity.

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

All 1,017 original upstream paths are mapped into the reorganized project, with 235 previously omitted files restored and the original 74-commit ProtoMotions history retained. Original LFS references download from NVlabs/ProtoMotions; approximately 5.3 GB of payloads are not mirrored into this repository. Existing adaptations remain in the merged source. See the [merge record (Chinese)](docs/UPSTREAM_MERGE_20261010_ZH.md).

The selected fine-tuned controllers, training parameters, EVT-EXP robot and four-scene assets are packaged in the [private LFS release](ASSET_RELEASE.md#english-instructions). Raw AMASS motions, SMPL parameters, SDK workspaces and videos remain separately prepared. [Configuration](docs/guides/configuration.en.md) explains cross-machine requirements.

## Quick start with an existing environment

Configure an installed external SDK from the project root. The launcher uses that SDK's Python directly; project additions live in `.runtime/external_sdk_packages`. No project venv or editable install into the SDK is required.

```bash
python3 scripts/bootstrap_reproduction.py configure --lab-root /absolute/path/to/IsaacLab
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
python3 scripts/run_external_python.py -m social_navigation preflight --scene YOHO
python3 scripts/run_external_python.py -m social_navigation simulate --scene YOHO --steps 60 --output output/sim61_smoke
```

For a new machine, `python3 scripts/bootstrap_reproduction.py install` installs the pinned Lab3EA/Sim6.1 SDK under `.runtime/IsaacLab61`. Use the same launcher afterwards. `--python` selects another SDK interpreter, including a binary Sim `python.sh` with Lab installed. Read and accept NVIDIA's EULA before first Kit launch. See [external SDK setup](docs/guides/external_sdk.zh-CN.md).

Preflight checks resources and native scene configuration. Simulation uses the owning parent launcher and a fresh output directory. The short 60-control smoke command does not replace 1,800-control (60-second) quality/recording acceptance. Run native jobs serially. For a new machine, follow [installation](docs/guides/installation.en.md) before these commands.

## Training, evaluation and recording

```bash
python3 scripts/run_external_python.py -m social_navigation train-amp --help
python3 scripts/run_external_python.py -m social_navigation evaluate-physics --help
python3 scripts/run_external_python.py -m social_navigation evaluate-posture --help
python3 scripts/run_external_python.py -m social_navigation evaluate-speed --help
python3 scripts/run_external_python.py -m social_navigation evaluate-stasis --help
python3 scripts/run_external_python.py -m social_navigation audit-speed --help
python3 scripts/run_external_python.py -m social_navigation audit-feet --help
python3 scripts/run_external_python.py -m social_navigation audit-camera --help
python3 scripts/run_external_python.py -m social_navigation record --help
```

Tools retain quality gates, actual child exits and resource provenance. Use new output paths and follow the complete [usage workflow](docs/guides/usage.en.md); a new checkpoint requires independent admission before recording.

## VS Code

Open `Socially-aware_Indoor_Navigation_Benchmark.code-workspace`. Recommended extensions are Python, Pylance, Python Debugger and Ruff; open the generated `.runtime/external_sdk.code-workspace` to use the configured external SDK interpreter. Tasks and debug configurations expose status, preflight, baseline and native entry points.

## Upstream projects and licenses

The project includes [ProtoMotions](https://github.com/NVlabs/ProtoMotions), [TRACE / Trace and Pace](https://github.com/nv-tlabs/trace), PACER and NavIsaacLab-related implementation/adaptations. The [original ProtoMotions README](docs/reference/PROTOMOTIONS_README.md) is preserved.

Retain [LICENSE.md](LICENSE.md), `legal/`, and third-party license files. AMASS, SMPL/SMPL-H, weights and scene assets follow their own terms; body-model assets are excluded from built wheel/sdist distributions.

## Upstream GPC and PEFT

See the [GPC and PEFT guide](docs/source/user_guide/gpc.rst). Humanoid MJCF assets live under `src/protomotions/data/assets/mjcf/` (compatibility path `protomotions/data/assets/mjcf/`). Model cards document supported runtimes and archived checkpoint restrictions.
