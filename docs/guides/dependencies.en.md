# Project dependencies

[简体中文](dependencies.zh-CN.md) · [Documentation index](README.md) · Updated: 2026-10-10

## 1. Sources of truth

| File | Purpose |
| --- | --- |
| [Root pyproject.toml](../../pyproject.toml) | Distribution, core requirements, backend extras and console commands |
| [SDK manifest](../../env/isaaclab3_sim6/pyproject.toml) | Complete environment configuration for the IsaacLab checkout root |
| [SDK uv.lock](../../env/isaaclab3_sim6/uv.lock) | 273 locked packages, origins and wheel hashes |
| [profile.json](../../env/isaaclab3_sim6/profile.json) | Platform, source revision, versions and manifest/lock hashes |
| [Standalone PPO package](../../packages/crowdsim-ppo/pyproject.toml) | NumPy/PyTorch navigation baseline without a simulator |
| [Complete dependency inventory](dependency-inventory.json) | Readable snapshot of declarations and all 273 locked packages |

The root requirements are version ranges, not a complete environment lock. `requirements_isaaclab.txt` is a supplementary list. Use the SDK manifest, lock and profile together for physical pedestrians.

## 2. Pinned physical pedestrian environment

| Item | Pinned value |
| --- | --- |
| Platform | Linux x86_64 |
| Python | 3.12; local recorded version 3.12.13 |
| IsaacLab source | `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8` |
| IsaacLab product / Python package version | `3.0.0` / `12.0.0` |
| Isaac Sim Python package | `6.0.0.1` |
| This project | `3.1.0` |

The pinned IsaacLab revision is a development commit, not the `v3.0.0-beta2` tag. Official [beta2](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2) supports Sim 6.0; this project still requires the exact commit. [3.0 EA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA) requires Sim 6.1, so the shared “3.0” label does not establish compatibility.


### Actual local state update (2026-10-10)

The SDK manifest/lock SHAs match the snapshots, all 272 nonvirtual locked versions match, and all 18 project core constraints are satisfied. `uv pip check` still reports the three overrides below. Current SDK HEAD is `5951d48dde083986489957a7b108ad48a46822a7` (dependency configuration commit), while the profile records `4ecd0b...`. Only manifest/lock changed between those commits; SDK source is unchanged, but `apply` still rejects the current HEAD through its exact commit gate. Package versions, 234 USD RECORD files and the standalone round-trip passed `check`. These checks have distinct scopes.

The separate `/project/EXP/IsaacLab` uses Sim6.1/package17.0.2 and lacks seven project core packages, with two further core version conflicts. See the [comprehensive audit (Chinese)](../RESOURCE_DEPENDENCY_MAP_20261010_ZH.md) and [machine-readable inventory](../resource_audit_20261010/inventory.json).

### Selected locked versions

These values come directly from the repository lock, not from an audit of another machine.

| Package | Locked version |
| --- | --- |
| `isaaclab` | `12.0.0` |
| `isaacsim` | `6.0.0.1` |
| `torch` | `2.11.0+cu128` |
| `torchvision` | `0.26.0+cu128` |
| `torchaudio` | `2.11.0+cu128` |
| `numpy` | `2.3.1` |
| `scipy` | `1.17.0` |
| `lightning` | `2.5.6` |
| `tensordict` | `0.9.0` |
| `wandb` | `0.23.0` |
| `usd-exchange` | `3.0.0` |
| `mujoco-usd-converter` | `0.2.0` |
| `urdf-usd-converter` | `0.1.2` |
| `smplx` | `0.1.28` |
| `chumpy` | `0.70` |
| `opencv-python-headless` | `4.13.0.90` |
| `pillow` | `12.1.1` |
| `moviepy` | `1.0.3` |
| `imageio-ffmpeg` | `0.6.0` |
| `rtree` | `1.4.1` |
| `scikit-image` | `0.26.0` |
| `trimesh` | `4.11.1` |
| `numba` | `0.64.0` |
| `warp-lang` | `1.15.0.dev20260626` |
| `newton` | `1.4.0.dev0` |
| `newton-usd-schemas` | `0.5.0` |
| `coverage` | `7.6.1` |
| `pytest` | `9.0.3` |

## 3. Core Python requirements

| Declaration in the root pyproject.toml |
| --- |
| `dm-control>=1.0` |
| `easydict>=1.10` |
| `lightning>=2.3,<2.6` |
| `matplotlib>=3.6` |
| `moviepy>=1.0.3,<2.0.0.dev0; python_version >= '3.9'` |
| `numpy>=1.21.1` |
| `omegaconf==2.3.0` |
| `PyYAML>=6` |
| `rich>=13` |
| `rtree>=1.2,<2` |
| `scikit-image>=0.19` |
| `scipy>=1.8` |
| `tensordict>=0.5,<0.10` |
| `termcolor>=1.1` |
| `tensorboardX>=2.6` |
| `torch>=2.2` |
| `trimesh>=4.4` |
| `wandb>=0.15,<0.24` |

Torch/TensorDict/Lightning support policies and training; NumPy/SciPy/image libraries support data and evaluation; MuJoCo/dm-control/Trimesh/Rtree support motion, geometry and conversion; OmegaConf/PyYAML handle configuration; MoviePy/image tools handle video; W&B/TensorBoardX handle logging.

## 4. Extras and separate environments

| Install target | Purpose | Guidance |
| --- | --- | --- |
| `isaaclab` | Physical pedestrian backend | Establish the pinned SDK environment before installing this checkout editable |
| `crowd` | Pillow and GUI OpenCV helpers | The SDK lock chooses headless OpenCV; avoid installing both OpenCV providers |
| `dev` | pytest, ONNX export, build and publishing tools | Select tools for the task |
| `docs` | Upstream Sphinx tools | These bilingual guides are Markdown |
| `mujoco` / `newton` / `genesis` | Other upstream simulator backends | Use separate environments and their requirements; the four-scene results do not validate these backends |
| `isaacgym` | Legacy upstream interface; empty extra | This extra does not install the IsaacGym SDK |
| `packages/crowdsim-ppo` | PointGoal PPO baseline | NumPy ≥1.24, Torch ≥2.2, Python ≥3.10 only |

Do not run root-level `uv sync --all-extras` inside the SDK environment. Backend conflicts and Torch requirements differ. Synchronize the complete SDK snapshot from the IsaacLab workspace.

## 5. Compatibility settings to preserve

- Standalone Python uses `usd-exchange==3.0.0`; `usd-core` is excluded. They overwrite the same `pxr` files; see the [official usd-exchange package](https://pypi.org/project/usd-exchange/3.0.0/).
- Standalone USD is `26.8`; the recorded local Kit runtime loaded `25.11`. The environment `check` validates a standalone process, not Kit's USD version.
- The manifest overrides Newton, schemas and coverage. Known `uv pip check` differences against Sim metadata are `newton==1.4.0.dev0`, `newton-usd-schemas==0.5.0`, and `coverage==7.6.1`. Compare actual diagnostics; investigate additional conflicts rather than accepting any nonzero result.
- Keep the Linux x86_64 `tool.uv.environments` restriction and platform conditions. They avoid impossible cross-platform `pytetwild` resolution branches.
- `--locked` checks manifest/lock consistency; `--frozen` skips that check and is not a conflict repair. See [uv's synchronization documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

## 6. System tools and external resources

Physical runs require Git/Git LFS, uv, an NVIDIA driver and hardware meeting the [Sim 6.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html). Video tools invoke `ffmpeg` and `ffprobe`; the Python `imageio-ffmpeg` package alone does not ensure both system commands exist. Pip installation requires Python 3.12 and Linux GLIBC ≥2.35; see [NVIDIA's installation guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html).

AMASS, SMPL models, TRACE/PACER weights, scene USD/materials, native query maps and historical execution evidence are separate resources. Continue with [configuration](configuration.en.md) after installing packages.

The public PACER downloader additionally uses `gdown==6.2.0` and `requests`; prefer a separate download environment. Consult root `LICENSE.md`, `legal/`, and reference repositories for data/model license files.

Next: [Installation](installation.en.md) → [Configuration](configuration.en.md) → [Usage](usage.en.md).
