# Installation guide

[简体中文](installation.zh-CN.md) · [Documentation index](README.md) · Updated: 2026-10-10

## 1. Choose an installation path

| Goal | Path |
| --- | --- |
| Browse source, inspect status and test layout | Section 5, without Isaac SDK |
| Standalone PointGoal PPO | Section 6, separate CPU environment |
| Reuse the local physical pedestrian environment | Section 4, existing `.venv` |
| Prepare a new physical environment | Sections 2–3, followed by resource preparation |

Examples use Bash. Replace `/absolute/path/to/...` with your paths. The new-workspace procedure requires adjacent source directories. This documentation verification did not reinstall the complete SDK or run an end-to-end simulation on a clean machine.

## 2. System checks and clone

Install Git, Git LFS and uv; recording also needs ffmpeg/ffprobe. On Ubuntu/Debian, an administrator can install system tools with `sudo apt install git git-lfs ffmpeg`. Hardware and GLIBC requirements are linked in [dependencies](dependencies.en.md).

```bash
nvidia-smi
ldd --version
uv --version
git lfs version
ffmpeg -version
ffprobe -version
```

`nvidia-smi` and `ffmpeg` are unnecessary for the CPU-only status check. The private GitHub repository requires an authorized account's Git credentials. Keep tokens out of URLs and configuration files.

```bash
export NAV_WORKSPACE=/absolute/path/to/new-workspace
mkdir -p "$NAV_WORKSPACE"
cd "$NAV_WORKSPACE"
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark.git
export NAV_ROOT="$NAV_WORKSPACE/Socially-aware_Indoor_Navigation_Benchmark"
cd "$NAV_ROOT"
git lfs install --local
python3 scripts/verify_upstream_integration.py
```

`verify_upstream_integration.py` checks the Git mapping of 1,017 upstream files; it neither downloads LFS nor validates private data. LFS downloads target NVlabs/ProtoMotions; selective downloads are in section 7.

## 3. Create the pinned SDK environment

Run this only in a fresh `NAV_WORKSPACE`. Clone a separate IsaacLab workspace, copy this project's snapshots to its root, then let uv create the environment. Do not synchronize from `env/isaaclab3_sim6/`, or replace the source revision with an arbitrary PyPI IsaacLab version.

```bash
cd "$NAV_WORKSPACE"
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab-Protomotions
export NAV_LAB_ROOT="$NAV_WORKSPACE/IsaacLab-Protomotions"
git -C "$NAV_LAB_ROOT" checkout --detach 4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8
test "$(cat "$NAV_LAB_ROOT/VERSION")" = 3.0.0
mkdir -p "$NAV_LAB_ROOT/.crowdsim-environment-backups/manual-bootstrap"
cp "$NAV_LAB_ROOT/pyproject.toml" \
  "$NAV_LAB_ROOT/.crowdsim-environment-backups/manual-bootstrap/pyproject.toml"
if [ -f "$NAV_LAB_ROOT/uv.lock" ]; then
  cp "$NAV_LAB_ROOT/uv.lock" \
    "$NAV_LAB_ROOT/.crowdsim-environment-backups/manual-bootstrap/uv.lock"
fi
cp "$NAV_ROOT/env/isaaclab3_sim6/pyproject.toml" "$NAV_LAB_ROOT/pyproject.toml"
cp "$NAV_ROOT/env/isaaclab3_sim6/uv.lock" "$NAV_LAB_ROOT/uv.lock"
cd "$NAV_LAB_ROOT"
uv python install 3.12
uv sync --python 3.12 --locked --extra isaacsim
export NAV_PY="$NAV_LAB_ROOT/.venv/bin/python"
cd "$NAV_ROOT"
if [ ! -e .venv ] && [ ! -L .venv ]; then
  ln -s ../IsaacLab-Protomotions/.venv .venv
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NAV_ROOT/src:$NAV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$NAV_PY" -I -B scripts/isaaclab3_environment.py check
"$NAV_PY" -m social_navigation status
```

Adjacent directories preserve the editable path. A different layout requires matching editable paths in both manifest and lock. For an existing environment, `apply --lab-root` rewrites this path.

If `uv sync --locked` fails, retain the error and inspect platform, revision, source paths and package-index access. Do not use `--frozen` to conceal inconsistency. NVIDIA wheels and extension caches can require substantial download time and disk space.

### First Kit use and the license setting

Importing Isaac Sim for the first time may prompt for NVIDIA's EULA. After reviewing and accepting the terms, the [official installation guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html) permits this setting in the terminal that will start Kit:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

The environment checker does not start Kit or accept the license automatically.

## 4. Reuse an existing environment

The local canonical checkout is `/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark`, next to `IsaacLab-Protomotions`. `ProtoMotions` is a local compatibility link; a new machine does not need that old directory name.

```bash
cd /absolute/path/to/Socially-aware_Indoor_Navigation_Benchmark
export NAV_ROOT="$PWD"
export NAV_LAB_ROOT="$(cd ../IsaacLab-Protomotions && pwd)"
export NAV_PY="$NAV_LAB_ROOT/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NAV_ROOT/src:$NAV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$NAV_PY" -m social_navigation status
```

Use `apply --sync` below only when applying the recorded dependency snapshot. It requires the exact IsaacLab revision and an existing `.venv` with `isaacsim==6.0.0.1`; it is not a fresh-environment bootstrapper.

```bash
git -C "$NAV_LAB_ROOT" rev-parse HEAD
"$NAV_PY" scripts/isaaclab3_environment.py apply --lab-root "$NAV_LAB_ROOT" --sync
"$NAV_PY" -I -B scripts/isaaclab3_environment.py check
uv pip check --python "$NAV_PY"
```

Previous manifests and locks are saved under IsaacLab's `.crowdsim-environment-backups/`. When migrating from two USD providers, this synchronization reinstalls the same usd-exchange version to restore shared files removed with usd-core. See [compatibility notes](dependencies.en.md) for the three known `uv pip check` overrides.

To rebuild only this project's editable metadata while retaining dependencies:

```bash
cd "$NAV_ROOT"
uv pip install --python "$NAV_PY" --no-deps --no-build-isolation -e .
"$NAV_PY" -m social_navigation --help
```

`--no-deps` does not install missing requirements. Use it only after establishing the pinned environment.

## 5. Source checks without SDK

Set `NAV_ROOT` to the cloned checkout. This environment installs inspection tools, not the full project's training dependencies.

```bash
cd "$NAV_ROOT"
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python pytest==9.0.3 setuptools==81.0.0
PYTHONPATH=src:. .venv-cpu/bin/python -m social_navigation status
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-cpu/bin/python -m pytest --import-mode=importlib -p no:cacheprovider \
  -q tests/test_project_layout.py
```

## 6. Simulator-free PPO baseline

Use a separate environment to keep CPU Torch out of the SDK environment. The selected Linux Python 3.12 wheel is available from the [official PyTorch CPU index](https://download.pytorch.org/whl/cpu/torch/).

```bash
cd "$NAV_ROOT"
uv venv --python 3.12 .venv-baseline
uv pip install --python .venv-baseline/bin/python torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-baseline/bin/python -e ./packages/crowdsim-ppo
.venv-baseline/bin/crowdsim-ppo-train --device cpu --total-steps 4096 \
  --output output/point_goal/new_run
.venv-baseline/bin/crowdsim-ppo-eval --device cpu \
  --checkpoint output/point_goal/new_run/latest.pt --episodes 20
```

This baseline does not control AMP pedestrians or load indoor USD scenes.

## 7. Download resources

Download only the groups you need. The second command requests larger upstream pretrained models.

```bash
cd "$NAV_ROOT"
git lfs pull --include='src/protomotions/data/assets/**'
git lfs pull --include='data/pretrained_models/**'
```

About 5.3 GB of upstream resources retain their original LFS references. ProtoMotions models differ from this project's TRACE/PACER/local fine-tuned weights. LFS pull does not provide local AMASS or the four-scene fine-tuned controllers.

Inspect pinned reference repositories and clone them when needed:

```bash
cd "$NAV_ROOT"
python3 scripts/reference_repositories.py
python3 scripts/reference_repositories.py --clone
```

`--clone` fetches the listed TRACE, PACER, FLUX and NavDP source checkouts. Existing matching checkouts are inspected, not reset. It does not install those older projects' runtime environments.

After completing [configuration and resource preparation](configuration.en.md), run:

```bash
"$NAV_PY" -m social_navigation preflight --scene YOHO
```

Proceed to [physical tests and recording](usage.en.md) only after resource checks pass. Package installation, preflight and simulation quality are separate results.

## 8. Installation troubleshooting

| Symptom | Check |
| --- | --- |
| Unsatisfiable `pytetwild` resolution | Use the stored manifest, Linux x86_64 environment restriction and `--extra isaacsim`; check for obsolete teleop/all extras |
| Cannot find `isaaclab==12.0.0` | Use pinned editable source rather than guessing a PyPI version from the product name |
| Missing `pxr` files or RECORD mismatch | Check the single USD provider; synchronize the profile to reinstall usd-exchange and rerun `check` |
| Missing `CrowdSim` or configs | Run from the checkout; set `PYTHONPATH=src:.`; retain compatibility links and editable source |
| OpenCV file collisions | The SDK uses headless OpenCV; avoid adding the GUI wheel |
| Package-index connection failure | Check NVIDIA, PyTorch and the recorded mirror endpoints; do not replace locked packages through arbitrary upgrades |
| Preflight fails on a new clone | Prepare the exact weights, AMASS bank, scene query files and hashes reported by the checker |
