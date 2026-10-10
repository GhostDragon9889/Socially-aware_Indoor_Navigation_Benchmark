# 安装指南

[English](installation.en.md) · [文档目录](README.md) · 更新：2026-10-10

## 1. 选择安装方式

| 目标 | 方式 |
| --- | --- |
| 查看源码、状态和布局测试 | 第 5 节，无需 Isaac SDK |
| 独立 PointGoal PPO | 第 6 节，单独 CPU 环境 |
| 复用本机物理行人环境 | 第 4 节，已有 `.venv` |
| 新机器准备物理环境 | 第 2–3 节，再准备资源 |

示例使用 Bash。将 `/absolute/path/to/...` 替换成实际位置；从新工作区操作时两个源码目录必须是相邻目录。本文描述安装流程，本次文档核验没有重新下载整套 SDK 或在新机器完成仿真。

## 2. 系统检查与克隆

先安装 Git、Git LFS、uv；录制时准备 ffmpeg/ffprobe。Ubuntu/Debian 的系统工具可由管理员安装，例如 `sudo apt install git git-lfs ffmpeg`。硬件和 GLIBC 条件见[依赖说明](dependencies.zh-CN.md)。

```bash
nvidia-smi
ldd --version
uv --version
git lfs version
ffmpeg -version
ffprobe -version
```

`nvidia-smi`、`ffmpeg` 不是纯 CPU 状态检查的必要条件。私有 GitHub 仓库需要授权账号的 Git 凭据；不要把访问令牌写进 URL 或配置文件。

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

`verify_upstream_integration.py` 核对 1,017 个上游文件的 Git 映射，不下载 LFS，不检查私有数据。LFS 下载来源已设为 NVlabs/ProtoMotions；资源按需下载见第 7 节。

## 3. 新建固定 SDK 环境

仅在全新的 `NAV_WORKSPACE` 执行。clone 创建独立 IsaacLab 工作区；把本仓库快照复制到该工作区，随后由 uv 创建环境。不要在 `env/isaaclab3_sim6/` 中执行同步，也不要用 PyPI 的任意 IsaacLab 版本替换固定源码。

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

这里采用相邻目录，因此 editable 路径无需修改。若调整目录位置，必须同步修改 manifest 和 lock 中的 editable 路径；已有环境可用 `apply --lab-root` 自动改写这项路径。

`uv sync --locked` 失败时应保留错误，检查平台、源码提交、源码路径和索引连接，不使用 `--frozen` 隐藏 manifest/lock 不一致。首次获取 NVIDIA wheel 和扩展缓存可能需要较长时间及较大磁盘空间。

### 首次使用 Kit 的许可设置

首次导入 Isaac Sim 可能要求接受 NVIDIA EULA。按 [官方说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)，阅读并接受条款后，可在需要启动 Kit 的终端设置：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

环境检查脚本自身不启动 Kit，也不自动接受许可。

## 4. 复用已有环境

本机 canonical 项目位于 `/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark`，SDK 位于相邻 `IsaacLab-Protomotions`。`ProtoMotions` 是本机兼容链接，新机器无需使用旧目录名。

```bash
cd /absolute/path/to/Socially-aware_Indoor_Navigation_Benchmark
export NAV_ROOT="$PWD"
export NAV_LAB_ROOT="$(cd ../IsaacLab-Protomotions && pwd)"
export NAV_PY="$NAV_LAB_ROOT/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NAV_ROOT/src:$NAV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$NAV_PY" -m social_navigation status
```

只有需要应用已记录的依赖快照时才执行下面的 `apply --sync`；它要求精确 IsaacLab 提交及已安装 `isaacsim==6.0.0.1` 的 `.venv`，不是新环境引导器。

```bash
git -C "$NAV_LAB_ROOT" rev-parse HEAD
"$NAV_PY" scripts/isaaclab3_environment.py apply --lab-root "$NAV_LAB_ROOT" --sync
"$NAV_PY" -I -B scripts/isaaclab3_environment.py check
uv pip check --python "$NAV_PY"
```

旧 manifest/lock 保存在 IsaacLab 的 `.crowdsim-environment-backups/`。从双 USD 提供者迁移时，该同步会重装同版 `usd-exchange`，恢复卸载 `usd-core` 后共享目录被移除的文件。`uv pip check` 的三项已知覆盖见[兼容说明](dependencies.zh-CN.md)。

仅重建本项目 editable 安装元数据时，沿用已安装依赖：

```bash
cd "$NAV_ROOT"
uv pip install --python "$NAV_PY" --no-deps --no-build-isolation -e .
"$NAV_PY" -m social_navigation --help
```

`--no-deps` 不会补齐缺失依赖；它只适用于已经建立的固定环境。

## 5. 无 SDK 的源码检查

定义 `NAV_ROOT` 为已克隆目录。下面环境只装检查工具，不安装整个项目的训练依赖。

```bash
cd "$NAV_ROOT"
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python pytest==9.0.3 setuptools==81.0.0
PYTHONPATH=src:. .venv-cpu/bin/python -m social_navigation status
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-cpu/bin/python -m pytest --import-mode=importlib -p no:cacheprovider \
  -q tests/test_project_layout.py
```

## 6. 无模拟器的 PPO 基线

使用单独环境，避免向 SDK 环境安装 CPU Torch。这里选择的 Linux Python 3.12 CPU wheel 可从 [PyTorch 官方索引](https://download.pytorch.org/whl/cpu/torch/)获取。

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

该基线不是 AMP 行人控制器，也不加载室内 USD。

## 7. 下载与资源准备

只下载要使用的资源组；第二条会请求上游预训练模型，体积较大。

```bash
cd "$NAV_ROOT"
git lfs pull --include='src/protomotions/data/assets/**'
git lfs pull --include='data/pretrained_models/**'
```

约 5.3 GB 上游资源保持原始 LFS 引用。公开 ProtoMotions 权重与本项目 TRACE/PACER/本地微调权重是不同资源；LFS pull 不会补齐本地 AMASS 或四场景自训权重。

查看固定参考仓库，按需获取：

```bash
cd "$NAV_ROOT"
python3 scripts/reference_repositories.py
python3 scripts/reference_repositories.py --clone
```

`--clone` 会取得清单中的 TRACE、PACER、FLUX 和 NavDP 源码，已有匹配目录只核对，不重置。它不安装这些旧项目各自的运行环境。

完成[配置与资源准备](configuration.zh-CN.md)后，运行：

```bash
"$NAV_PY" -m social_navigation preflight --scene YOHO
```

只有资源核验成功后才进入[物理测试与录像](usage.zh-CN.md)。安装通过、预检通过和仿真质量通过分别记录。

## 8. 常见安装问题

| 现象 | 排查方式 |
| --- | --- |
| `pytetwild` 解析无解 | 确认使用项目快照、Linux x86_64 环境限制和 `--extra isaacsim`，没有旧 teleop/all extras 混入 |
| 找不到 `isaaclab==12.0.0` | 使用固定源码 editable 源，不从 PyPI 猜测产品版本 |
| `pxr` / USD 文件缺失或 RECORD 不匹配 | 检查唯一提供者；用 profile 同步重装 usd-exchange，再运行 `check` |
| 包安装后找不到 `CrowdSim` / 配置 | 在源码根运行；设置 `PYTHONPATH=src:.`，确认兼容链接与 editable 源码存在 |
| OpenCV 文件冲突 | 专用 SDK 使用 headless；不要额外混装 GUI wheel |
| 下载索引不可达 | 检查官方 NVIDIA、PyTorch 及快照所记录镜像的连接；不要通过任意升级替换锁定包 |
| 新 clone 的 preflight 失败 | 按错误列出的权重、AMASS、场景图及哈希依次补齐，见配置指南 |
