# Technology Stack
审查：2026-10-10；基于本机文件、安装元数据、AST 和 SHA。此次未启动 Kit、训练或物理重测。完整机器清单见 [inventory.json](../resource_audit_20261010/inventory.json)。

## Core Sections (Required)

### 1) Runtime Summary

物理行人链运行于 Linux x86_64、Python 3.12.13、Isaac Sim 6.0.0.1，以及 `IsaacLab-Protomotions` 的产品 VERSION 3.0.0 / Python package 12.0.0。根 `.venv` 是外部 SDK 的符号链接。独立 PointGoal PPO 只要求 Python ≥3.10、NumPy ≥1.24、Torch ≥2.2。两条链的安装范围不同。[来源](../../packages/crowdsim-ppo/pyproject.toml)

本机 GPU 为 RTX 4080 SUPER，驱动 595.99.02，显存 16,376 MiB；读取设备元数据未执行 CUDA 运算。系统 FFmpeg/FFprobe 为 6.1.1。SDK lock 有 273 项：257 registry、14 editable、1 Git、1 virtual。272 个非 virtual 项的版本均匹配本机；uv 检查 272 包。去掉继承 ROS PYTHONPATH 后 metadata 搜索返回 274 条，其中本项目 src egg-info/dist-info 同版重复，另有独立 crowdsim-ppo；不能把每条 metadata 都解释为新增 SDK 依赖。

### 2) Production Frameworks and Dependencies

根 core 共18项，完整声明：

| Requirement |
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

实际关键版本（不是推荐升级列表）：

| Distribution | 本机版本 |
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
| `warp-lang` | `1.15.0.dev20260626` |
| `newton` | `1.4.0.dev0` |
| `newton-usd-schemas` | `0.5.0` |
| `moviepy` | `1.0.3` |
| `opencv-python-headless` | `4.13.0.90` |
| `pillow` | `12.1.1` |
| `smplx` | `0.1.28` |
| `chumpy` | `0.70` |
| `einops` | `0.9.0.dev0` |
| `pytest` | `9.0.3` |

| Extra | 声明条数 | 用途 / 边界 |
| --- | ---: | --- |
| `crowd` | 2 | Pillow/OpenCV；SDK 使用 headless provider |
| `mujoco` | 5 | MuJoCo 后端，需单独验收 |
| `newton` | 10 | Newton 1.0/不同 Warp 组合，单独环境 |
| `isaacgym` | 0 | 空组；不安装 IsaacGym SDK |
| `isaaclab` | 8 | 固定 Sim 6.0.0.1 / Lab package 12.0.0 |
| `genesis` | 9 | Torch 2.5，单独环境 |
| `docs` | 5 | Sphinx 文档构建 |
| `dev` | 6 | 测试、ONNX、构建及发布工具 |

TRACE 的 `einops`、Torchvision、AMP 皮肤录制的 `smplx/chumpy` 由 SDK manifest 明确补充；根 core/isaaclab extra 本身不完整声明这些功能依赖。`shapely/rvo2` 仅特定 ORCA 生成路径使用，当前环境未安装；已有预生成语料与四场景推理不需要重新生成 ORCA。JAX/pyroki/viser/yourdfpy、bvh、soma 属于可选转换链，不能把其缺失归为现有四场景缺权重。[导入证据](../resource_audit_20261010/inventory.json) 包含文件及行号。

### 3) Development Toolchain

setuptools≥69.5.1 + wheel 构建 src 包；pytest 9.0.3 已安装。`.pre-commit-config.yaml` 固定 Ruff hook 0.8.1、typos 1.28.1、通用 hooks 5.0.0，但本机未安装 pre-commit/Ruff/Sphinx 发行包。docs extra 和 `docs/requirements.txt` 存在额外插件差异，构建 Sphinx 时应核对后者。公共 CI 为源码语法、项目布局和上游映射检查，不启动 SDK。

### 4) Key Commands

在仓库根、既有 SDK Python 下：

```bash
.venv/bin/python -m social_navigation status
.venv/bin/python -m social_navigation preflight --scene YOHO
.venv/bin/python -B -I scripts/isaaclab3_environment.py check
uv pip check --python .venv/bin/python
python scripts/manage_release_assets.py verify
```

`check` 当前通过；`pip check` 非零，确切三项是 Sim metadata 要求 Newton1.2.0/schema0.2.0/coverage7.4.4，而锁定安装为1.4.0.dev0/0.5.0/7.6.1。该覆盖记录不代表其他新冲突也可接受。不要把 `--all-extras` 用于 SDK：五个 simulator extras 有10组显式互斥关系。

### 5) Environment and Config

SDK manifest/lock 需放在 IsaacLab checkout 根使用，根项目没有 uv.lock。实际 SDK HEAD `5951d48dde083986489957a7b108ad48a46822a7` 与 profile 的 `4ecd0b...` 不同；此次确认 `source/isaaclab` 的提交间 diff 为空、manifest/lock SHA一致，但 `apply` 的 require_checkout 仍失败。不能把包版本检查通过写成源码 revision 检查通过。

`/project/EXP/IsaacLab` 为 Sim6.1.0.0、isaaclab17.0.2、TensorDict0.13、NumPy2.5.1，7项 core包缺失、2项 core版本越界；也不满足现有 isaaclab extra 固定版本。需独立迁移 profile 与原生接口验收。详见[资源与依赖总览](../RESOURCE_DEPENDENCY_MAP_20261010_ZH.md)。

### 6) Evidence

- [实际环境、文件身份、依赖和目录清单](../resource_audit_20261010/inventory.json)
- [本轮四场景预检与发布资源校验](../resource_audit_20261010/resource_checks.json)
- [环境/USD 检查与版本覆盖诊断](../resource_audit_20261010/environment_checks.json)

- [根依赖与入口](../../pyproject.toml)、[SDK manifest](../../env/isaaclab3_sim6/pyproject.toml)、[lock](../../env/isaaclab3_sim6/uv.lock)、[profile](../../env/isaaclab3_sim6/profile.json)
- [环境检查实现](../../scripts/isaaclab3_environment.py)、[pre-commit](../../.pre-commit-config.yaml)、[CI](../../.github/workflows/public-ci.yml)、[Sphinx requirements](../requirements.txt)

## Clone portability update — 2026-10-10

The current four-scene flow uses `configs/reproduction/four_scene`, exact in-memory resource bindings, the committed SDK bootstrap/native shutdown bundle, fixed-bank rebuild templates and `scripts/run_four_scene_benchmark.py`. See [clone reproduction](../../CLONE_REPRODUCTION.en.md) and [validation](../CLONE_VALIDATION_20261010.json). Historical experiment receipts remain provenance, not new-run configuration.
