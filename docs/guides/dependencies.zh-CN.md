# 项目依赖

[English](dependencies.en.md) · [文档目录](README.md) · 更新：2026-10-10

## 1. 依赖的权威来源

| 文件 | 含义 |
| --- | --- |
| [根 pyproject.toml](../../pyproject.toml) | 项目发行包、核心要求、可选后端和控制台入口 |
| [SDK manifest](../../env/isaaclab3_sim6/pyproject.toml) | 应用于 IsaacLab 源码根目录的完整环境配置 |
| [SDK uv.lock](../../env/isaaclab3_sim6/uv.lock) | 273 个锁定包及其来源、wheel 哈希 |
| [profile.json](../../env/isaaclab3_sim6/profile.json) | 平台、源码提交、版本及 manifest/lock 哈希 |
| [独立 PPO 包](../../packages/crowdsim-ppo/pyproject.toml) | 无模拟器的 NumPy/PyTorch 导航基线 |
| [完整依赖清单](dependency-inventory.json) | 上述声明与全部 273 个锁定包的可读快照 |

根目录依赖范围不是精确环境锁。`requirements_isaaclab.txt` 是补充列表，也不是锁；复用物理行人环境时以 SDK manifest、lock 和 profile 三者为准。

## 2. 固定物理行人环境

| 项目 | 固定值 |
| --- | --- |
| 平台 | Linux x86_64 |
| Python | 3.12；本机记录 3.12.13 |
| IsaacLab 源码 | `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8` |
| IsaacLab 产品 / Python 包版本 | `3.0.0` / `12.0.0` |
| Isaac Sim Python 包 | `6.0.0.1` |
| 本项目版本 | `3.1.0` |

该 IsaacLab 提交属于开发线，不等于 `v3.0.0-beta2` tag。官方 [beta2](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2)支持 Sim 6.0，但安装本项目时仍须使用精确提交。[3.0 EA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)要求 Sim 6.1，不能仅依据“3.0”名称替换本环境。


### 本机实际状态补充（2026-10-10）

本轮核对 SDK manifest/lock SHA 与存档相同，272 个非 virtual 锁定版本全部匹配；18 项项目 core 约束满足，`uv pip check` 仍报告下述3项覆盖冲突。当前 SDK HEAD 已是 `5951d48dde083986489957a7b108ad48a46822a7`（依赖配置提交），profile 仍记录 `4ecd0b...`；两提交之间只改变 manifest/lock，SDK source 未改变，但 `apply` 的精确 commit 门禁会拒绝当前 HEAD。`check` 的版本、234 个 USD RECORD 文件及 standalone round-trip 验证通过。不要把两种检查混写为同一结论。

`/project/EXP/IsaacLab` 的 Sim6.1/package17.0.2 与此环境不同，其项目 core 尚缺7包并有2项版本越界。迁移问题及实际资源缺口见[全面审查](../RESOURCE_DEPENDENCY_MAP_20261010_ZH.md)和[机器清单](../resource_audit_20261010/inventory.json)。

### 主要锁定版本

以下值直接来自仓库 lock；不表示已在任意机器安装或验证。

| 包 | 锁定版本 |
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

## 3. 核心 Python 要求

| 根 pyproject.toml 中的声明 |
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

主要职责：Torch/TensorDict/Lightning 用于策略、训练与张量；NumPy/SciPy/图像库用于数据和评估；MuJoCo/dm-control/Trimesh/Rtree 用于动作、几何和转换；OmegaConf/PyYAML 用于配置；MoviePy/图像工具用于视频；W&B/TensorBoardX 用于日志。

## 4. 可选依赖与环境分工

| 安装项 | 用途 | 使用方式 |
| --- | --- | --- |
| `isaaclab` | 本项目物理行人后端 | 先建立固定 SDK 环境，再 editable 安装源码 |
| `crowd` | Pillow 和 GUI 版 OpenCV 辅助工具 | 专用 SDK lock 已选择 headless OpenCV；不要直接混装两种 OpenCV 提供者 |
| `dev` | pytest、ONNX 导出、构建和发布工具 | 按实际开发任务选取 |
| `docs` | 上游 Sphinx 文档工具 | 本组双语指南本身是 Markdown |
| `mujoco` / `newton` / `genesis` | 上游其他模拟后端 | 使用各自独立环境和 requirements；本次四场景不验证这些后端 |
| `isaacgym` | 上游旧后端接口，extra 为空 | 不能通过该 extra 获得 IsaacGym SDK |
| `packages/crowdsim-ppo` | PointGoal PPO 基线 | 仅 NumPy ≥1.24、Torch ≥2.2；Python ≥3.10 |

不要在专用 SDK 环境执行根目录 `uv sync --all-extras`。后端有互斥项及不同 Torch 约束；SDK 的完整快照只在 IsaacLab 工作区同步。

## 5. 必须保留的兼容设置

- 独立 Python 的 USD 提供者为 `usd-exchange==3.0.0`；`usd-core` 被排除。两者覆盖相同 `pxr` 文件，官方 [usd-exchange 说明](https://pypi.org/project/usd-exchange/3.0.0/)要求避免共装。
- 独立 USD 是 `26.8`；历史本机 Kit 加载 `25.11`。`check` 验证的是独立进程，不应把该版本断言套到 Kit 进程。
- manifest 覆盖 Newton、schemas 和 coverage。`uv pip check` 已知会报告三项与 Sim 元数据的差异：`newton==1.4.0.dev0`、`newton-usd-schemas==0.5.0`、`coverage==7.6.1`。记录并比对差异；新增冲突需要另行排查，不要把全部失败当作允许项。
- Linux x86_64 的 `tool.uv.environments` 限制与正确的平台条件用于避免 `pytetwild` 的跨平台不可能分支。保留仓库的 manifest，不通过删除依赖解决冲突。
- `--locked` 检查 manifest 与 lock 是否一致；`--frozen` 跳过这项检查，不适合作为修复冲突的方法。见 [uv 官方同步说明](https://docs.astral.sh/uv/concepts/projects/sync/)。

## 6. 系统工具和外部资源

需要 Git/Git LFS、uv、可用 NVIDIA 驱动和支持目标 SDK 的 GPU；录像需要命令行 `ffmpeg` 与 `ffprobe`，只有 Python `imageio-ffmpeg` 不等于已具备这两个系统命令。按 [Sim 6.0 系统要求](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)核对硬件；pip 环境要求 Python 3.12 和 Linux GLIBC ≥2.35，见 [官方安装说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)。

AMASS、SMPL 模型、TRACE/PACER 权重、场景 USD/材质、原生查询地图和历史执行证据都不是 pip 依赖。克隆源码和安装依赖后，继续按[资源与配置](configuration.zh-CN.md)准备。

公开 PACER 下载器还使用 `gdown==6.2.0` 与 `requests`；它们不在专用 SDK 的保证范围内，建议使用独立下载环境。数据和模型许可位置见根 `LICENSE.md`、`legal/` 和参考仓库的许可证。

下一步：[安装](installation.zh-CN.md) → [配置](configuration.zh-CN.md) → [运行](usage.zh-CN.md)。
