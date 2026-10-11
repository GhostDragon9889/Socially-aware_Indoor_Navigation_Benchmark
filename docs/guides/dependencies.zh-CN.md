# 项目依赖

[English](dependencies.en.md) · [文档目录](README.md) · 更新：2026-10-11

## 默认环境：IsaacLab 3.0 EA / Isaac Sim 6.1

[官方 EA 文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html)与本项目固定源码、锁文件共同定义目标环境。产品名、Python 包版本及源码提交需要分别核对。

| 项目 | 固定值 |
| --- | --- |
| 平台 / Python | Linux x86_64 / 3.12.13 |
| IsaacLab 源码提交 | `ae37b028ea415c91ea2bc32609efcd759ed2b974` |
| IsaacLab 源码树 | `27e5b57ad0928e0339e121a06be040ae234a99ed` |
| 产品 VERSION / Python 包 | `3.0.0` / `isaaclab==17.0.2` |
| Isaac Sim 包 | `6.1.0.0` |
| Torch / NumPy / Warp | `2.11.0+cu128` / `2.5.1` / `1.16.0` |
| 独立 USD 提供者 | `usd-exchange==2.3.0`；USD `25.5` |
| 项目运行解释器 | `.runtime/external_sdk.json` 的 `python` |

## 权威文件与环境边界

| 文件 | 用途 |
| --- | --- |
| [根 pyproject.toml](../../pyproject.toml) | 核心依赖范围、五个源码包及可选后端 |
| [SDK manifest](../../env/isaaclab3_ea_sim61/pyproject.toml)、[SDK uv.lock](../../env/isaaclab3_ea_sim61/uv.lock) | 新建固定 SDK 环境的完整快照 |
| [profile.json](../../env/isaaclab3_ea_sim61/profile.json) | 源码、二进制提供者版本及 manifest/lock 哈希 |
| [完整锁定依赖清单](dependency-inventory-isaaclab3-ea-sim61.json) | 2026-10-10 旧 overlay 依赖快照；当前附加包以 lock 为准 |
| [requirements.overlay.lock](../../env/isaaclab3_ea_sim61/requirements.overlay.lock) | 33 项精确项目补充依赖 |
| [运行环境说明](../../env/isaaclab3_ea_sim61/README.md) | 已有 SDK 复用、安装与只读验证 |
| [约束收据](../../env/isaaclab3_ea_sim61/runtime_constraints_receipt.txt) | 本机实测可见包版本记录；不是独立 SDK 安装锁 |

项目直接使用外部 SDK 的 Python；也可在 `.runtime/IsaacLab61` 完整安装 SDK。33 项补充包安装到 `.runtime/external_sdk_packages`，无需另建项目解释器。配置检查提供者来源、169 项 USD RECORD 文件、补充包依赖闭合及项目核心要求，不向 SDK 安装或覆盖 Torch、NumPy、Warp、Isaac 和 USD 包。运行使用 `scripts/run_external_python.py`；旧 overlay 仅供已有部署重放。详见[外部 SDK 指南](external_sdk.zh-CN.md)。

## 项目补充依赖

| 包 | 精确版本 | 用途 |
| --- | --- | --- |
| `omegaconf` / `hydra-core` | `2.3.0` / `1.3.2` | 本项目配置及已保存配置兼容 |
| `tensordict` | `0.9.0` | 策略张量与批数据 |
| `lightning` / `pytorch-lightning` | `2.5.6` / `2.5.6` | 训练入口 |
| `wandb` / `protobuf` | `0.23.0` / `6.33.6` | 训练日志与 SDK 兼容 |
| `dm-control` / `dm-env` | `1.0.44` / `1.6` | MuJoCo 动作、运动学处理 |
| `smplx` / `chumpy` | `0.1.28` / `0.70` | 许可 SMPL 模型读取及蒙皮 |
| `scikit-image` | `0.26.0` | 图像与独立评估 |
| `moviepy` / `imageio-ffmpeg` | `1.0.3` / `0.6.0` | 视频处理 |
| `easydict` / `termcolor` | `1.13` / `3.3.0` | 参考控制器配置和日志 |
| `tensorboardX` | `2.6.5` | ProtoMotions SummaryWriter |
| `pytest` | `9.0.3` | CPU 功能测试 |

完整列表以 overlay lock 为准。`dm-control==1.0.40` 在目标 MuJoCo 3.11 上真实出现 `flex_bandwidth` 缺失；改为 [官方 1.0.44](https://github.com/google-deepmind/dm_control/releases/tag/1.0.44) 后完成 CPU 接触、步进、复制、重置回归。不能以 import 成功代替该数值/API 检查。

## 物理与资产合约

- 明确保持旧 AMP 动力学：`use_newton_actuators=False`、`enable_external_forces_every_iteration=False`。
- 新转换器根 USD 默认 `Physics=physics`；行人 spawn 和路径检查显式选择 `physx` 图层，保留 SMPL 24 bodies / 69 DOFs。
- Warp、ProxyArray 与 Torch 之间使用显式张量视图；写状态和目标使用当前官方 `_index` 接口。
- [历史上游注册表](../../data/pretrained_models/upstream_reference_registry.json)仅将两份旧 NavIsaacLab 分支配置列为归档、不支持当前运行。其权重/sidecar 原字节保留；现行 TRACE/PACER 与本地微调权重不在该排除范围内。
- 本轮完成新 SDK 四场景短时物理、256 环境训练及恢复，实测范围见[核验报告](../EXTERNAL_SDK_VERIFICATION_20261011_ZH.md)；短时成功不能转述成完整 60 秒质量、视频或论文效果验证。

## 系统依赖与其他后端

Git/Git LFS、uv、NVIDIA 驱动和匹配 GPU 按 [Isaac Sim 6.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)准备。录制还需要系统 `ffmpeg`、`ffprobe`；Python 视频包不等于这两个命令已可用。AMASS、SMPL 参数、权重、场景及材质属于外部资源，见[配置](configuration.zh-CN.md)。

默认 extra `isaaclab` 对应新环境；`isaaclab6` 对应旧 Lab12/Sim6.0。[旧环境快照](../../env/isaaclab3_sim6/README.md)、历史依赖清单与实验结果仅用于重放旧记录。不要在 SDK 或项目 runtime 中执行根目录 `uv sync --all-extras`；MuJoCo、Newton、Genesis、IsaacGym 与独立 PPO 有各自环境边界。

根安装元数据支持 Python3.12，并将 uv 全平台解析限定为受支持的 Linux x86_64。两个 Lab 包通过官方 Git 子目录和固定提交取得；Sim6.1 明确声明必需的 `tinyobjloader==2.0.0rc13`。六种仿真后端的每一对 extra 均互斥，官方双 Git 来源在 uv 默认预发布策略下真实解析通过；dry-run 不代表安装或动力学验收。核心 MoviePy 范围允许已有1/2写视频接口，本运行环境仍精确固定1.0.3。历史独立 `newton[sim]==1.0.0` extra 使用 Warp1.12，须与 SDK 内置 Newton1.5.2 / Warp1.16 分开。

下一步：[安装](installation.zh-CN.md) → [配置](configuration.zh-CN.md) → [运行](usage.zh-CN.md)。
