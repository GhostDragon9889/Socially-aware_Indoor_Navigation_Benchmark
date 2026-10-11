# 从 Git 克隆复现四场景行人模拟

[English](CLONE_REPRODUCTION.en.md) · 2026-10-10

## 交付范围

当前四场景主流程的算法、训练工具、仿真适配、检查工具、录制工具、配置、锁文件和原生退出补丁均由本项目管理。无需复制 `/project/EXP/NavIsaaclab2.0`、外部 TRACE/PACER 工作区或聊天工作区中的代码。已发布权重、机器人及四场景模型由项目的私有 Git LFS 资源包恢复。

| 内容 | 项目位置 |
| --- | --- |
| Diffusion、AMP、行人世界、地图与物理适配 | `src/CrowdSim/humanoid/` |
| ProtoMotions、IsaacLab 后端与资产转换 | `src/protomotions/` |
| 项目入口、路径迁移、资源恢复 | `src/social_navigation/` |
| 独立 PointGoal PPO 基线 | `src/crowdsim_ppo/` |
| 微调、独立评估、录制、AMASS 转换入口 | `scripts/`、`data/scripts/` |
| 当前四场景配置、相机与资源绑定 | `configs/reproduction/` |
| AMASS 步态先验与已完成微调配方 | `configs/diffusion_amp/four_scene_speed_20261010/` |
| 默认 SDK 与精确项目 overlay | `env/isaaclab3_ea_sim61/`；历史 `env/isaaclab3_sim6/` |
| 权重、恢复训练检查点、EVT 机器人、四场景 | `resources/releases/pedestrians_20261010/` |
| 人体 UV 与有来源许可的纹理 | `resources/pedestrian_visuals/` |

需要自行获取 **AMASS 原始动作**及受许可约束的 **SMPL 参数文件**；它们不属于可随源码分发的资源。Isaac Sim/IsaacLab、CUDA 驱动、Git LFS、uv、ffmpeg 是软件安装依赖。既有四场景已随项目提供，新场景可另外下载并重新生成地图。

本指南复现的是已经实现的 TRACE Diffusion + 本地 AMASS 微调 PACER AMP 行人模拟。它包含明确记录的方法调整；不是作者未公开权重、动作清单及完整八场景协议的等价复现。EVT 模型已提供，但当前四场景视频入口尚未实现完整机器人社交导航评测。`docs/style_improvement_*` 中的历史试验、恢复状态及诊断记录不作为新运行的输入；不能把历史运行脚本均视为经过维护的端到端入口。

## 1. 克隆并安装固定 SDK

Linux x86_64、支持 Isaac Sim 6 的 NVIDIA GPU。先安装 Git、Git LFS、uv；录制需要 ffmpeg/ffprobe。私有仓库需登录具有访问权限的 GitHub 账号。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark.git
cd Socially-aware_Indoor_Navigation_Benchmark
export NAV_ROOT="$PWD"
git lfs install --local
uv python install 3.12.13
uv run --no-project --python 3.12.13 python scripts/audit_checkout.py
uv run --no-project --python 3.12.13 python scripts/bootstrap_reproduction.py install --profile isaaclab3_ea_sim61
export NAV_PY="$NAV_ROOT/.runtime/isaaclab3_ea_sim61/.venv/bin/python"
export PYTHONPATH="$NAV_ROOT/src:$NAV_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export NAV_LAB="$NAV_ROOT/.runtime/IsaacLab61"
"$NAV_PY" -m social_navigation.runtime_environment check --lab-root "$NAV_LAB"
"$NAV_PY" -I scripts/isaaclab3_environment.py --profile isaaclab3_ea_sim61 check
```

引导器默认固定 IsaacLab 源码 `ae37b028ea415c91ea2bc32609efcd759ed2b974`、Python 3.12.13、Sim 6.1.0.0、IsaacLab 产品 3.0.0（包 17.0.2）、Torch 2.11.0+cu128。SDK 位于 `.runtime/IsaacLab61`，项目补充依赖位于独立 `.runtime/isaaclab3_ea_sim61/.venv`；不修改或重复安装已装 SDK 源码和二进制提供者。按 NVIDIA 条款接受 EULA 后再启动 Kit；必要时设置 `OMNI_KIT_ACCEPT_EULA=YES`。不要在存档 profile 目录直接运行 `uv sync`。

复用匹配 SDK 时运行 `python3 scripts/bootstrap_reproduction.py prepare-runtime --profile isaaclab3_ea_sim61 --lab-root /absolute/path/to/IsaacLab`，再选取输出的项目解释器。本机已真实安装并验证独立 overlay；完整新 SDK 网络安装未在本轮重复下载。详见[运行环境](env/isaaclab3_ea_sim61/README.md)与[依赖说明](docs/guides/dependencies.zh-CN.md)。已有四场景视频保持历史 Sim6.0 身份；新 SDK 的 60 秒物理、录制和质量验收须独立保存证据，不能由迁移/import 测试推导。

## 2. 恢复项目内的 LFS 资源

```bash
"$NAV_PY" scripts/manage_release_assets.py fetch
"$NAV_PY" scripts/manage_release_assets.py prepare --check-usd
# 仅获取 SMPL 刚体网格和 AMASS split 清单，保留上游 LFS 引用。
git -c lfs.url=https://github.com/NVlabs/ProtoMotions.git/info/lfs \
  lfs pull --include='src/protomotions/data/assets/mesh/smpl/**,data/yaml_files/amass_smpl_*.yaml,src/protomotions/data/assets/usd/terrain_material.usda' --exclude=
```

`fetch` 从 URL 自动识别私有项目远程，兼容新克隆的 `origin` 和旧工作区的 `benchmark`。`prepare` 校验 467 个文件及 14 份原始 USD 快照，恢复模型、固定动作清单、人体纹理和精确的已测场景层。对旧绝对路径的处理只发生在内存视图和自有匿名 USD 副本中；保留原记录、几何、物理参数及哈希。未注册或缺失的依赖会报错。

## 3. 准备 AMASS 和 SMPL

从 [AMASS](https://amass.is.tue.mpg.de/) 获取对应 SMPL 动作集，并按原文件名放入 `<dataset>/raw/*.tar.bz2`。转换器中的 `ARCHIVES` 映射和已提交的 `data/yaml_files/amass_smpl_{train,validation,test}.yaml` 定义所需集合及片段，不使用另一版本动作替代缺失文件。转换会拒绝不匹配的归档目录、重复动作或未验证的已有输出。

```bash
export NAV_AMASS=/absolute/path/to/AMASS
"$NAV_PY" data/scripts/prepare_amass_dataset.py "$NAV_AMASS" \
  --stage prepare --splits train validation test --workers 4
"$NAV_PY" scripts/rebuild_fixed_amass.py --source-bank "$NAV_AMASS/amass_smpl_train.pt"
"$NAV_PY" scripts/rebuild_fixed_amass.py --split validation --source-bank "$NAV_AMASS/amass_smpl_validation.pt"
"$NAV_PY" scripts/rebuild_fixed_amass.py --split test --source-bank "$NAV_AMASS/amass_smpl_test.pt"
```

若已有由本仓库转换器生成的通用 `.pt`，可直接执行后三条。最终文件位于 `data/datasets/amass/`。重建固定 **2,848 / 22 / 34** 个训练/验证/测试片段，保留原动作顺序、截取范围、标签、指纹和 child-joint 角速度约定；验证/测试库保持完整通用库，片段选择由清单定义。最终数据包必须符合原完整 SHA：

| 数据包 | SHA256 |
| --- | --- |
| `manifest_labeled_v7_train.pt` | `46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0` |
| `amass_smpl_validation.pt` | `ae0389770ad4a1dfb6d916b80bbe88b8ff89c7ab80a22d71b77597b6c302cf03` |
| `amass_smpl_test.pt` | `ae3e55e7ab6934919f769e95a141fa1e97b0d305d1b96ce7503ea8fff49d9f23` |

已验证从本地通用动作库重建固定训练库。原始归档的全量重新转换未在本次重跑；不同源版本或数值结果会被完整哈希检查拒绝，不会自动降低要求。固定清单是本地真实 AMASS 筛选，作者未公开同一份动作 ID。

从 [SMPL](https://smpl.is.tue.mpg.de/) 按许可获取中性模型，将 `SMPL_NEUTRAL.pkl` 放在 `data/smpl/`。UV/纹理由步骤 2 恢复。物理刚体使用项目 MJCF/STL，不需要外部动作生成代码；参数模型用于人体蒙皮及脚部/画面检查。

可以通过忽略的 `data/external_resources.json` 挂载已有动作库，无需复制数据：

```json
{
  "amass_train": "/dataset/manifest_labeled_v7_train.pt",
  "amass_validation": "/dataset/amass_smpl_validation.pt",
  "amass_test": "/dataset/amass_smpl_test.pt"
}
```

挂载只改变文件位置，不改变内容校验。

## 4. 检查、仿真及录制

```bash
"$NAV_PY" scripts/audit_checkout.py --require-resources --output output/checkout_audit.json
"$NAV_PY" scripts/run_four_scene_benchmark.py --preflight-only --output output/four_preflight
# 四场景各 1800 个 30Hz 控制帧，完成独立检查后录制各 60 秒。
"$NAV_PY" scripts/run_four_scene_benchmark.py --record --output output/four_scene_run
```

可用 `--scenes SecondFloor` 先运行一个场景。不录制时去掉 `--record`。单独的短物理检查：`"$NAV_PY" -m social_navigation simulate --scene SecondFloor --steps 60 --output output/smoke`；短检查不替代 60 秒姿态/路线验收。

结果在 `output/four_scene_run/<scene>/`：物理状态、配置与源码快照、退出码、独立姿态/速度/接触/脚部/停滞/120Hz 速度检查；视频位于 `videos/<scene>_60s.mp4`。录制保留真实测量姿态和时间，不做姿态插值或补帧。若质量门未通过，流程保存失败信息并停止，不生成合格结论。全部新输出必须使用新目录。

## 5. 继续训练

已完成配方和权重链均由项目提供；无需外部聊天日志：

```bash
export NAV_CFG="$NAV_ROOT/configs/diffusion_amp/four_scene_speed_20261010"
"$NAV_PY" -m social_navigation train-amp \
  --output output/amp_new --updates 128 --num-envs 160 \
  --parent-checkpoint data/pretrained_models/local_amass_finetune/walking_turn_20261009/Humanoid_AMASS.pth \
  --optimization-config "$NAV_CFG/locomotion_conservative128_v3.json" \
  --gait-prior "$NAV_CFG/gait_conservative_prior_v3.json" \
  --temporal-gait-prior "$NAV_CFG/temporal_speed_prior_v1.json" \
  --walking-turn-windows "$NAV_CFG/walking_conservative_windows_v3.json" \
  --task-horizons-s 0.5 1.0 2.0 4.5
```

新权重需重新独立评估并在另建 profile 中绑定 SHA，不能自动替换已测模型。Diffusion 研究训练入口为 `scripts/train_paper_diffusion.py`（AMASS 清单）和 `scripts/train_orca_diffusion.py`（另需 ORCA 数据清单）；两者与四场景所用公开 TRACE 权重的训练协议不同。详见[使用说明](docs/guides/usage.zh-CN.md)。

## 检出验证记录

本次验证的细项、范围和限制见 [CLONE_VALIDATION_20261010.json](docs/CLONE_VALIDATION_20261010.json)。只把实际完成的检查作为证据。新 SDK 安装、四场景完整 60 秒新运行和论文指标等价性需要各自独立验证。

人体 UV/纹理直接存储在 Git 中；既有权重、机器人和场景使用 Git LFS。
