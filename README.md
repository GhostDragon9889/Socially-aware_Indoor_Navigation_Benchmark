# Socially-aware_Indoor_Navigation_Benchmark

**从 Git 克隆复现：** [完整步骤](CLONE_REPRODUCTION.md)（SDK 引导、资源恢复、固定 AMASS 重建、四场景评估和录制）。

[简体中文](README.md) · [English](README.en.md) · [完整双语文档](docs/guides/README.md)

## 安装与使用文档

| 主题 | 中文 | English |
| --- | --- | --- |
| 依赖和兼容性 | [依赖](docs/guides/dependencies.zh-CN.md) | [Dependencies](docs/guides/dependencies.en.md) |
| 安装与常见问题 | [安装](docs/guides/installation.zh-CN.md) | [Installation](docs/guides/installation.en.md) |
| 权重、数据与场景配置 | [配置](docs/guides/configuration.zh-CN.md) | [Configuration](docs/guides/configuration.en.md) |
| 微调、评估与录像 | [使用](docs/guides/usage.zh-CN.md) | [Usage](docs/guides/usage.en.md) |
| 发布的权重、机器人与四场景 | [资源发布](ASSET_RELEASE.md) | [Asset release](ASSET_RELEASE.md#english-instructions) |
| 实际资源和依赖审查 | [2026-10-10 全面清单](docs/RESOURCE_DEPENDENCY_MAP_20261010_ZH.md) | [Codebase map](docs/codebase/STACK.md) |

基于 Isaac Sim 6 / IsaacLab 3、TRACE 轨迹扩散与 PACER/AMP 物理行人的室内社交导航研究项目。
核心实现使用 `src/` 布局，保留 ProtoMotions、CrowdSim 的导入名及原始项目署名。

## 当前可用成果

- 在本地真实 AMASS 动作上微调的 AMP 行人控制器，以及四场景的配置、路线与相机方案。
- YOHO、NewMetro、SecondFloor、ThirdFloor 各一段 60 秒视频与对应的物理、姿态、速度、停滞和录制校验。
- 物理过程的实际退出码、源文件快照、权重与资源校验值、完整视频解码校验。
- 独立的 PointGoal PPO 基线，用于无需模拟器的导航实验。

已完成结果、速度实测及覆盖范围见 [四场景说明](docs/FOUR_SCENE_SPEED_20261010_ZH.md) 和
[最终交付索引](docs/four_scene_speed_final_delivery_20261010.json)。当前结果不代表论文完整协议的复现，
也未验证达到原论文指标；机器人社交导航和全场景路线覆盖仍需独立实验。

## 项目结构

```text
src/
  social_navigation/  统一命令入口、项目定位
  CrowdSim/           人群、轨迹扩散、AMP 接口、行人运行与评估
  protomotions/       物理模拟后端、动作库、策略训练与推理
  crowdsim_ppo/       独立的 PointGoal PPO 基线
  deployment/         部署与动作处理
configs/              实验、场景、相机与导航配置
data/                 AMASS、模型权重、场景资源、导航地图
resources/releases/   私有 LFS 权重、机器人、四场景及发布清单
scripts/              训练、模拟、验证与录制的进程入口
tools/                资产转换与动作重定向工具
third_party/          上游参考仓库
packages/             独立基线的安装配置
env/                  IsaacLab 3 / Isaac Sim 6 环境配置
tests/                跨模块与原生执行合同测试
docs/                 复现说明、来源与实验记录
output/               生成的训练、模拟与录制结果
.vscode/              解释器、调试、任务与扩展配置
```

详细模块职责、目录映射及验证记录见 [项目结构](docs/PROJECT_STRUCTURE_ZH.md)。

## GitHub 源码与资源

上游完整文件树和原始 Git 历史已合并；文件映射、LFS 下载和本地历史说明见
[上游合并记录](docs/UPSTREAM_MERGE_20261010_ZH.md)。

私有仓库：[GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark](https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark)。访问和克隆需要该仓库的授权账号。

源码仓库上传范围、固定上游来源、资源准备和旧 Git 历史保留方式见
[GitHub 源码说明](docs/GITHUB_SOURCE_ZH.md)。最新选用的行人权重、训练参数、机器人和四场景资源已整理为
[私有 LFS 发布包](ASSET_RELEASE.md)；AMASS 原始动作、SMPL 参数、SDK 和视频独立准备。

## 使用现有环境

从项目根目录运行。
`.venv` 指向已经验证的 IsaacLab/Isaac Sim 环境。

```bash
cd '/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark'
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m social_navigation status
.venv/bin/python -m social_navigation preflight --scene YOHO
.venv/bin/python -m social_navigation simulate --scene YOHO --steps 1800
```

`preflight` 检查本地资源与原生场景配置；`simulate` 使用既有父进程启动器执行原生物理模拟，
输出到自动生成的新目录。1800 个 30 Hz 控制步对应 60 秒模拟时间。
原生任务须逐个运行，避免多个 Isaac 实例争用 GPU。

现有环境安装本项目时，可在项目根目录执行以下命令，沿用现有依赖：

```bash
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
.venv/bin/social-navigation --help
```

创建新环境请先遵循 [中文安装指南](docs/guides/installation.zh-CN.md)；
环境版本与测试资源不能用未经验证的依赖升级替代。

## 训练、评估与录制

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

这些命令转交原有已验证工具的参数，保留质量门槛、实际子进程退出码和资源记录。
训练配方及录制来源见四场景说明；每次训练或录制应指定新的输出位置。

## VS Code

打开根目录的 `Socially-aware_Indoor_Navigation_Benchmark.code-workspace`。
推荐扩展为 Python、Pylance、Python Debugger 与 Ruff；默认解释器为 `.venv/bin/python`。
“任务”提供资源检查、原生模拟、CPU 迁移测试和工具帮助；“运行和调试”提供状态、资源检查与基线调试。

## 上游与许可

项目包含 [ProtoMotions](https://github.com/NVlabs/ProtoMotions)、
[TRACE / Trace and Pace](https://github.com/nv-tlabs/trace)、PACER 及 NavIsaacLab 相关实现与本地适配。
原始 ProtoMotions README 保存在 [参考文档](docs/reference/PROTOMOTIONS_README.md)。
保留 [LICENSE.md](LICENSE.md) 与 `legal/` 及各第三方的许可。
SMPL/SMPL-H、AMASS、权重和场景资产遵循各自许可，模型人体资产不随构建的 wheel/sdist 分发。
