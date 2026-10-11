> **2026-10-11 迁移入口：** 当前默认为 IsaacLab 3.0 EA（包17.0.2）／IsaacSim 6.1.0.0，项目 `.venv` 指向独立 SDK overlay。请先阅读[迁移报告](ISAACLAB3_EA_SIM61_MIGRATION_20261011_ZH.md)。下列旧 SDK 版本、缺包诊断、训练/评估和视频结果是原日期的历史快照，不能当作新环境验收。新 SDK 依赖缺口已在 overlay 修复；本轮真实运行与剩余边界以迁移报告为准。

# 项目结构与目录迁移（2026-10-10）

项目名称：**Socially-aware_Indoor_Navigation_Benchmark**。
规范目录：`/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark`。
旧目录 `ProtoMotions` 为指向规范目录的兼容链接，已有配置、权重路径和实验记录可继续访问。

最新实际依赖、资源身份、外部路径和恢复缺口见[全面清单](RESOURCE_DEPENDENCY_MAP_20261010_ZH.md)及[代码结构](codebase/STRUCTURE.md)。下方迁移验证保留原实验范围与日期。

## 模块与数据流

```text
AMASS 动作 → protomotions 动作数据/AMP 训练 → 物理行人策略
场景地图/路线 → CrowdSim.humanoid.diffusion (TRACE) → 目标轨迹
目标轨迹 + 策略 → CrowdSim.humanoid.amp/runtime → IsaacLab 物理执行
执行数据 → analysis 与验证工具 → 质量报告 → 校验后的视频录制
social_navigation CLI → scripts 父进程入口 → 实际子进程退出码与源文件快照
```

| 目录 | 职责 |
| --- | --- |
| `src/social_navigation` | 命令分发、资源检查及模拟入口；不在导入时启动 SDK |
| `src/CrowdSim/humanoid/diffusion` | 轨迹扩散与引导算法 |
| `src/CrowdSim/humanoid/amp`、`training` | AMP 控制器适配与训练 |
| `src/CrowdSim/humanoid/runtime`、`application` | 物理步进、运行配置、资源和生命周期 |
| `src/CrowdSim/humanoid/analysis`、`trajectory` | 姿态/轨迹处理与评估 |
| `src/CrowdSim/{world,control,ppo,pref,flow}` | 场景导航、控制、策略及偏好模型 |
| `src/protomotions/{agents,envs,simulator,components,utils}` | 学习算法、环境、模拟器和动作/场景组件 |
| `src/crowdsim_ppo` | 独立 PointGoal PPO 基线 |
| `src/deployment` | 部署与运动工具 |
| `configs/crowdsim` | 原 CrowdSim 配置的统一位置 |
| `configs/diffusion_amp` | 行人模拟、训练、场景、路线与相机配置 |
| `data/navigation_maps/crowdsim` | 原 CrowdSim 导航地图 |
| `data/diffusion_amp/amass_style_2848`、`data/pretrained_models` | 固定动作子集和模型；原始/通用 AMASS 位于外部 `Assets/motion`，当前没有 `data/amass` 目录 |
| `scripts` | 原生进程所有者、训练/评估/录制命令；保留历史入口 |
| `tools/{asset_conversion,retargeting}` | USD 资产转换与动作重定向 |
| `third_party/{trace,pacer,flux,navdp}` | 上游参考实现，保留各自许可与 Git 历史 |
| `packages/crowdsim-ppo` | 最小基线发行包配置；`src` 链接到公共源码 |
| `env`、`tests`、`docs`、`output` | 环境、跨模块测试、复现记录和生成数据 |

## 兼容映射

| 原路径 | 规范路径 |
| --- | --- |
| `ProtoMotions/` | `Socially-aware_Indoor_Navigation_Benchmark/` |
| `protomotions/`、`CrowdSim/`、`deployment/` | `src/` 下同名包 |
| `CrowdSim/config/` | `configs/crowdsim/` |
| `CrowdSim/maps/` | `data/navigation_maps/crowdsim/` |
| `FLUX/`、`NavDP/` | `third_party/flux/`、`third_party/navdp/` |
| `pyroki/`、`usd_convert/` | `tools/retargeting/`、`tools/asset_conversion/` |
| `src/pyproject.toml` | `packages/crowdsim-ppo/pyproject.toml` |

除独立安装配置外，旧路径均通过相对兼容链接保留。旧导入名保持，避免已有权重的 Python 类型路径失效。
只安装基线使用 `pip install -e ./packages/crowdsim-ppo`，主项目使用根目录 `pyproject.toml`。
新命令使用 `python -m social_navigation` 或安装后的 `social-navigation`。

源码仓库定位集中到 `protomotions.repository.repository_root`，通过标记结构向上查找规范根目录，
支持 `SOCIAL_NAVIGATION_ROOT` 覆盖；已安装包的资产仍使用原有 `protomotions.assets` 定位机制。
SDK 环境、权重字节、历史报告与视频不因目录迁移而改写。生成的旧 build/egg-info 副本移入
`output/structure_migration_20261010/generated`。

## VS Code

打开根目录的 `.code-workspace` 文件；编辑器隐藏根目录兼容链接，展示规范源码。
默认解释器复用已验证 `.venv`。CPU 测试关闭外部 pytest 插件自动加载，避免系统 ROS 插件影响本项目。任务使用 `process` 及参数数组，可正确处理带空格的项目路径。
原生模拟通过任务运行；调试配置仅用于状态、资源检查和基线命令。

## 迁移验证

已完成以下实际验证：

- 377 项迁移相关 CPU 测试通过；749 个 Python 源文件语法解析通过。
- 四个场景完整资源、原生地图、支持面记录和路线配置检查通过。
- 23 个关键资源和历史交付文件在迁移前后的 SHA256 一致。
- 主项目与最小 PPO 基线的 wheel/sdist 构建通过，从 wheel 解包后的包导入通过。
- 新发行包及六个兼容/统一控制台入口检查通过；环境锁离线检查通过，273 个锁定包版本不变。
- YOHO 四行人原生模拟实际运行 360 控制步（12 秒）；无摔倒、无重置，SDK 清理成功，子进程退出码为 0。

迁移检查先后发现并修正了旧路径字符串比较、SDK 空格参数拆分及迁入 src 后的模块来源判断。
修正采用规范文件路径比较；保留文件大小、SHA256、类与方法来源和清理所有权检查。
已封存的 SDK 校验源文件、构建清单和 SDK 文件保持原字节；目录适配记录可逆转换及校验值。
早期失败记录均保留，最终成功结果以 `native_migration_smoke_v5` 为准。

验证结果保存在 [verification.json](migration_20261010/verification.json)。
该次原生检查验证迁移后的入口和完整退出；之前已完成的四场景 60 秒视频继续作为上一轮模型结果。
全部历史训练/评估协议未在本次结构迁移中重跑。
旧根目录兼容链接仍用于历史资源记录和 SDK 的无空格启动路径；删除链接前必须迁移这些引用并重新验证。
迁移前后资源校验、构建内容、CPU 测试与原生迁移检查以该记录为准。
源文件修改前备份和逐项迁移清单保存在聊天证据目录 `project_reorganization_20261010`。

## 结构设计依据

- [Python Packaging：src 布局](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [Setuptools：包发现与 package-dir](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)
- [VS Code：Python 调试配置](https://code.visualstudio.com/docs/python/debugging)
- [VS Code：工作区](https://code.visualstudio.com/docs/editing/workspaces/multi-root-workspaces)

当前克隆入口与路径迁移说明：[CLONE_REPRODUCTION.md](../CLONE_REPRODUCTION.md)。
