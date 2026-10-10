# Codebase Structure
审查：2026-10-10；基于本机文件、安装元数据、AST 和 SHA。此次未启动 Kit、训练或物理重测。完整机器清单见 [inventory.json](../resource_audit_20261010/inventory.json)。

## Core Sections (Required)

### 1) Top-Level Map

| 目录 | 实际职责 |
| --- | --- |
| src/social_navigation | 统一 CLI、源码定位、预检、发布资源管理 |
| src/CrowdSim | diffusion/AMP、人群场景、训练、运行、分析与机器人导航分支 |
| src/protomotions | agents、envs、simulator、robot_configs、components、utils、人体/机器人资产 |
| src/crowdsim_ppo | 无 SDK 的 PointGoal PPO |
| src/deployment | tracker 导出、MuJoCo 部署及 motion_utils |
| configs | 实验配方、场景/路线/相机、模型身份与准入 |
| data | 模型、固定 AMASS 子集、生成语料、参考地图；原始 AMASS 在外部 Assets/motion |
| resources/releases/pedestrians_20261010 | 私有 LFS 发布资源及清单 |
| scripts / tools | 进程入口、训练/评估/录制、转换与重定向 |
| third_party | TRACE/PACER/FLUX/NavDP 参考 checkout |
| packages/crowdsim-ppo | 独立基线 manifest；源码复用 src/crowdsim_ppo |
| env/isaaclab3_sim6 | SDK manifest/lock/profile 快照 |
| tests | 跨模块/原生接口测试 |
| examples / docs / legal | 上游示例、方法与证据档案、许可 |
| output | 训练、物理状态、地图及历史运行产物 |
| .vscode | 解释器、调试与任务配置 |

此次 AST 范围：src609、scripts139、tools7、tests154、examples47、data/scripts27 个 Python 文件，共983，无语法错误。仅 src/scripts/tools/tests 合计909；这是语法覆盖，不是逐行功能认证。扫描器48,569文件/约1,176万行还含历史源码快照和巨型JSON，不作为生产代码规模。

### 2) Entry Points

六个 console scripts 由根 manifest 注册：social-navigation、crowdsim-ppo-train/eval、protomotions、protomotions-train-agent、protomotions-inference-agent。`social_navigation.cli` 分发训练、预检、原生模拟、独立基线及录制/审计；多数子任务依赖源码中的 scripts/configs。因此只安装 wheel 不等价于拥有完整实验 checkout。[CLI](../../src/social_navigation/cli.py)

### 3) Module Boundaries

CrowdSim/humanoid 下 diffusion 为轨迹；amp 为公开 PACER 合同/网络/PD动作；training 为本地训练；application 为参数和资源 gate；runtime 为场景/地图/接触/生命周期；analysis 为离线评价。AMPWorld 接 ProtoMotions IsaacLab simulator。CrowdSim/world/control/ppo/pref/flow 包含导航、机器人控制和其他数据收集分支，不能把其 PPO 输出当作当前行人 AMP 权重。

**docs 并非全是可删除历史资料**：social_evaluation_lifecycle 动态载入 docs 中封存 terminal helper；它及其 overlay/SDK绑定是运行依赖。output/scene_maps 中选定 native-map 也是当前必需资源。删除历史目录前要按实际引用筛选。

### 4) Naming and Organization Rules

根兼容链接 CrowdSim/protomotions/deployment→src，FLUX/NavDP→third_party，pyroki/usd_convert→tools，.venv→外部SDK；旧 ProtoMotions 目录是当前仓库别名。`third_party` 保留各自 .git，根无 .gitmodules，且第三方目录被忽略/从 sdist 排除；普通 clone 不恢复其 checkout。NavIsaaclab2.0/protomotions 则是 Nav 仓库内目录，不是独立 Git 项目。

CLI场景名 YOHO/NewMetro/SecondFloor/ThirdFloor 与源场景长目录名分开。data/amass 当前不存在；应使用 data/diffusion_amp/amass_style_2848 与外部 Assets/motion。[结构说明](../PROJECT_STRUCTURE_ZH.md)

### 5) Evidence

- [实际环境、文件身份、依赖和目录清单](../resource_audit_20261010/inventory.json)
- [本轮四场景预检与发布资源校验](../resource_audit_20261010/resource_checks.json)
- [环境/USD 检查与版本覆盖诊断](../resource_audit_20261010/environment_checks.json)

- [打包边界](../../MANIFEST.in)、[忽略规则](../../.gitignore)、[包声明](../../pyproject.toml)
- [动态 terminal 依赖](../../src/CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py)、[源码定位](../../src/protomotions/repository.py)

## Clone portability update — 2026-10-10

The current four-scene flow uses `configs/reproduction/four_scene`, exact in-memory resource bindings, the committed SDK bootstrap/native shutdown bundle, fixed-bank rebuild templates and `scripts/run_four_scene_benchmark.py`. See [clone reproduction](../../CLONE_REPRODUCTION.en.md) and [validation](../CLONE_VALIDATION_20261010.json). Historical experiment receipts remain provenance, not new-run configuration.
