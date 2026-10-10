# 核心源码

| 包 | 职责 |
| --- | --- |
| `social_navigation` | 统一 CLI、规范化仓库定位 |
| `CrowdSim` | 人群导航、Diffusion 轨迹、AMP 桥接、评估与运行 |
| `protomotions` | 物理模拟、动作数据、策略训练与推理 |
| `crowdsim_ppo` | 不依赖模拟器的 PointGoal PPO |
| `deployment` | 运动处理与部署 |

安装主项目使用根目录 `pyproject.toml`；仅安装最小基线使用 `packages/crowdsim-ppo`。
目录布局见根目录 README 和 `docs/PROJECT_STRUCTURE_ZH.md`。
