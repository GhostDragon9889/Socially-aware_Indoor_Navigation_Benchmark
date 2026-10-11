# 项目文档 / Project documentation

更新 / Updated: **2026-10-11**。中文和英文指南覆盖相同范围，命令示例保持一致。
The Chinese and English guides cover the same scope and use matching command examples.

| 主题 / Topic | 中文 | English |
| --- | --- | --- |
| 项目总览 / Overview | [README](../../README.md) | [README](../../README.en.md) |
| 依赖与兼容 / Dependencies and compatibility | [依赖](dependencies.zh-CN.md) | [Dependencies](dependencies.en.md) |
| 安装与排错 / Installation and troubleshooting | [安装](installation.zh-CN.md) | [Installation](installation.en.md) |
| 资源与配置 / Resources and configuration | [配置](configuration.zh-CN.md) | [Configuration](configuration.en.md) |
| 训练、评估、录像 / Training, evaluation, video | [使用](usage.zh-CN.md) | [Usage](usage.en.md) |

推荐顺序 / Suggested order: **Dependencies → Installation → Configuration → Usage**。

## 配置与证据 / Configuration and evidence

- [实际资源与依赖全面审查 / Verified resource and dependency audit (Chinese)](../RESOURCE_DEPENDENCY_MAP_20261010_ZH.md)
- [机器清单 / Machine-readable actual inventory](../resource_audit_20261010/inventory.json)
- [代码结构与依赖分层 / Codebase map](../codebase/STRUCTURE.md)
- [当前 Lab3 EA / Sim6.1 依赖清单 / Current dependency inventory](dependency-inventory-isaaclab3-ea-sim61.json)
- [默认 SDK manifest](../../env/isaaclab3_ea_sim61/pyproject.toml)、[lock](../../env/isaaclab3_ea_sim61/uv.lock)、[profile](../../env/isaaclab3_ea_sim61/profile.json)、[项目 overlay lock](../../env/isaaclab3_ea_sim61/requirements.overlay.lock)
- [历史 Sim6.0 273 包快照 / Historical 273-package snapshot](dependency-inventory.json) 与 [旧 SDK profile](../../env/isaaclab3_sim6/profile.json)
- [四场景配置 / Four-scene configuration](../../configs/diffusion_amp/four_scene_speed_20261010/)
- [项目结构 / Detailed structure (Chinese)](../PROJECT_STRUCTURE_ZH.md)
- [四场景实测记录 / Recorded local four-scene results (Chinese)](../FOUR_SCENE_SPEED_20261010_ZH.md)
- [本地交付索引 / Local delivery index](../four_scene_speed_final_delivery_20261010.json)
- [上游合并记录 / Upstream integration record (Chinese)](../UPSTREAM_MERGE_20261010_ZH.md)
- [原始 ProtoMotions README / Original ProtoMotions README](../reference/PROTOMOTIONS_README.md)

新指南描述当前安装与部署入口；历史报告保留各自日期与失败结果。克隆源码后按[私有资源发布说明](../../ASSET_RELEASE.md)获取选定权重、训练状态、机器人与四场景；授权 AMASS/SMPL、SDK、视频及搬迁后的地图准入需另行准备。
These guides describe current installation and deployment. Historical reports retain their dates and failures. After cloning, fetch selected models, training state, robot and four scenes using the [private release instructions](../../ASSET_RELEASE.md#english-instructions). Licensed AMASS/SMPL, SDKs, videos and relocated native-map admission remain separate.

本项目仍未验证作者完整评估协议或达到论文全部指标。
The complete author evaluation protocol and all reported paper metrics remain unverified.

当前克隆入口与路径迁移说明：[CLONE_REPRODUCTION.md](../../CLONE_REPRODUCTION.md)。
