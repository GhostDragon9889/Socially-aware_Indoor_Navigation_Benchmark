# 项目文档 / Project documentation

更新 / Updated: **2026-10-10**。中文和英文指南覆盖相同范围，命令示例保持一致。
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

- [273 包依赖快照 / 273-package dependency inventory](dependency-inventory.json)
- [SDK manifest](../../env/isaaclab3_sim6/pyproject.toml)、[lock](../../env/isaaclab3_sim6/uv.lock)、[profile](../../env/isaaclab3_sim6/profile.json)
- [四场景配置 / Four-scene configuration](../../configs/diffusion_amp/four_scene_speed_20261010/)
- [项目结构 / Detailed structure (Chinese)](../PROJECT_STRUCTURE_ZH.md)
- [四场景实测记录 / Recorded local four-scene results (Chinese)](../FOUR_SCENE_SPEED_20261010_ZH.md)
- [本地交付索引 / Local delivery index](../four_scene_speed_final_delivery_20261010.json)
- [上游合并记录 / Upstream integration record (Chinese)](../UPSTREAM_MERGE_20261010_ZH.md)
- [原始 ProtoMotions README / Original ProtoMotions README](../reference/PROTOMOTIONS_README.md)

新指南描述当前安装与部署入口；历史报告保留各自日期与失败结果。源码克隆不包含本地授权资源、全部自训模型、场景地图和视频。
These guides describe current installation and deployment. Historical reports retain their dates and failures. Cloning source does not provide licensed local data, all trained models, native scene maps or videos.

本项目仍未验证作者完整评估协议或达到论文全部指标。
The complete author evaluation protocol and all reported paper metrics remain unverified.
