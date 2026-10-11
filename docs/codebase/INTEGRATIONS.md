# External Integrations

> **当前默认环境（2026-10-11）**：IsaacLab 3.0 EA（包 17.0.2）/ IsaacSim 6.1.0，使用项目独立 overlay。迁移与实际测试见[迁移审查](../ISAACLAB3_EA_SIM61_MIGRATION_20261011_ZH.md)。下列 2026-10-10 SDK、依赖缺口、远程账户和实验记录保留其当时身份，不能据此推断当前 SDK 或账户状态。

审查：2026-10-10；基于本机文件、安装元数据、AST 和 SHA。此次未启动 Kit、训练或物理重测。完整机器清单见 [inventory.json](../resource_audit_20261010/inventory.json)。

## Core Sections (Required)

### 1) Integration Inventory

| 集成 | 类型 / 身份 | 使用范围 |
| --- | --- | --- |
| IsaacLab-Protomotions + Isaac Sim | 本地editable SDK + NVIDIA wheel/index；单独许可 | 核心原生物理/渲染 |
| GitHub benchmark | GhostDragon9889私有仓库；授权Git账户 | 源码与自己发布的LFS包 |
| NVlabs ProtoMotions LFS | .lfsconfig默认下载端点 | 原始上游模型/示例；不是自有资源包端点 |
| TRACE / PACER | 固定本地Git参考仓库 + src适配 +模型 | 当前行人模型来源 |
| NavIsaaclab2.0 | 独立本地参考仓库；内嵌protomotions | 论文方法/旧接口对照 |
| AMASS / SMPL | 授权本地文件，无自动账户下载 | 专家动作、身体参数和皮肤 |
| Google Drive / Dropbox | 公开PACER/TRACE原始下载链接 | 备用资源获取；runtime不联网拉权重 |
| W&B / TensorBoard | 可选训练日志 | core声明不等于每次推理必须登录 |
| Hugging Face | GhostDGGG预定私有模型仓库 | 本地记录仍待有效CLI身份；本轮插件返回not found，可能不存在或权限不足，未核实上传成功 |
| ROS / FLUX / NavDP | 系统环境/参考项目 | 机器人扩展；非当前四场景录制主链必需 |

TRACE本地revision594b2af、PACER baeb624、FLUX2189820、NavDPbff5eb8、NavIsaaclab2.0 d742f0c；TraceAndPace的两个参考目录与third_party对应commit相同，不是额外算法依赖。精确remotes/40位commit见inventory；原始参考依赖Python3.8/旧Torch/IsaacGym，不能把旧requirements整体安装进Python3.12 SDK。[参考源码记录](../resource_audit_20261010/inventory.json)

### 2) Data Stores

数据以本地JSON/YAML、Torch包、NPZ、USD/材质、视频和Git LFS保存；核心四场景入口没有服务数据库。模型data/与私有release/保留两套路径，install-models仅安装7项模型/配置/manifest，不安装原始AMASS、SMPL或迁移native-map。data/smpl三个pkl为外部arena_assets符号链接；归档时需按授权保留其真实目标。

### 3) Secrets and Credentials Handling

GitHub/HF写入凭据由各自CLI/插件管理；此次不读取令牌、.aws或认证缓存正文。GitHub私有源访问与LFS下载均需仓库授权；HF仓库ID或本地打包就绪不能证明远程上传。训练W&B凭据单独配置。[Git/LFS双端点](../../.lfsconfig)

### 4) Reliability and Failure Behavior

运行前使用本地路径及SHA，不默默下载或换模型；发布verify拒绝路径穿越、缺文件、SHA不符，install拒绝覆盖不同文件。git fetch helper显式指定自有LFS URL和release目录；普通默认LFS拉取不能恢复这套自有对象。下载脚本管理暂存/校验/重试；网络配额失败应保留错误，不把失败文件作为权重。[PACER下载器](../../scripts/fetch_pacer_reference.py)

本机四场景原始map gate通过；已搬迁的releaseUSD引用闭合在发布时核实，但catalog标记native_map_paths_migrated=false、relocated_scene_physics_rerun=false，不能把新位置当作已经物理验收的输入。

### 5) Observability for Integrations

模型profile有path/SHA；motionmanifest有split/ID/source；原生执行有输入/源码快照、日志和实际退出；录像有时长帧数/解码/相机记录。GitHub发布receipt记载独立远程LFS取回核验。HF旧status和本轮读请求另列，不混入GitHub成功状态。

### 6) Evidence

- [实际环境、文件身份、依赖和目录清单](../resource_audit_20261010/inventory.json)
- [本轮四场景预检与发布资源校验](../resource_audit_20261010/resource_checks.json)
- [环境/USD 检查与版本覆盖诊断](../resource_audit_20261010/environment_checks.json)

- [发布/安装实现](../../src/social_navigation/asset_release.py)、[发布说明](../../ASSET_RELEASE.md)、[远程发布核验](../ASSET_PUBLICATION_20261010.json)
- [来源许可](../../legal/THIRD-PARTY%20SOFTWARE%20NOTICES%20AND%20ASSET%20LICENSES%20-%20ProtoMotions.txt)、[TRACE派生NOTICE](../../src/CrowdSim/humanoid/diffusion/NOTICE.md)

## Clone portability update — 2026-10-10

The current four-scene flow uses `configs/reproduction/four_scene`, exact in-memory resource bindings, the committed SDK bootstrap/native shutdown bundle, fixed-bank rebuild templates and `scripts/run_four_scene_benchmark.py`. See [clone reproduction](../../CLONE_REPRODUCTION.en.md) and [validation](../CLONE_VALIDATION_20261010.json). Historical experiment receipts remain provenance, not new-run configuration.
