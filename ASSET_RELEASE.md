# 行人权重、机器人与四场景资源 / Pedestrian, robot and four-scene assets

发布版本 / Release: **`pedestrians_20261010`**

目标仓库 / Private repository: [GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark](https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark)

## 中文说明

### 1. 本次包含什么

本次发布采用四场景最终交付配置实际使用的权重。`最新` 指已通过本地四场景验证的选用版本；选择依据是 [最终交付索引](docs/four_scene_speed_final_delivery_20261010.json) 和 [四场景报告](docs/FOUR_SCENE_SPEED_20261010_ZH.md)。

资源位于 [`resources/releases/pedestrians_20261010/`](resources/releases/pedestrians_20261010/)。

| 资源 | 发布路径（相对资源目录） | 用途 |
| --- | --- | --- |
| AMP 保守速度控制器 | `models/amp/speed_conservative_20261010/Humanoid_AMASS.pth` | YOHO、NewMetro |
| AMP 行走/转弯控制器 | `models/amp/walking_turn_20261009/Humanoid_AMASS.pth` | SecondFloor、ThirdFloor |
| AMP 训练父权重 | `models/amp/walking_turn_20261009/initialization_compact_v8.pth` | 保存训练来源 |
| TRACE Diffusion | `models/trace/iter40000.ckpt`、`config.json` | 轨迹扩散与引导 |
| PACER 教师 | `models/pacer_teacher/Humanoid.pth` | 微调冻结教师 |
| 两次微调的训练状态与参数 | `training/{speed_conservative_20261010,walking_turn_20261009}/` | `last.pt`、参数、配方、指标、退出记录 |
| 机器人 | `robot/EVT-EXP.usd` | 用户指定的 EVT-EXP 原始机器人 |
| 四个场景 | `scenes/{YOHO,NewMetro,SecondFloor,ThirdFloor}/` | 当前收集场景、测试静态版本、材质和纹理 |
| 场景/相机/路线配置 | `configs/original/` | 原始配置逐字节保留 |
| 模型路径可迁移配置 | `configs/portable_models/` | 模型和 AMASS 清单路径相对项目根目录 |
| AMASS 来源清单 | `provenance/amass_manifest_labeled_v7.json` | 固定 2,848 条训练动作的来源和划分元数据 |
| 评估与原生地图记录 | `evidence/` | 原始证据、查询数组、地图、探测脚本 |
| 原始 USD 快照 | `source_snapshots/` | 被转换或迁移图层的原字节 gzip 副本 |

[`manifest.json`](resources/releases/pedestrians_20261010/manifest.json) 记录每个文件的大小、SHA-256、用途与来源；[`catalog.json`](resources/releases/pedestrians_20261010/catalog.json) 给出机器人和各场景的具体入口路径。
资源文件共 468 个（含清单），文件内容约 **2.69 GB / 2.50 GiB**；Git LFS 按内容去重，实际传输量以上传记录为准。

### 2. 下载与验证

需要有该私有仓库的访问权限，并安装 Git LFS。原上游 LFS 引用仍从 NVlabs 下载；本发布包使用本私有仓库的 LFS 端点。克隆时跳过自动下载，然后调用资源工具：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark.git
cd Socially-aware_Indoor_Navigation_Benchmark
git remote add benchmark \
  https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark.git
git lfs install --local
python3 scripts/manage_release_assets.py fetch
python3 scripts/manage_release_assets.py verify
python3 scripts/manage_release_assets.py catalog
```

已有项目远程 `benchmark` 的工作区省略 `git remote add`。工具只对本发布目录下载私有 LFS 文件，保留原有上游下载设置。

在已安装 OpenUSD 的模拟器 Python 环境中，额外验证九个 USD 入口（四个当前场景、四个测试静态场景和一个机器人）的本地依赖闭合：

```bash
.venv/bin/python scripts/manage_release_assets.py verify --check-usd
```

验证检查全部文件 SHA、原始 gzip 解压后的 SHA，以及 USD 本地依赖均位于发布目录内。`OmniSurface.mdl`、`OmniPBR.mdl`、`OmniGlass.mdl` 是 Isaac Sim 提供的材质模块，由 SDK 材质搜索路径加载。

GitHub 文件页显示大型文件的 LFS 指针；必须执行下载工具后才能获得真实权重、USD 和纹理。普通源码 ZIP 是否含 LFS 实体取决于仓库设置；以清单校验为准。详见 [GitHub LFS 文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

### 3. 权重与配置使用

模型可以直接通过发布目录下的路径加载；也可以恢复到现有代码要求的 `data/` 路径：

```bash
python3 scripts/manage_release_assets.py install-models
```

安装前检查所有目标：相同内容复用，内容不同则停止并保留现有文件。该操作包括选用 AMP、训练父权重、TRACE、教师和 AMASS 清单，不包含 AMASS 动作文件。

新机器查看 `configs/portable_models/{场景}_profile.json`；其中资源路径相对项目根目录。路线/相机数值与原选用配置一致。训练超参数见 `training/*/parameters.json` 与 `checkpoint_parameters.json`，实际配方见 `recipe/`，历史原始命令和来源仍保留绝对路径。

`last.pt` 保存优化器、归一化器、随机数及训练器状态；本发布验证文件完整性，不声称跨 SDK/跨机器精确恢复训练已经通过物理测试。

### 4. 场景使用与复现范围

通过 `catalog` 查看并在 Isaac Sim 中打开 `current_stage` 或 `tested_static_stage_relocated`。本发布仅迁移 USD 资产引用，并将大型文本几何图层转成二进制 USD；没有修改关节、碰撞、物理参数或场景坐标。被改写图层的原始字节保存在 `source_snapshots/`，其解压校验值见清单。

当前场景来自 `/project/EXP/Sence/Sences/` 的四个场景目录。`tested_static` 来源是原 60 秒 Diffusion+AMP 实验使用的静态冻结版本；当前外部场景后续修改不被视作已经重做同一实验。

原始 `preflight`、地图结果和实验记录包含本机绝对路径及 SDK 来源校验，按原字节保存。新机器应按 [安装](docs/guides/installation.zh-CN.md)、[配置](docs/guides/configuration.zh-CN.md) 准备固定环境与本地授权数据，并针对新资源位置重新执行地图探测、资源预检和物理验收。资源副本的 USD 依赖闭合验证不能替代这些运行证明。`configs/portable_models` 仅迁移模型路径，未将历史地图记录重新认证成新机器的有效证据。

已验证环境为本项目固定的 **Isaac Sim 6.0.0.1 / IsaacLab 工作区**；与 `/project/EXP/IsaacLab` 的 Sim 6.1 / Lab 3.0 EA 迁移仍有独立兼容性工作。四场景现有视频和物理实验没有加入机器人；本次提供机器人模型，不据此声明机器人社交导航通过验收，也不声明达到/超越论文效果。

### 5. 许可与发布记录

保留 `LICENSES/` 中 TRACE、PACER、ProtoMotions 原始许可证和署名；AMASS、SMPL、场景及 SDK 按各自授权使用。原始 AMASS 动作、SMPL 参数、SDK 二进制及视频仍按项目既有流程独立准备。

本发布的大型文件由 Git LFS 上传到 `benchmark` 私有仓库；上游约 5.3 GB 的资源继续保留原 LFS 引用和下载来源。源码提交成功与 LFS 实体上传成功分别核验。远程核验结果见 [资源发布记录](docs/ASSET_PUBLICATION_20261010.json)。

## English instructions

### Contents

This release publishes the controllers selected by the completed four-scene delivery: the conservative AMP model for YOHO/NewMetro and the walking/turning model for SecondFloor/ThirdFloor, TRACE diffusion/config, the frozen PACER teacher, a training parent model, both training states and recipes, the original EVT-EXP robot, four collected scenes and their tested static variants, textures, cameras/routes, the fixed 2,848-motion AMASS provenance manifest and original evaluation/native map evidence.

The root is `resources/releases/pedestrians_20261010/`. Its `manifest.json` contains full-byte SHA-256 checksums and source provenance; `catalog.json` lists all scene and robot entry points. There are 468 files including the manifest, approximately 2.69 GB (2.50 GiB) before LFS content deduplication.

### Download and check

Use the clone/download commands above with an authorized GitHub account and Git LFS. `scripts/manage_release_assets.py fetch` overrides the LFS endpoint only for this private release. Original ProtoMotions references continue to download from NVlabs.

`verify` checks every file and decompresses source snapshots to verify the exact original USD hashes. `verify --check-usd`, using a Python environment with `pxr`, additionally checks dependency closure for nine entry stages. All local dependencies must remain inside the release. The three named Omni MDL modules are provided by Isaac Sim.

### Models, training and scenes

`install-models` copies weights/config/provenance into the existing `data/` layout and refuses to overwrite a different local resource. Model profiles under `configs/portable_models/` refer to paths relative to the checkout root. Original deployment and camera/route settings remain under `configs/original/`.

Training directories include actual recipes, hyperparameters, metrics, process receipts and `last.pt` with optimizer/normalizer/RNG/runner state. Exact physical resume across machines or SDK versions has not been validated by this publication.

USD references were relocated and large text mesh layers converted to binary USD. Geometry, joint/collision parameters and coordinates were not edited. Changed source layers retain exact gzip snapshots. The tested static scenes come from the original 60-second experiments; collected scene entry layers represent the current external scene files.

Historical native query/evaluation records retain their source-bound absolute paths. On a fresh machine, prepare the pinned SDK and authorized motion/body data, then regenerate native maps, preflight and physical acceptance evidence. Model-path portability and USD dependency closure do not establish a successful relocated physics run. See [installation](docs/guides/installation.en.md) and [configuration](docs/guides/configuration.en.md).

The validated stack used Isaac Sim 6.0.0.1 and the project's pinned IsaacLab workspace. Migration to the separate local Sim 6.1/Lab 3.0 EA workspace remains separate work. The robot was not present in the selected four-scene physics/video experiments. This release makes no robot-navigation acceptance or paper-level performance claim.

Keep the included upstream licenses and attribution. Raw AMASS motions, SMPL parameters, SDK binaries and videos are prepared separately. The private LFS upload and Git commit are verified separately; see `docs/ASSET_PUBLICATION_20261010.json` for the publication receipt.
