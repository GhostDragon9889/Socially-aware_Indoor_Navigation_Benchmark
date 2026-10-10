# 项目资源与依赖总览

审查日期：**2026-10-10**。仓库 `/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark`，起点commit `dd59d1aaeeab85408be0932d913efcf4516a42f1`。旧路径ProtoMotions为兼容别名。

## 1. 结论与本次范围

**现有四场景行人所选权重、动作库、地图证据、人体参数、场景/机器人资产均在本机存在；发布包467文件SHA核验通过，四场景原生配置预检4/4通过。** 资源存在和预检通过只覆盖身份/配置。此次未重训、未启动Kit、未重跑物理；历史视频/指标仍按其原实验范围解释。

已核实源码层、根18项core与全部extras、SDK273项锁依赖/实际版本、可选转换链、外部参考仓库、固定AMASS与通用动作库、当前模型/训练状态、USD/地图/SMPL链接、远程发布记录及四段视频。扫描/AST不构成“每一行都人工认证”或全项目兼容性保证。[机器清单](resource_audit_20261010/inventory.json) / [本轮资源检查](resource_audit_20261010/resource_checks.json)

## 2. 功能与依赖关系

| 功能 | 代码入口 | 软件/资源必需项 |
| --- | --- | --- |
| 轨迹规划 | src/CrowdSim/humanoid/diffusion | Torch/Torchvision/einops、TRACE config+weight、邻居历史、地图和路线 |
| 物理动作 | humanoid/amp + amp_world + ProtoMotions simulator | 固定IsaacLab/Sim、选定AMP、69DOF/24body资产、native场景证据、AMASS/初始化引用 |
| AMASS微调 | scripts/finetune_pacer_amass_verified.py | v7训练bank、父AMP、PACER教师、训练配方/先验/窗口、真实PhysX |
| 独立风格评价 | analysis + evaluate/audit脚本 | 22验证/34测试选择清单、对应动作包、真实物理轨迹；不能只看训练reward |
| 场景物理评价 | run_trace_pace_verified + contact/physics audit | USD层/材质、支持面、源地图/查询NPZ/源码、30Hz与120Hz记录 |
| 视频 | record_pedestrian_coverage | 合格物理源、相机、SMPL neutral/UV/texture、smplx/chumpy/Pillow、Kit、ffmpeg/ffprobe |
| PointGoal基线 | src/crowdsim_ppo | NumPy/Torch；无需Kit，无人体AMP生成能力 |
| EVT机器人社交导航 | CrowdSim/world/control/ppo等扩展 | 机器人USD + 驱动/传感器/导航策略与耦合测试；资产已在，闭环尚未验收 |

流程图及模型合同：[ARCHITECTURE](codebase/ARCHITECTURE.md)。完整目录/入口：[STRUCTURE](codebase/STRUCTURE.md)。

## 3. 三层依赖事实

1. 根 [pyproject](../pyproject.toml) 声明版本范围和后端extra；本身没有uv.lock。
2. [SDK manifest](../env/isaaclab3_sim6/pyproject.toml) + [lock](../env/isaaclab3_sim6/uv.lock) + [profile](../env/isaaclab3_sim6/profile.json) 是当前物理行人环境快照，供IsaacLab checkout根使用。
3. 本机清洁PYTHONPATH下的metadata与uv实际核查：18core均满足、272非virtual锁定版本均匹配；**仍有3项Sim元数据覆盖冲突**。本机ROS PYTHONPATH能额外暴露大量ROS包，此次没有把它们算成项目必需包。

| 层 / 环境 | 核实结果 |
| --- | --- |
| Python / 硬件 | 3.12.13 / RTX4080SUPER 16GB / 驱动595.99.02 |
| 既有 SDK | 产品3.0.0、package isaaclab12.0.0、Sim6.0.0.1、Torch2.11+cu128 |
| 主训练/数据 | NumPy2.3.1、SciPy1.17、Lightning2.5.6、TensorDict0.9、W&B0.23 |
| 视觉 | smplx0.1.28、chumpy0.70、Pillow12.1.1、headless OpenCV4.13、MoviePy1.0.3、FFmpeg6.1.1 |
| USD | usd-exchange3.0.0；排除usd-core；standalone USD26.8，234 RECORD文件/round-trip通过；Kit历史25.11未在本轮启动验证 |
| uv lock | 273条：257registry / 14editable / 1git / 1virtual |
| 根可选backend | IsaacGym/IsaacLab/MuJoCo/Newton/Genesis互斥；Newton/Genesis约束不能整体混入SDK |
| 元数据冲突 | Newton1.4dev vs Sim要求1.2、schemas0.5 vs0.2、coverage7.6.1 vs7.4.4 |

**源码revision漂移：** 当前SDK HEAD `5951d48dde083986489957a7b108ad48a46822a7`（配置依赖提交），profile仍记录`4ecd0b...`。manifest/lock SHA一致，`source/isaaclab`两提交间diff为空，但 `require_checkout` 实际失败，故再次执行apply会被拒绝。此次未改SDK/profile，也没有将门禁删掉。[记录](resource_audit_20261010/environment_checks.json)

**另一个SDK并非同环境：** `/project/EXP/IsaacLab` 为Sim6.1.0.0、package17.0.2、NumPy2.5.1、TensorDict0.13。缺dm-control/easydict/lightning/moviepy/scikit-image/termcolor/wandb；OmegaConf2.4dev和TensorDict越界。现有extra固定12.0.0/6.0.0.1，无共同安装解；scene复制/关闭流程也有已复现差异。[同日兼容性审查](</home/lfm/Documents/ChatGPT/Social Navigation/isaaclab_compatibility_audit_20261010/ISAACLAB_COMPATIBILITY_REVIEW_ZH.md>)是先前证据，其中HEAD旧值应结合本轮记录阅读。

完整18项core、extras、开发工具与命令：[STACK](codebase/STACK.md)。全部包及源/Wheel hashes保留在SDK lock，机器清单另保存每项安装版本/requirement检查。

### 功能依赖缺口

- root未明确声明TRACE的einops及皮肤的smplx/chumpy；SDK完整manifest已补充，所以现有主链不因它们缺包，但仅pip安装根包不能视为功能完整。
- crowd extra声明GUI OpenCV，而SDK锁使用headless；两种发行包共享cv2，避免同时安装。目标SDK是否提供cv2不能只按发行包名判断。
- rvo2/shapely用于ORCA生成，JAX/pyroki/viser/yourdfpy用于重定向，bvh/soma用于可选数据转换；本机主SDK中缺这些可选工具。gdown仅公开下载器需要，当前主SDK未装。pandas/rl-games不是适配后的四场景推理必需项。
- pytest已装；Ruff/pre-commit/Sphinx未装，属于开发/文档功能。docs extra与docs/requirements插件集合不同。根isaacgym extra为空，不代表IsaacGym已安装。

## 4. 模型与训练状态

| 资源 | 当前运行位置（相对仓库） | 核实状态 / 用途 |
| --- | --- | --- |
| TRACE ORCA | data/pretrained_models/paper_references/trace/ckpt/trace/orca_mixed/iter40000.ckpt + config.json | 376.74MB，SHA `fd6ba631…` / config `35a5fb32…`；当前四场景规划 |
| AMP较宽场景 | data/pretrained_models/local_amass_finetune/speed_conservative_20261010/Humanoid_AMASS.pth | 39.98MB，`cb278369…`；YOHO/NewMetro |
| AMP窄场景 | data/pretrained_models/local_amass_finetune/walking_turn_20261009/Humanoid_AMASS.pth | 39.98MB，`0cd96134…`；SecondFloor/ThirdFloor及后续微调父模型 |
| PACER no_shape教师 | data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth | 119.76MB；冻结锚定教师，不能与当前微调模型混写 |
| compact v8初始化 | local_amass_finetune/walking_turn_20261009/initialization_compact_v8.pth | 发布包保留历史初始化来源 |
| 完整恢复状态 | resources/releases/pedestrians_20261010/training/{speed_conservative_20261010,walking_turn_20261009}/last.pt | 训练状态、参数、配方、metrics/result/execution/CPU audit成套 |
| 其他模型 | masked_mimic、motion_tracker、gpc_prior、TRACE nusc两变体、PACER shape、旧AMP候选 | 本机存在，属于替代实验/上游示例/历史；不是当前四场景部署的缺失项 |

当前profile四个资源的完整SHA都已重新核对。发布包含上述选定模型、教师/config及manifest，install-models可恢复7项对应运行路径；不只靠上游NVlabs LFS。[发布manifest](../resources/releases/pedestrians_20261010/manifest.json)保存全SHA和每个role。当前TRACE权重未重训，AMP有本地微调；两者不能称为NavIsaacLab作者未公开的定制权重。[实测方法](FOUR_SCENE_SPEED_20261010_ZH.md)

## 5. AMASS、标注与身体资源

| 资源层 | 当前内容 / 状态 |
| --- | --- |
| 原始/转换根 | /project/EXP/SangforAndSaut/Assets/motion；raw、processed/smpl、preparation |
| generic MotionLib | train11,303 / validation345 / test138；来源preparation历史报告，当前3bank SHA重新核对一致 |
| 早期curation manifest | data/diffusion_amp/amass_manifest.json实际选择319/22/34；不是generic bank全部数量 |
| 当前v7 | data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json；2848/22/34；trainbank1,155,817,682B，SHA`46e02411…` |
| 固定train来源 | 2848片段来自1853个父.motion文件，8个AMASS子集；不是2848个不同原始文件 |
| split隔离 | parent-source路径交集0；selected IDs无重复，来源文件无缺失；不等于跨数据集内容去重或教师预训练无重叠 |
| SMPL人体参数 | data/smpl/{SMPL_NEUTRAL,SMPL_MALE,SMPL_FEMALE}.pkl实际存在但为外部arena_assets链接；neutral皮肤是录像必需项 |
| SMPL视觉 | smpl_uv.npz、texture等本地文件；物理MJCF/USD/mesh另在src/protomotions/data/assets |
| 其他人体数据 | SOMA/G1/Kimodo/GRAB等是其他上游转换/机器人实验，不是当前SMPL行人必需项 |

固定训练子集：KIT1233、CMU699、Eyes_Japan_Dataset422、BioMotionLab_NTroje304、ACCAD88、EKUT55、TotalCapture44、TCD_handMocap3。验证HumanEva10/SFU12，测试Transitions_mocap31/SSM_synced3。当前训练manifest的diffusion_eligible片段为1131，并非2848条全用于轨迹扩散。验证/测试path仍指完整通用动作库，选择由clips定义。

AMASS 22份归档/19份已解压是preparation历史记录，本轮不重新解压或反序列化银行。manifest明确author_clip_list=false。v7记录2,149条文件名标签与699条官方描述标签，类别为walking_like720、turning_like1568、sidestepping_like116、standing_looking_like427、running_like17；这些是筛选类别，未逐条由人工认证。BABEL归档在外部Assets，是其他语义标注实验资源，不由当前四场景profile直接引用。历史label provenance和v7/先验来源需成套保存。原始AMASS与SMPL参数没有随私有GitHub资源包分发。原始数据/身体参数按授权单独准备。[配置准备](guides/configuration.zh-CN.md)

## 6. 四场景、机器人与地图

| CLI名 | 原始场景入口 | 当前人数 |
| --- | --- | ---: |
| YOHO | /project/EXP/Sence/Sences/Rest_Room_HongKong_YOHO/Collected_scene/scene.usda | 4 |
| NewMetro | /project/EXP/Sence/Sences/Rest_Room_HongKong_NewMetro/Collected_scene/scene.usda | 4 |
| SecondFloor | /project/EXP/Sence/Sences/Rest_Room_Second_Floor/restroom_v19/restroom_v19.usda | 1 |
| ThirdFloor | /project/EXP/Sence/Sences/Rest_Room_Third_Floor/Collected_third_floor_optimized/third_floor_optimized.usda | 1 |

原始机器人 `/project/EXP/Sence/Robot/EVT-EXP.usd`。私有包同时保存四场景collected及tested_static、机器人、依赖材质/贴图和原USD gzip；[catalog](../resources/releases/pedestrians_20261010/catalog.json)给出9个USD入口及portable model profiles。发布时已核验全部本地USD依赖闭合，SDK提供OmniPBR/OmniGlass/OmniSurface MDL，不需误补为本地geometry。

当前生效配置为 `configs/diffusion_amp/four_scene_speed_20261010/`。地图三件套preflight/native_map/probe_script_snapshot及后续NPZ/occupancy/支持面/场景资源记录是当前运行依赖。它们位于output/scene_maps/gait_<场景>_cell_v1及本聊天任务目录；此次库级准入4/4通过。地图CPU投影目录data/diffusion_amp/maps*和data/navigation_maps不是可任意替换的native PhysX地图。

**发布包可恢复文件，但尚非跨机一键运行：** native_map_paths_migrated=false，relocated_scene_physics_rerun=false。需要迁移SMPL链接、SDK/terminal绑定，并为新场景位置重新生成/核验源绑定地图、路由和物理证据。保留原静态有限支持面假设；不声称完整复现所有非平坦地面。

## 7. 数据语料与参考仓库

data/diffusion_amp含ORCA-Maps/Interact raw、usable、all_temporal、pilot多个版本；用于扩散训练/验证或历史实验，当前部署使用已训练TRACE模型。6份manifest及10,072条map/episode/window文件引用的尺寸和完整SHA本轮核对均通过；视图间共享文件，10,072不是唯一文件数，窗口计数来自manifest，未重新逐个加载窗口张量。

| 本地语料 / 视图 | train / val / test场景 | manifest声明窗口数 |
| --- | --- | --- |
| orca_interact_all_temporal_1000_v2 | 800 / 100 / 100 | 151650 / 19386 / 19908 |
| orca_interact_usable_1000_v1 | 800 / 100 / 100 | 150439 / 19237 / 19692 |
| orca_maps_all_temporal_1000_v2 | 800 / 100 / 100 | 81396 / 10566 / 9558 |
| orca_maps_expanded_1000_v1 | 800 / 100 / 100 | 137689 / 17359 / 16698 |
| orca_maps_official_usable_1000_v1 | 800 / 100 / 100 | 78149 / 10211 / 9129 |
| orca_social_pilot_v1 | 16 / 4 / 4 | 26597 / 6560 / 11118 |

- maps_resource_complete中Office/Warehouse/Full_Warehouse/Hospital四份CPU投影的composition_complete均true，physx_verified仍false；旧maps中两份warehouse仍为incomplete原型，应按版本区分。它们与本次四个collected室内场景的source-bound PhysX地图是不同资源。
- 外部Assets/DynBench保存这些参考场景、Materials/Props/SkyTexture等，Assets/NVIDIA通过两个有效软链接提供HDR天空图（真实文件在data/scene_resources/hdri），Assets/Sence保存旧场景/机器人目录；主链四场景原件在/project/EXP/Sence。完整尺寸/路径另列机器清单，不将不同目录同名场景自动等同。
- TRACE参考checkout的datasets目录当前只有README，nuScenes/ETH-UCY不在该目录；重训对应公开变体需另外配置其数据。PACER sample_data的LFS占位/实文件状态单独记录，不能替代当前固定v7动作库。
- third_party/trace=`594b2af…`，pacer=`baeb624…`；外部TraceAndPace目录是相同固定版本。原项目旧Python/Torch/IsaacGym依赖仅适用于原仓库环境。
- FLUX=`2189820…`、NavDP=`bff5eb8…`是独立参考分支；现有行人主链不从这两者获取权重。NavIsaaclab2.0=`d742f0c…`，其protomotions属于同一仓库目录。
- 根无.gitmodules，third_party被ignore/prune；Git源码完整合并不等于所有嵌套参考checkout都随clone存在。[外部commit/remotes](resource_audit_20261010/inventory.json)

## 8. 远程资源与成片

**GitHub**：自有私有资源包已有独立取回校验证据；467manifest行+manifest共468逻辑文件，2.687GB（不含manifest），LFS去重207对象约2.599GB。原上游LFS默认端点与自有发布端点分开；详见[ASSET_RELEASE](../ASSET_RELEASE.md)与[发布receipt](ASSET_PUBLICATION_20261010.json)。

**Hugging Face**：计划仓库GhostDGGG/Socially-aware_Indoor_Navigation_Benchmark。本地publication_status仍记录prepared_upload_waiting_for_valid_cli_authentication、0上传文件；本轮只读插件返回not found，不能区分不存在与权限问题。**[TODO]远程HF上传/文件身份尚未核实，不能当作可靠恢复来源。** 本轮不读取令牌、不再次上传。

最新四片仍在本任务目录：

```text
/home/lfm/Documents/ChatGPT/Social Navigation/
  trace_pace_reproduction/speed_coverage_optimization/videos_final_20261010_v1/
    YOHO_60s.mp4
    NewMetro_60s.mp4
    SecondFloor_60s.mp4
    ThirdFloor_60s.mp4
```

本轮四片文件身份/ffprobe确认960×720、30FPS、1800帧、60秒；以前全量解码与物理验收另见交付索引，此次未再次全解码。视频不在上述GitHub资源包；没有EVT机器人参与，不能称耦合机器人社交导航成果。[交付说明](FOUR_SCENE_SPEED_20261010_ZH.md)

## 9. 磁盘与迁移保留清单

| 树 | 逻辑文件数 | 逻辑大小（GB十进制） |
| --- | ---: | ---: |
| 仓库data | 12528 | 17.11 |
| resources | 468 | 2.69 |
| 仓库output | 27189 | 170.11 |
| 外部Assets/motion | 37918 | 81.64 |
| src/protomotions/data/assets | 271 | 0.097 |
| third_party | 1090 | 0.928 |

统计不跟随目录/文件符号链接，排除.git/.venv/cache等；不是du物理块，也不是可去重相加的总量。data/smpl参数由外部链接提供，因此其真实参数字节不计入data树。

迁移现有成果至少保留：选定TRACE/config/两AMP/教师/父初始化、v7与三split银行、人体物理资产和授权SMPL参数/视觉资产、完整USD引用、匹配的map/probe/source/SDK生命周期档案、训练恢复状态/配方、真实物理源/质量报告/相机/录像receipt。data里其他权重、旧AMASS版本和output历史训练可列为候选归档，但没有检查引用前不要删除；docs中的runtime helper也不能整体清空。

## 10. 未完成事项与验证界限

1. 环境profile与SDK HEAD的apply门禁漂移需协调；目标Sim6.1迁移、scene队列/生命周期另行修正与原生验收。
2. 远程HF资源状态未核实；GitHub文件恢复已具证据，但授权AMASS/SMPL/SDK需独立准备。
3. 新位置USD/地图准入与跨主机安装尚未完成一键封装；第三方checkout和外部SMPL软链接是恢复缺口。
4. 作者定制五状态Diffusion/AMP权重、精确2848动作清单与八场景评估列表在所核查资源中仍未获得；当前v7是本地真实AMASS重建，不冒充相同作者ID。
5. 全场景覆盖、高密度/多种子泛化、非平坦地面、接触斑滑移/摩擦、EVT闭环社交导航及完整论文指标协议尚未验收。
6. 全套测试存在先前失败；本轮983文件AST与局部资源gate通过不能替代全套绿灯或动作真实感的人类评价。

此次没有需要用户额外回答的意图问题；没有安装/升级/清理/推送。原有terrain_material.usda修改保留。

## 11. 可追溯文档

- [STACK](codebase/STACK.md)：全部core/extras、实际依赖与环境。
- [STRUCTURE](codebase/STRUCTURE.md)：目录、入口、链接、包分发边界。
- [ARCHITECTURE](codebase/ARCHITECTURE.md)：算法与运行资源关系。
- [CONVENTIONS](codebase/CONVENTIONS.md)：源码、导入、日志、配置约定。
- [INTEGRATIONS](codebase/INTEGRATIONS.md)：外部仓库、平台、授权数据与下载链。
- [TESTING](codebase/TESTING.md)：本轮检查和历史物理结果的边界。
- [CONCERNS](codebase/CONCERNS.md)：风险、迁移缺口与待核实项。
- [inventory.json](resource_audit_20261010/inventory.json)、[resource_checks.json](resource_audit_20261010/resource_checks.json)、[environment_checks.json](resource_audit_20261010/environment_checks.json)、[检查过程](resource_audit_20261010/COMMANDS.md)。
