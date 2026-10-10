# 资源、数据与权重复核（2026-10-01）

本次重新检查当前 ProtoMotions、相邻 `Assets`、`arena_sangfor` 的 SMPL 模型和两份 NavIsaaclab2.0。所有验证使用 CPU；未启动 SimulationApp、下载资源、移动数据或执行未知 pickle。机器证据见 [resources.json](resources.json)。

## 可确认的结论

当前 SFM＋MaskedMimic 行人控制链路所需的 **SMPL checkpoint、配套推理配置、MJCF 骨架和通用 AMASS 数据均已具备**。本次不仅检查文件存在，还在新 Python 3.12／Torch 2.11 环境验证了实际模型权重加载及 AMASS 插值。

默认 CrowdSim 配置仍不能据此认定已可运行：默认机器人 PPO 权重缺失，人体网格默认目录缺模型，默认仓库有一个缺失 USD 引用，另有代码接口与状态处理问题。严格复现 NavIsaacLab 论文还需要对应的轨迹扩散、SMPL AMP 及精确动作清单。

## 1. 模型权重与配置：8 套严格匹配通过

扫描 `data/pretrained_models/` 找到 8 个 `last.ckpt`；各目录同时存在 `resolved_configs_inference.pt` 和 YAML。使用 `torch.load(weights_only=True, map_location="cpu")` 加载 checkpoint；推理配置使用明确白名单的本地配置 dataclass、enum、字段恢复对象与纯张量 MDP 函数。没有退回到不受限 pickle 加载。

随后实例化现有模型，按实际 `BaseAgent._load_model_state_dict()` 的顺序初始化 lazy 归一化及模块，再执行 `load_state_dict(strict=True)`。**8／8 成功，没有缺失键、额外键或形状冲突。**

| 预训练目录 | checkpoint 字节数 | 配置骨架 bodies／DOF | 策略类型 | 严格匹配 |
|---|---:|---:|---|---|
| `masked_mimic/smpl` | 229,131,184 | 24／69 | MaskedMimic／Supervised | 通过 |
| `motion_tracker/smpl` | 126,361,575 | 24／69 | PPO motion tracker | 通过 |
| `motion_tracker/smpl-terrains` | 132,661,223 | 24／69 | PPO terrain tracker | 通过 |
| `motion_tracker/soma-bones` | 125,291,239 | 23／66 | PPO motion tracker | 通过 |
| `motion_tracker/soma_bones_fsq` | 168,493,309 | 23／66 | PPO／FSQ | 通过 |
| `motion_tracker/soma_bones_fsq_amp_muon` | 233,783,352 | 23／66 | AMP／FSQ | 通过 |
| `motion_tracker/g1-bones-deploy` | 238,154,731 | 33／29 | G1 AMP | 通过 |
| `gpc_prior/soma_bones` | 2,415,496,421 | 23／66 | 自回归 latent prior | 通过 |

对默认 SMPL MaskedMimic 和 SMPL PPO 两套权重另做了 ZIP CRC 与全量模型浮点张量有限性检查，均通过。其他模型本次验证结构加载，没有穷举所有浮点张量的有限性。部分模型还有 `inference_last.ckpt`、编译 ONNX 或 `last_lab.ckpt`；这些文件存在性记录在 JSON，不能将上述 `last.ckpt` 的验证扩大为每种部署产物均已验收。

骨架依据来自实际 `.pt` 配置及 MJCF。8 套模型使用的 MJCF 均存在；SMPL／SOMA 不依赖外部 mesh，G1 使用的 35 个外部 STL 引用全部存在。仅验证加载不能证明观测、时间步、四元数、环境 reset 或物理行为正确。

## 2. P1：SMPL tracker 的默认动作文件指向 G1 数据

文件：`data/pretrained_models/motion_tracker/smpl/resolved_configs_inference.yaml:1579`，以及 `smpl-terrains` 同名推理配置。

两套 SMPL 配置中的 `motion_lib.motion_file` 都为如下路径；已从实际 `resolved_configs_inference.pt` 安全加载核实，并非仅 YAML 展示过时：

```text
data/motion_for_trackers/g1_random_subset_tiny.pt
```

本次安全加载此文件，实际 `gts=[16252,33,3]`、`dps=[16252,29]`，即 G1 33 bodies／29 DOF；SMPL 模型要求 24／69。`protomotions/inference_agent.py:92` 的 `--motion-file` 是可选参数，不传时会沿用 checkpoint 配置，因此直接使用两套 SMPL tracker 默认配置存在实际骨架不一致风险。

需要明确传入 SMPL 动作库，例如 `../Assets/motion/amass_smpl_validation.pt`。**CrowdSim 在 `protomotions_runtime.py:159` 覆盖动作文件，已避开这个默认值问题。** 本轮未修改已下载的 checkpoint 或配置。

其他随仓库携带的动作样本已用安全加载核对：G1 seed 是 33／29，SOMA seed 和 crouch 是 23／66，H1 是 29／27。它们需要匹配各自机器人。

## 3. AMASS：数据已完成，新环境可读

目录：`/project/EXP/SangforAndSaut/Assets/motion/`。

| 文件 | 动作段 | 帧数 | 当前字节数 |
|---|---:|---:|---:|
| `amass_smpl_train.pt` | 11,303 | 4,331,631 | 9,565,828,107 |
| `amass_smpl_validation.pt` | 345 | 307,046 | 678,009,566 |
| `amass_smpl_test.pt` | 138 | 29,705 | 65,614,188 |

三份文件本次 stat 大小均与此前全量验证一致，总共 11,786 段、4,668,382 帧。此前验证包括张量有限性、24／69 形状、xyzw 四元数、帧索引、每段动作起点／37.1%／终点插值；本次没有再次加载或哈希约 9.6 GB train。

本次额外在 **Python 3.12.13／Torch 2.11.0+cu128** 环境中通过现有 MotionLib 读取 test 文件，强制其内部 `torch.load` 使用 `weights_only=True`；138 段全部完成三个时间点的 CPU 插值，位置、姿态和 DOF 形状正确且有限。

原始目录 `Assets/motion/raw/` 的 22 份 `.tar.bz2` 均保留；通用 splits 所需的 19 份已处理。GRAB、SOMA、HUMAN4D 是保留的额外包。原先 `converted_train.*` 和 H36M17joint 数据不能直接替代 SMPL MotionLib，但已经有正确转换输出，无需为当前 SFM＋MaskedMimic 重复下载 AMASS。

这些计数不能证明与论文按 locomotion 筛选的 2,848 段完全一致。相关原始相对路径、片段区间、标签、过滤和划分清单仍未确认。

## 4. P1／条件：SMPL 人体外观模型未接入

当前 `CrowdSim/config/env.yaml:7` 启用 `human_mesh: true`。加载器使用 `smplx.create(model_type="smpl", gender="neutral", num_betas=10)`，默认查找：

```text
/project/EXP/SangforAndSaut/ProtoMotions/data/smpl/SMPL_NEUTRAL.pkl
/project/EXP/SangforAndSaut/ProtoMotions/data/smpl/smpl/SMPL_NEUTRAL.pkl
```

两个位置均缺失。`smplx` 软件包已安装不代表参数数据随之安装。模型缺失阻碍人体蒙皮外观，物理 MJCF 控制本身有独立资源。

本机已有真实 neutral、male、female 候选文件：

```text
/project/EXP/SangforAndSaut/arena_sangfor/arena_ws/arena_assets/smpl/
SMPL_python_v.1.1.0/smpl/models/
basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
```

neutral 文件 247,186,228 字节，male 247,101,031 字节，female 247,530,000 字节。本次仅用 `pickletools` 检查 neutral 操作码，确认 `v_template`、`weights`、`shapedirs`、`posedirs`、`J_regressor`、`kintree_table`、`f` 等字段存在；它引用 Python 2、Chumpy 和 SciPy 稀疏类型。**尚未实际反序列化、蒙皮或验证旧模型兼容性，应优先兼容并接入该候选。** 当前固定 neutral，male／female 不是必要条件。

以下贴图相关资源在当前加载位置仍缺：

```text
data/smpl/smpl_uv.npz                 # 需 uvs、uv_faces 两个数组
data/smpl/SMPLitex-textures/          # 或指定其他人体纹理目录
data/smpl/smpl_body_texture.png
data/smpl/smpl_uv_20200910.png
```

UV／纹理影响真实表面外观，纯色网格可继续使用。`smpl_mesh_visualizer.py:369` 提及的 `tools/generate_smpl_uv.py` 在项目中不存在，不能按该提示当作可直接执行的修复脚本。

## 5. 场景与地图：4 张地图、4 个唯一外部目标缺失

本次使用 Sdf 递归读取 6 份场景配置引用的场景，检查 composition references、sublayers、asset 类型默认属性和时间样本；共涉及数千个层文件。没有创建完整 USD stage，因此 shader 编译、物理碰撞或布局仍需后续运行验收。

| CrowdSim 场景配置 | 场景 USD | 地图／图片 | 缺失外部目标 |
|---|---|---|---|
| `warehouse` | 存在 | `newmap.yaml`／`World1.png` 存在 | KLT USD |
| `bidirectional_open` | 存在 | 地图／图片存在 | KLT USD |
| `dynabench_full_warehouse` | 存在 | occupancy.yaml 缺 | Forklift USD |
| `dynabench_hospital` | 存在 | occupancy.yaml 缺 | Cloudy HDR |
| `dynabench_office` | 存在 | occupancy.yaml 缺 | Clear HDR |
| `dynabench_warehouse` | 存在 | occupancy.yaml 缺 | KLT USD |

四份地图准确路径：

```text
/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Full_Warehouse/map/occupancy.yaml
/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Hospital/map/occupancy.yaml
/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Office/map/occupancy.yaml
/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Warehouse/map/occupancy.yaml
```

四个唯一缺失资源：

```text
/project/EXP/SangforAndSaut/Assets/DynBench/Props/KLT_Bin/small_KLT_visual_collision.usd
/project/EXP/SangforAndSaut/Assets/DynBench/Props/Forklift/forklift.usd
/project/EXP/SangforAndSaut/Assets/NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr
/project/EXP/SangforAndSaut/Assets/NVIDIA/Assets/Skies/Clear/sunflowers_4k.hdr
```

地图缺失会在加载对应 scene 配置时阻断导航初始化；场景 USD 主文件存在不能证明引用齐全。默认场景只涉及上述 KLT 缺口，不能将另三个场景的缺口全部归为默认启动必要条件。现有地图与实际 USD 的坐标、移除的 Shelf／PalletBin、代码生成障碍物是否一致，本轮未验收。

## 6. P1／条件：默认机器人导航权重缺失

`CrowdSim/config/env.yaml:27` 默认 `car.rl_policy: true`，并指定：

```text
output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt
```

本次该路径及 `output/` 目录均不存在。`CrowdSim/crowd_sim.py:183` 实际调用 `ppo.load()`，默认机器人策略推理需要训练产物或有效的外部权重路径。机器人 PPO 从头训练不需要预先有这份输出，单独行人控制也不需要它。

本地 `Assets/Robots/NovaCarter/nova_carter.usd` 亦不存在。代码可回退到 IsaacLab 的官方远程资产根；这是**远程可用性未验证**，不是已经判定无法使用。离线运行需要补齐其本地资产及引用。

## 7. LFS：49 个 USD 指针需要按实际路径处理

扫描当前 `protomotions/data/assets`、`data`、`examples/data`、FLUX checkpoint 与 NavDP，共发现 **49 个 Git LFS 指针**，都位于 ProtoMotions USD 资产目录，详细 oid／目标长度见 JSON。其中包括 `smpl_humanoid.usda`、`smplx_humanoid.usda`、`terrain_material.usda`、marker、projectile，以及 G1／H1／SOMA 的 USD 子层。

8 套当前预训练配置走 MJCF 路径，外部 mesh 依赖已核对，所以不能因 `smpl_humanoid.usda` 是指针直接断定默认 SMPL 物理骨架缺失。选择这些 USD 资产、相关可视化或 projectile／其他路径时仍需补齐真实 LFS 文件。

相邻 `Assets` 扫描 73,946 个文件、141,539,232,060 字节，没有发现 LFS 指针；总大小含原始包、解压文件和转换输出，不能当作独立必需数据量。两份 NavIsaaclab2.0 参考目录重新检索未发现 `.ckpt/.pt/.pth/.onnx/.safetensors/.bin` 权重产物。

## 8. FLUX／NavDP：条件资源与部署状态

`FLUX/checkpoints/flux_v1.ckpt` 为真实文件，543,338,953 字节。安全加载得到 1,062 个张量、135,722,756 个元素；全量浮点张量有限，且有 RGBD ViT 骨干参数。新建 IsaacLab 环境缺少 `diffusers`，本次不能直接构造其完整 CFM 模型，因此没有完成 FLUX 模型严格加载／HTTP 推理验收。

**FLUX 权重到位不能证明部署服务可运行。** 主审查另已复现服务端图像／depth 维度处理问题，需结合代码审计处理。`--normalization-config` 默认为 None，对动作尺度的训练约定也尚未确认。

NavDP 当前没有找到实际模型权重；其原始 server 默认指向原作者机器的 `/home/PJLAB/.../cross-waic-final4-125.ckpt`，FLUX 内 NavDP 基线默认引用 `checkpoints/navdp-cross-modal.ckpt`，这两处当前均不可用。运行其他 GNM／ViNT／NoMaD／iPlanner／ViPlanner／DD-PPO 基线需要各自权重，不能拿 `flux_v1.ckpt` 替代。

FLUX／NavDP 自带 assets 主要是图片、示例 USD 和 episode `.npy`，不是完整 InternScene 场景数据。它们的 README 所需 benchmark 场景、Materials／SkyTexture 及其他基线权重应按选定评估任务另行补齐；本轮没有把无关导航评估数据列为单独行人模拟的必需项。

## 9. 严格论文复现：未确认对应产物

当前入口是 SFM＋MaskedMimic，而论文的 pedestrian modeling 使用条件轨迹扩散及 AMP。当前 8 套模型、FLUX／NavDP 候选和两份参考仓库中，未找到可确认与论文一致的以下产物：

| 产物 | 何时必需 | 尚需确认 |
|---|---|---|
| 条件轨迹扩散 checkpoint | 论文方法推理 | 网络、条件编码、归一化、采样步数、预测时域及代码接入 |
| SMPL AMP actor | 论文方法物理执行 | SMPL 24／69 映射、观测统计、二维轨迹跟踪任务及训练配置 |
| 2,848 段 AMASS 动作清单 | 精确数据复现 | 源文件、起止区间、locomotion 标签、过滤、权重及 splits |
| 轨迹训练数据和预处理 | 从头训练轨迹模型 | 数据来源、地图及邻居历史构造、划分、尺度约定 |
| AMP discriminator 与奖励设置 | 从头训练 AMP | 判别器输入、风格采样、奖励／终止规则、训练配置 |
| 论文完整场景与评估 manifest | 重现实验指标 | 地图／场景／episode 与论文设置逐项对应 |

SOMA AMP 是 23／66，G1 AMP 是 33／29，GPC prior 是 SOMA latent 模型；它们虽均严格匹配本项目模型类，不能据名称替代论文 SMPL 行人控制器或二维轨迹扩散。仅运行预训练 AMP actor 通常不必单独加载训练用 discriminator。AMASS 是单人动作来源，不能直接证明已有多行人条件轨迹训练数据。

“未确认对应产物”是此次本地审计结论；不表示作者未公开，也不穷举尚未展开的压缩包或远程存储。

## 10. 建议处理顺序

1. 修复当前 CrowdSim 运行时接口、四元数、subset reset 与评估状态问题，使用已有 MaskedMimic＋AMASS 做单行人物理闭环。
2. 从已发现的 neutral 候选安全兼容并接入 SMPL 外观，纯色模式先验证；真实外观再补 UV／纹理。
3. 处理当前选择场景的缺失引用、地图对齐；机器人 PPO 推理补训练产物，离线执行补 NovaCarter。
4. 使用 SMPL tracker 时显式覆盖 SMPL motion 文件；按真正用到的资产补 Git LFS。
5. 若目标仍是严格论文复现，获取或重训对应轨迹扩散与 SMPL AMP，并落实精确动作／场景清单。

上述资源加载验收不替代 GPU 仿真、人体蒙皮、相机 depth、碰撞及完整导航评估。
