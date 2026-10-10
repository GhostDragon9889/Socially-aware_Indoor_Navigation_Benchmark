# NavIsaacLab 行人模拟资源复核

> 后续迁移已接入本地 SMPL neutral，并完成四人8秒物理、蒙皮及视频验收；本篇保留接入前的资源复核。
> 当前必需资源、运行步骤和仍缺的论文资源见 [Isaac Sim6 / IsaacLab3 迁移说明](ISAAC_SIM6_LAB3_MIGRATION_ZH.md)。UV/纹理与严格论文 Diffusion+AMP 的缺口仍存在。

审查日期：2026-10-01。这里将用户所说的 navisaacsim 理解为本地 NavIsaacLab 论文及 CrowdSim 行人模拟模块。

## 结论

**严格复现论文的 pedestrian modeling 仍有关键缺口：论文对应的轨迹扩散模型、SMPL AMP 控制器及配套配置、2,848 段动作的精确筛选清单，以及轨迹模型训练数据与预处理说明尚未找到可确认对应的本地产物。**

**当前 SFM＋MaskedMimic 实现的控制权重和通用 AMASS 动作库已经具备。** 其人体外观资源尚未接入；本次在相邻项目发现真实 SMPL 模型，不能再把 SMPL 参数模型笼统列为“本机没有、必须下载”。UV 和人体纹理仍未在当前加载路径及已发现的人体资源目录中找到。

补齐资源后还需要解决已有运行时接口、四元数及 reset 问题。本轮未启动 Isaac Sim，不能据文件存在判定仿真已通过。

## 1. 范围与证据强度

- 递归检索 `/project/EXP/SangforAndSaut` 与 `/project/EXP/NavIsaaclab2.0`，覆盖当前 ProtoMotions、两份 NavIsaaclab2.0、Assets 及相邻项目的资源候选。排除 Git、缓存和部分环境目录；不跟随目录符号链接，不穷举压缩包内部。
- 对当前入口要求的文件逐项检查存在性、大小、LFS 指针；对两份 SMPL checkpoint 及各自 inference config 做 SHA-256 和 ZIP CRC 检查，全部通过。没有将配置或未知 pickle 作为代码执行。
- AMASS 本次复核路径、大小；全量张量、四元数、帧索引、逐动作插值检查来自本会话此前已完成的 CPU 验证，未重复加载约 9.6 GB train。
- 论文依据是仓库内 PDF 的 III-B、IV-A/B/C；公开摘要亦明确使用轨迹扩散和对抗运动控制：[NavIsaacLab](https://arxiv.org/abs/2606.26265)。
- 未启动 GPU 推理、实际蒙皮或 USD 场景组合；其他目录中任意命名文件、未展开压缩包或远程存储仍可能含资源。“未找到论文对应产物”不是断言作者没有公开。

原始索引：[pedestrian_resources_20261001.json](resource_audit/pedestrian_resources_20261001.json)。它包含环境 `.pth` 钩子和无关模型候选，**不能将条目数当作有效模型数**。

本次检查结果：[pedestrian_resource_checks_20261001.json](resource_audit/pedestrian_resource_checks_20261001.json)。

## 2. 当前代码与论文方法的差别

| 层次 | 当前实际入口 | 论文要求 |
|---|---|---|
| 全局导航 | A* 与场景占据地图 | A* 与场景占据地图 |
| 局部行人轨迹 | `CrowdSim/nav_manager.py` 调用 `Social_Force` | 以历史、邻居轨迹和地图为条件的 U-Net 轨迹扩散，预测 5 秒轨迹 |
| 全身控制 | `masked_mimic/smpl/last.ckpt`，`SupervisedAgent` / `MaskedMimicModel` | 跟随二维轨迹、考虑障碍物的 SMPL AMP 策略 |
| 参考动作 | 已处理的项目通用 AMASS splits | 按 locomotion 标签筛选的 2,848 段 AMASS 动作 |
| 人体表面 | `smplx.create(model_type="smpl", gender="neutral")` | SMPL 参数生成的蒙皮表面 |

代码依据：`CrowdSim/config/env.yaml:7`、`CrowdSim/nav_manager.py:21`、`:769`、`CrowdSim/protomotions_runtime.py:106`、`data/pretrained_models/masked_mimic/smpl/resolved_configs_inference.yaml:2075`。

因此，当前实现也不能直接称为论文 SFM＋AMP 消融版本：其低层控制器仍是 MaskedMimic。

## 3. 已有、无需重复下载的核心资源

### 3.1 AMASS

根目录：`/project/EXP/SangforAndSaut/Assets/motion/`。

| 文件 | 动作段数 | 帧数 | 当前字节数 |
|---|---:|---:|---:|
| `amass_smpl_train.pt` | 11,303 | 4,331,631 | 9,565,828,107 |
| `amass_smpl_validation.pt` | 345 | 307,046 | 678,009,566 |
| `amass_smpl_test.pt` | 138 | 29,705 | 65,614,188 |

合计 11,786 段、4,668,382 帧，24 bodies、69 DOF、xyzw；当前配置使用 validation 文件。22 个原始压缩包保留，通用 splits 所需的 19 个包已解压、转换。GRAB、SOMA、HUMAN4D 未作为这套 splits 的必要输入。

这些数量不证明覆盖论文的 2,848 段。还需原始相对路径、片段起止帧、标签、过滤规则、采样权重和划分。没有这些信息，不能判断当前原始包是否含论文全部参考片段，也不应继续无目标地下载 AMASS。

此前验证：[amass_summary.json](resource_audit/amass_summary.json)、[AMASS 准备说明](AMASS_PREPARATION_ZH.md)。

### 3.2 控制权重及配置

| 资源 | 当前状态 | 用途 |
|---|---|---|
| `data/pretrained_models/masked_mimic/smpl/last.ckpt` | 229,131,184 字节；容器 CRC 通过 | 当前 CrowdSim 行人控制 |
| 同目录 `resolved_configs_inference.pt` | 46,693 字节；容器 CRC 通过 | 实际入口依赖的推理配置 |
| 同目录 training config、YAML、experiment config | 存在 | 配套配置与可读审计依据 |
| `data/pretrained_models/motion_tracker/smpl/last.ckpt` | 126,361,575 字节；容器 CRC 通过 | SMPL 参考动作跟踪 |
| tracker 的 `resolved_configs_inference.pt` | 39,893 字节；容器 CRC 通过 | tracker 配套配置 |
| `protomotions/data/assets/mjcf/smpl_humanoid.xml` | 23,780 字节；真实 XML | 当前 checkpoint 指定的物理骨架 |

MJCF 没有外部 mesh、texture、include 文件引用。`smpl_humanoid.usda` 是 131 字节 LFS 指针，但当前 checkpoint 选择 MJCF；不能据这个 USD 指针认定当前骨架缺失。

容器 CRC 通过只证明文件容器可完整读取，不证明权重与当前代码能成功实例化或闭环推理。

### 3.3 相邻项目中的 SMPL 参数模型：本次新增发现

目录：

```text
/project/EXP/SangforAndSaut/arena_sangfor/arena_ws/arena_assets/smpl/
└── SMPL_python_v.1.1.0/smpl/models/
    ├── basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl  # 247,186,228 bytes
    ├── basicmodel_m_lbs_10_207_0_v1.1.0.pkl        # 247,101,031 bytes
    └── basicmodel_f_lbs_10_207_0_v1.1.0.pkl        # 247,530,000 bytes
```

对 neutral 做了不执行对象的 pickle 操作码检查，找到 `v_template`、`shapedirs`、`posedirs`、`J_regressor`、`kintree_table`、`weights`、`f`；STOP 正好位于文件末尾。它包含 `chumpy.ch.Ch` 及旧 NumPy/SciPy 类引用，仍需验证旧格式的加载兼容性或进行转换，不能承诺简单改名后立即可用。

当前代码接受 `data/smpl/SMPL_NEUTRAL.pkl` 或 `data/smpl/smpl/SMPL_NEUTRAL.pkl`，两处目前为空。建议后续把这个本地 neutral 模型兼容处理后接入前一路径。模型源文件无需移动或删除。当前代码固定 neutral，male/female 并非启动必需。

相邻目录还存在 SMPL-H male/female/neutral `model.npz`、CMU 动作与派生动画 USD；这些不能直接替代当前 SMPL 24-body 配置。

证据：[local_smpl_candidate_20261001.json](resource_audit/local_smpl_candidate_20261001.json)。如果本地模型最终不可用，官方模型来源是 [SMPL 官网](https://smpl.is.tue.mpg.de/)。

## 4. 仍缺少或尚未确认的资源

| 项目 | 状态 | 必要性与应补齐的内容 |
|---|---|---|
| 论文轨迹扩散模型 | 未找到可确认对应的权重与接入 | 严格论文推理必需；checkpoint、模型配置、归一化、历史/预测时长、时间步、地图编码、邻居编码和推理调度 |
| 论文 SMPL AMP 控制器 | 未找到可确认对应的权重与任务配置 | 严格论文推理必需；actor、观测统计、关节映射、动作缩放、控制频率、轨迹与障碍物观测接口 |
| 精确 2,848 段 AMASS manifest | 未找到 | 从头训练及复现实验数据口径所需；已有通用 splits 不能替代这个证据 |
| 轨迹扩散训练集和预处理 | 论文所用来源、版本与划分未确认 | 从头训练必需；带地图、邻居交互及历史/未来轨迹的数据，单人 AMASS 骨架动作不等价于该训练集 |
| AMP 训练定义 | 核心存在通用 AMP 能力，未确认论文对应配置 | 从头训练还需判别器输入、奖励、终止、初始化、采样、超参数；恢复训练需训练 checkpoint/优化器状态 |
| `data/smpl/smpl_uv.npz` | 缺失 | 贴图所需；必须有与 SMPL 拓扑相符的 `uvs`、`uv_faces` |
| 人体贴图 | 当前配置路径与已发现人体模型目录未找到 | 逼真衣着/皮肤渲染所需；可用 `data/smpl/SMPLitex-textures/` 或指定纹理目录 |
| SMPL neutral 的项目接入 | 本地已有旧格式，当前入口未接入 | 开启 `human_mesh: true` 时所需；还需 `smplx` 及旧模型兼容验证 |

UV/贴图不影响无纹理人体或纯物理关节控制；缺少 SMPL 参数模型则无法创建该 SMPL 蒙皮。当前默认 `human_mesh: true`、`hide_humanoid: true`，需要验收实际可见人体，不能把“程序未退出”视为外观已成功加载。

`smpl_mesh_visualizer.py` 注释提到的 `tools/generate_smpl_uv.py` 在当前仓库不存在，不能将其当作可直接执行的补齐步骤。

**AMP 判别器用于训练时的风格奖励；只执行已训练 actor 时不必额外加载判别器。** 从头训练不要求预先存在 AMP 权重，但必须有可复现的任务、数据和训练配置。

### 不应误用的已有模型

- `motion_tracker/soma_bones_fsq_amp_muon/` 确有 AMP 权重，但对应 SOMA 23 bodies / 66 actions、BONES-SEED 与 FSQ。当前 SMPL 是 24 bodies / 69 actions，不能直接替代。
- `motion_tracker/smpl` 是参考动作跟踪 PPO；它的存在不能证明论文的二维导航目标 AMP 控制器已经具备。
- `gpc_prior/soma_bones` 是另一模型体系，不是论文所述行人轨迹扩散 checkpoint。
- FLUX、NavDP 及机器人 PPO 权重服务于机器人导航，不是单独生成行人姿态的必需权重；MonoLoco、YOLO 等感知模型同理。

论文引用的 [TRACE 官方仓库](https://github.com/nv-tlabs/trace) 提供其自身预训练模型及数据说明，包括 ORCA、nuScenes 和 ETH/UCY。它可以作为实现来源，但不能据此认定 NavIsaacLab 用了全部这些数据，或其 checkpoint 能直接接入当前 CrowdSim。TRACE 与 PACER 也是两个分别发布的组件。

## 5. 场景资源：按使用场景区分

默认 warehouse 的入口 USD、`CrowdSim/maps/newmap.yaml` 和 `World1.png` 都存在，地图与几何的实际对齐尚未通过仿真验收。此前递归 USD 依赖记录中的缺失目标，本次复核仍不存在：

```text
../Assets/DynBench/Props/KLT_Bin/small_KLT_visual_collision.usd
../Assets/DynBench/Props/Forklift/forklift.usd
../Assets/NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr
../Assets/NVIDIA/Assets/Skies/Clear/sunflowers_4k.hdr
```

默认仓库依赖记录涉及 KLT；其他条目随场景而变化。入口文件存在不代表所有引用已就绪；移除某些 prim 是否避开缺失引用需以实际场景组合为准。

四个 DynBench 场景的地图仍缺：

```text
../Assets/DynBench/isaacsim_scene/Full_Warehouse/map/occupancy.yaml
../Assets/DynBench/isaacsim_scene/Hospital/map/occupancy.yaml
../Assets/DynBench/isaacsim_scene/Office/map/occupancy.yaml
../Assets/DynBench/isaacsim_scene/Warehouse/map/occupancy.yaml
```

需要生成各自匹配的 YAML＋栅格图，而不是复制同一地图。它们不是默认 warehouse 起步的全部前置条件；要复现跨场景结果才需要逐场景补齐。论文指定场景/评估 episodes、随机种子及姿态多样性评估的聚类与参考统计也尚未确认对齐。

场景依赖来源：[scene_dependencies.json](resource_audit/scene_dependencies.json)。本轮对已知缺失目标重新 stat，没有重做完整 USD 引用图扫描或远程 NVIDIA 资产可达性测试。

## 6. 环境与代码仍有独立阻塞

新确认相邻环境 `/project/EXP/SangforAndSaut/IsaacLab/env_isaaclab` 安装元数据：Python 3.12.13、Isaac Sim 6.0.0.1、IsaacLab 6.1.11、Torch 2.10.0+cu128；未安装 `smplx`、`lightning`。本项目 `pyproject.toml` 的 isaaclab extra 声明 IsaacLab 12.0.0、Torch >=2.11,<2.12。该环境存在，但不能当作已经匹配当前仓库的环境。

此前代码审查已确认与行人链路直接相关的阻塞：

1. `protomotions_runtime.py:122` 导入不存在的 `apply_backward_compatibility_fixes`。
2. 当前 xyzw 数据与人体网格/部分朝向逻辑的 wxyz 假设冲突。
3. 子集 reset 的人体目标没有按全局 env ID 正确索引。
4. 评估入口存在不更新人体观测、未消费人体 done 的问题。

详细证据与复现见 [代码审查报告](CODE_REVIEW_20261001_ZH.md)。补下载文件不会修复这些问题。

## 7. 建议执行顺序

1. **跑通当前实现**：修复上述接口与状态问题，选择匹配环境，使用已有 MaskedMimic＋validation MotionLib 进行单行人、无贴图、简单场景闭环验收。
2. **恢复人体外观**：兼容处理本地 neutral PKL，接入 `data/smpl/SMPL_NEUTRAL.pkl`，检查蒙皮与物理骨架姿态；再补 UV 和人体纹理。
3. **严格对齐论文**：取得作者对应的 diffusion/AMP checkpoint、完整配置、2,848 段 manifest、轨迹数据来源/预处理及评估协议；若拿不到，需明确这是自行重建的方法并开展相应训练。
4. **扩展场景与指标**：补场景引用和占据地图，验证地图对齐，再做多行人避障、摔倒率、轨迹误差与姿态多样性评估。

本次只新增审计报告与证据，没有移动模型、修改业务代码、下载大型数据或更改仿真环境。
