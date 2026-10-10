# NavIsaacLab 当前资源完整性审查

> 本文是2026-09-30资源快照。后续 SMPL neutral 已接入，最小行人场景及其地图已创建，四人8秒模拟通过。
> 当前资源与验收以 [Isaac Sim6 / IsaacLab3 迁移说明](ISAAC_SIM6_LAB3_MIGRATION_ZH.md) 为准；四套 DynBench 地图、相关场景引用和严格论文资源仍按所选任务适用。

日期：2026-09-30，数据迁移之后。范围：整个 ProtoMotions 工作区（含 FLUX、NavDP 的本地资源）及相邻 `Assets/`；排除 Git 数据库、下载缓存、Python 缓存及重复的目录符号链接。本报告的当前资源结论更新此前的缺失场景判断。

## 一、结论

**动态场景主体已就位，不需要重新下载整套 DynBench。通用 AMASS train/test/validation MotionLib 已转换并验证；当前仍缺 SMPL 外观参数、四套地图、四个场景依赖文件，以及严格论文复现的数据与权重。AMASS 处理最新状态见 [数据准备记录](AMASS_PREPARATION_ZH.md)；第十节保留解压前目录解析。**

不同缺口适用范围不同：地图只阻断选择对应场景的配置；HDR 主要影响视觉一致性；预训练机器人 PPO 是训练产物；论文数据子集和扩散/AMP 权重关系到严格复现。

## 二、审查覆盖与方法

- 工作区文件 1,858 个，Assets 文件 36,058 个；以上不计缓存及重复链接。
- 建立 6,658 条资源清单，涵盖模型、权重、动作、配置和 USD 等指定扩展名。
- 扫描 6,483 个 USD 候选：49 个是 LFS 文本指针，另外 6,434 个均可被 Sdf 打开。
- 查询层引用、asset 类型属性默认值和时间采样值，追踪六个测试场景入口的递归 USD 依赖。
- 共发现 19 条缺失本地引用，归并为 16 个唯一目标；其中实际测试场景可达的为 5 条引用、4 个唯一文件。其余属于 `.thumbs` 缩略图辅助层。
- 检查 232 个 MDL 文件中的 1,362 条字面量图片引用，按材质目录解析均存在；不等于完整 MDL 编译通过。
- 六个测试场景共 600 个 episode JSON，均可解析，具有 episode/robot/characters 基本结构；没有据此宣称任务与 CrowdSim 接口兼容。
- 检查关键权重的 ZIP 容器结构；未执行 pickle、加载模型或检验 tensor shape。
- 未启动 GPU 仿真、渲染、网络资产下载；没有修改资源和配置。

## 三、已具备的资源

| 资源 | 本地位置/状态 | 可用性边界 |
|---|---|---|
| DynBench | `../Assets/DynBench`，约 55.8 GiB（不计缓存） | 场景主体、材质、天空贴图、训练/测试文件存在 |
| 六个测试入口 | Full_Warehouse、Hospital、Jetracer、Office、Warehouse、Warehouse_multiple_shelves | 均有真实二进制 USD，不是 LFS 指针 |
| 测试任务 | 各 100 个 episode JSON | 包含机器人起终点、行人出生点与 commands；不是 AMASS 动作数据 |
| MaskedMimic 人体权重 | `data/pretrained_models/masked_mimic/smpl/last.ckpt` | 229,131,184 字节；容器可列出，未完成模型加载 |
| 配套人体配置 | 同目录 `resolved_configs_inference.pt` | 文件存在；此前已静态确认 69 个 ProtoMotions 符号引用可对应源码 |
| SMPL motion tracker 权重 | `data/pretrained_models/motion_tracker/smpl/last.ckpt` | 文件存在；不能当作论文 AMP 权重的证明 |
| 默认地图 | `CrowdSim/maps/newmap.yaml` → `World1.png` | 图片可读取，395×603；分辨率 0.05 m，未验证与场景几何对齐 |
| 人体物理骨架 | `protomotions/data/assets/mjcf/smpl_humanoid.xml` | MJCF 存在，XML 未引用外部 file 资源；支持后端生成 USD |
| 其他动作示例 | G1/H1-2/SOMA 小数据、单条 SMPL 坐姿、少量 SMPL-X 动作 | 可作相关示例，不是所需 SMPL 导航动作库 |
| FLUX 权重 | `FLUX/checkpoints/flux_v1.ckpt` | 属于 FLUX 导航模型，不替代 NavIsaacLab 人体/导航论文模型 |

`FLUX/assets/dynbench` 和 `../Assets/Simple_Warehouse` 均为兼容链接，指向已经迁移的数据。六份 CrowdSim 场景 YAML 均能找到相应 USD。

## 四、明确缺失：运行当前 CrowdSim 所需

### R1. SMPL AMASS MotionLib（默认验证集已补齐）

默认训练/采集配置引用：

```text
../Assets/motion/amass_smpl_validation.pt
```

2026-09-30 数据处理更新：该文件现已生成，含 345 段动作；138 段测试库与 11,303 段训练库亦已生成。三库通过当前 MotionLib 加载、全浮点字段有限值、索引/时长和逐动作插值检查。此前“文件不存在”的结论已失效。现有 G1/H1-2/SOMA 的 `.pt` 骨架不同，不能更名替代本次 SMPL 产物。

本次已将相关 AMASS 原始 `.npz` 转为 `.motion` 并打包，详细步骤和其余训练数据状态见 [数据准备记录](AMASS_PREPARATION_ZH.md)。仓库已有 `data/scripts/convert_amass_to_motionlib.py` 与 `data/yaml_files/amass_smpl_{train,test,validation}.yaml`；它们是转换工具和清单，不包含对应原始动作。

仅运行已有人体权重不一定需要下载完整 AMASS，但必须提供运行配置所需且兼容的动作库。从头训练及严格论文评测另需论文数据子集。

### R2. SMPL 参数模型

`data/smpl/` 只有 `smpl_joint_names.py`，未找到参数 `.pkl`。

默认 mesh 代码明确使用 SMPL、neutral，首先需要：

```text
data/smpl/SMPL_NEUTRAL.pkl
```

代码也支持 `data/smpl/smpl/SMPL_NEUTRAL.pkl`。当前 `convert_amass_to_proto.py` 使用固定 MJCF 骨架做前向运动学，不直接读取 neutral/male/female 参数 PKL；若改用按性别、体型生成网格的其他转换流程，则按该流程另行准备参数。当前默认路径不要求先准备 SMPL-X 全套模型。

这是模型参数，不是 pip 的 `smplx` 包；安装包不能提供这些文件。默认 `human_mesh=true`，缺模型会使 mesh 初始化失败后降级；进程继续运行不代表视觉数据达到预期。

### R3. 四套占据地图

下列 YAML 以及它们应引用的 PNG/PGM 地图均未找到：

```text
../Assets/DynBench/isaacsim_scene/Full_Warehouse/map/occupancy.yaml
../Assets/DynBench/isaacsim_scene/Hospital/map/occupancy.yaml
../Assets/DynBench/isaacsim_scene/Office/map/occupancy.yaml
../Assets/DynBench/isaacsim_scene/Warehouse/map/occupancy.yaml
```

需要从对应场景生成地图，记录 resolution、origin、障碍阈值并验证坐标对齐。不要用已有的 World1.png 直接替代不同场景；也不要只创建空 YAML。

默认 warehouse/bidirectional_open 使用已有 `newmap.yaml`，不因这四套地图缺失而必然阻断；其几何一致性仍需验证。

### R4. 场景递归依赖中的四个唯一缺失文件

以下路径以项目旁的 `Assets/` 为起点：

| 缺失文件 | 使用场景 | 影响 |
|---|---|---|
| `DynBench/Props/Forklift/forklift.usd` | Full_Warehouse | 叉车模型；可能改变障碍与碰撞几何 |
| `DynBench/Props/KLT_Bin/small_KLT_visual_collision.usd` | Warehouse、Warehouse_multiple_shelves | KLT 箱视觉/碰撞资产；默认仓库也受影响 |
| `NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr` | Hospital | 天空环境光贴图，影响渲染与视觉一致性 |
| `NVIDIA/Assets/Skies/Clear/sunflowers_4k.hdr` | Office | 天空环境光贴图，影响渲染与视觉一致性 |

没有在审查范围发现同名可用副本。前两项在迁移时已发现；后两项是本次增加 asset 属性检查后发现的，早先只查层引用未覆盖它们。

同名文件来源、版本和依赖需匹配场景。若改用替代道具/光照，应记录场景差异，不能将替代版本视作严格相同实验条件。

### R5. 人体 UV 与纹理（视觉保真需要）

未找到：

- `data/smpl/smpl_uv.npz`。
- 配套人体贴图，如代码搜索的 `smpl_body_texture.png`、`smpl_uv_20200910.png`。
- 默认搜索的 `data/smpl/SMPLitex-textures/` 纹理集合。

没有 UV/纹理时可使用纯色 mesh，因此它们不是所有物理测试的启动前提；但复现照片级人体外观和基于 RGB 的感知时需要补齐或明确记录外观差异。

## 五、未本地化或未验证，不能直接判定必然缺失

### Nova Carter 与运行时环境贴图

本地 `../Assets/Robots/NovaCarter/nova_carter.usd` 不存在，审查范围未找到副本。但 `CrowdSim/protomotions_runtime.py` 会退回 `ISAAC_NUCLEUS_DIR` 下的官方资源地址。

ProtoMotions 场景还从该资源根目录加载 `kloofendal_43d_clear_puresky_4k.hdr`。这些运行时构造的远程地址不在本地 USD 文件扫描中；未验证当前后端能否联网取得它们。

在线运行：需要验证实际可访问性。离线复现：需要连同 USD 的子资源一起本地化。FLUX/NavDP 的 Dingo 不能无条件替换 Nova Carter 的关节、相机挂点和动力学配置。

### MDL 与缩略图资源

- 6 条 bare-name `OmniPBR.mdl` 引用依赖运行时搜索路径，不能因为旁边没有同名文件就认定缺包。
- 19 条远程 `omniverse://isaac-dev.../AssetRenderer/default_scene.usd` 引用全部位于 `.thumbs`。
- 14 条本地缺失引用（12 个唯一目标）属于缩略图辅助层，未从六个测试入口递归可达；不应与真正运行场景缺口混在一起。

### Git LFS 占位文件

仓库中发现 49 个 USD/USDA 仍是 LFS 指针，清单见证据文件。包括 SMPL、其他机器人及旧辅助场景。

当前 SMPL 配置选择 `mjcf/smpl_humanoid.xml`，IsaacLab 场景代码调用 MJCF→USD 转换；因此 `smpl_humanoid.usda` 的占位状态不证明当前启动一定依赖它。实际选择到这些旧 USD 的配置才需要对应 LFS 二进制对象。不要为了复现默认 SMPL 路径盲目把所有机器人资产列为强制下载。

## 六、严格复现论文额外缺少的资源

以下结论指未在当前仓库中找到可确认对应论文的产物，不证明作者没有发布。

| 论文资源 | 当前缺口 |
|---|---|
| 2,848 段 AMASS 导航动作子集 | 准确 clip 清单、筛选和切分、预处理配置及对应数据；自带通用 YAML 尚未证明与论文一致 |
| 行人轨迹扩散训练数据 | 历史/未来轨迹、邻居、地图条件、采样频率、训练/验证划分；确切来源未从仓库确认，不能直接假定 ETH/UCY 足够 |
| 条件轨迹扩散权重及配置 | 没有可确认接入 CrowdSim 的论文模型权重；FLUX/NavDP 导航权重不等价 |
| 论文 AMP 人体控制器 | 配套权重、任务/风格奖励、训练配置与控制频率；已有 MaskedMimic/motion tracker 不自动等价 |
| 视觉行人感知资源 | 关键点检测、MonoLoco++、跟踪所需权重/标定/接口配置；若使用现成模型，不必先自行训练全部感知模型 |
| 论文时空注意力导航策略 | 对应权重与完整配置，或从头训练所需的模型实现、环境及训练协议 |
| 30 个论文场景与任务集 | OmniGibson 场景 ID、资产、地图、划分、起终点/种子/行人配置；现有 DynBench 六类测试场景和大量 cluttered 训练文件尚不能映射成论文集合 |
| 动作质量和消融参考数据 | 15 关键点/30 Hz 处理、AMASS 参考集、聚类与 Motion-MMD 参数；固定的 8 场景、3–15 行人、200 s 等评测任务定义 |
| 原始实验记录 | 可对照的配置、种子、指标、训练/推理计时与性能绘图输入 |

论文要求据本地 NavIsaacLab PDF 第 IV–VI 节。取得全部上述文件后，仍需完成前一份兼容性报告中的代码适配与算法实现；资源齐全不等于结果已复现。

## 七、训练产物与非必需项

- `output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt` 当前不存在；基础 PPO 从头训练会生成它，推理/采集阶段才需要已有权重。
- 偏好奖励数据、奖励模型及微调权重属于本仓库扩展；不是本地论文 v1 复现的首要前置资源。
- 实机实验另需机器人、传感器、标定、地图及运行日志；不阻塞纯仿真复现。
- 没有扫描用户所有磁盘或外部数据服务器；“缺失”以本次明确的仓库、Assets 和配置解析路径为范围。

## 八、建议补齐顺序

1. **先完成一个场景闭环**：SMPL neutral 模型、兼容 AMASS MotionLib、默认 Warehouse 的 KLT 箱、可用 Nova Carter；校准现有地图。需要带纹理人体时同步补 UV/贴图。
2. **扩展四个 DynBench 场景**：生成四套占据地图，补 Forklift 与两张 HDR，验证场景、地图和任务坐标一致。
3. **再追求严格论文复现**：取得论文动作子集、30 场景任务清单、扩散/AMP/感知/导航模型资源及评测定义。

## 九、审查证据

- [统计与方法限制](resource_audit/summary.json)
- [资源文件清单 CSV](resource_audit/resource_inventory.csv)
- [USD 未解析引用 CSV](resource_audit/usd_unresolved_references.csv)
- [六个测试入口的递归依赖结果](resource_audit/scene_dependencies.json)
- [CrowdSim 场景配置与地图状态](resource_audit/scene_config_status.json)
- [600 个 episode 基础验证](resource_audit/episode_summary.json)
- [49 个 LFS 占位文件](resource_audit/lfs_pointers.txt)
- [迁移位置记录](DATASET_LOCATIONS_ZH.md)
- [代码兼容性与复现阶段](NAVISAACLAB_COMPATIBILITY_AND_REPRODUCTION_ZH.md)

## 十、历史快照：Assets/motion 解压前目录解析

本节保留本次处理前证据。当前 `.motion` / `.pt` 状态以 [AMASS 数据准备记录](AMASS_PREPARATION_ZH.md) 为准。

本节更新此前原始动作资源尚未到位的判断。目录 `/project/EXP/SangforAndSaut/Assets/motion` 当前有 **27 个文件（不计下载缓存）、34,156,748,486 字节（约 31.81 GiB）**。

### 10.1 文件组成

- `raw/`：22 个 AMASS 子数据集 `.tar.bz2` 压缩包，约 21.33 GiB，尚未解压成子目录。
- `amass_joints_h36m_60.pkl`：4,229,858,537 字节，H36M 风格 17 关节预处理数据。
- `converted_train.zip`：4,280,624,352 字节。
- `converted_train.tar.bz2`：2,742,519,431 字节。
- `README.md`、`.gitattributes`。
- 目录下没有 `.motion` 或 `.pt`，`amass_smpl_validation.pt` 仍不存在。

### 10.2 原始包可用于当前转换路线

22 个包包括 ACCAD、BMLhandball、BMLmovi、BMLrub、CMU、DFaust、DanceDB、EKUT、EyesJapanDataset、GRAB、HDM05、HUMAN4D、HumanEva、KIT、MoSh、PosePrior、SFU、SOMA、SSM、TCDHands、TotalCapture、Transitions。

已逐包读取首个 NPZ 的目录/字段信息。常见运动样本含 `poses (T,156)`、`trans (T,3)`、`betas (16,)`、gender、帧率及 dmpls；156 维是原始姿态参数布局，不是 ProtoMotions 的 69 维控制动作。部分包还含 marker 数据、stageii 格式或仅含体型的 `shape.npz`。DanceDB 首个样本恰为 shape 文件；SOMA 首个 NPZ 超过抽样大小阈值，只读取了成员信息，未检查内部字段。

压缩包名与内部目录名存在差异，例如：

- `HDM05.tar.bz2` → `MPI_HDM05/`
- `MoSh.tar.bz2` → `MPI_mosh/`
- `PosePrior.tar.bz2` → `MPI_Limits/`
- `BMLrub.tar.bz2` → `BioMotionLab_NTroje/`
- `EyesJapanDataset.tar.bz2` → `Eyes_Japan_Dataset/`
- `TCDHands.tar.bz2` → `TCD_handMocap/`
- `DFaust.tar.bz2` → `DFaust_67/`
- `SSM.tar.bz2` → `SSM_synced/`
- `Transitions.tar.bz2` → `Transitions_mocap/`

应保留归档内部目录名，使其与 YAML 匹配；不要按压缩包简称额外套一层目录。

### 10.3 默认验证集：345/345 路径已匹配

完整遍历四个相关 tar 包，按转换脚本的后缀及文件名规范化规则匹配 `data/yaml_files/amass_smpl_validation.yaml`：

- HumanEva：28/28。
- MPI_HDM05：206/206。
- SFU：35/35。
- MPI_mosh：76/76。
- 合计 **345/345**，无缺失路径。这四个包全部成员的展开尺寸约 1.64 GiB，转换生成物另需空间。

这证明默认验证清单要求的源文件名都在归档内；未逐动作验证帧数、YAML 截取区间、数值有效性或执行转换，不代表现成 MotionLib 已通过验收。

仓库通用 train 清单有 11,303 个动作文件，test 清单有 138 个；其所需子数据集目录均在已识别原始包中，但本轮没有完整遍历它们并逐文件匹配。论文 2,848 条动作清单仍未确认。

### 10.4 两类预处理数据不能直接作为人体控制 MotionLib

`amass_joints_h36m_60.pkl` 经完整静态 Pickle 指令扫描（跳过大数组载荷，不执行反序列化）：

- 顶层为 list，识别出 **14,919 个 `(17,T,3)` 数组形状**。
- 总计 10,365,045 帧，序列长度 1–2,916 帧。
- 数组字节载荷与这些形状按 float64 计算的大小吻合。
- 文件名暗示 60 Hz，但本轮未发现独立帧率字段，也未验证单位、坐标系和关节名称映射。

`converted_train.zip` 完整目录包含 **96,179 个 `.pkl` 文件**，全部位于 `train/`，每个文件 49,772 字节，总展开尺寸 4,787,021,188 字节。两份样本静态解析均为：

```text
{
  data_input: None,
  data_label: float32[243, 17, 3]
}
```

这些是按窗口组织的 17 关节姿态样本；不能将 96,179 个窗口视为 96,179 段独立原始动作。它们缺少当前 MotionLib 所需的 SMPL 全身旋转、69 自由度状态及相应刚体速度等结构，不能改扩展名为 `.pt` 使用。仅凭 17 个位置点也不能唯一恢复原始完整关节旋转。

`converted_train.tar.bz2` 的首个实际文件与 ZIP 中同名文件内容完全一致，说明至少该样本重复；未遍历整个 tar，不能宣布两个归档完全相同或建议删除其中之一。

README 所说的 `tools/convert_amass.py` 未在本轮搜索的两个本地项目中找到，且 README 写的 `train.tar.bz2` 与实际文件名不同；应以实测结构和当前 ProtoMotions 转换入口为准。

### 10.5 下一步与重要更正

默认 CrowdSim 所需动作数据现在主要缺 **解压 → 转换 → 打包**，不是缺上述四个数据集的源归档。

建议先仅解压 HumanEva、HDM05、SFU、MoSh 四个包到 `Assets/motion/raw/`，保留内部目录。准备转换依赖后，从 ProtoMotions 根目录运行：

```bash
python data/scripts/convert_amass_to_motionlib.py \
  ../Assets/motion/raw ../Assets/motion \
  --humanoid-type smpl \
  --motion-config data/yaml_files/amass_smpl_validation.yaml
```

预期产物为 `../Assets/motion/amass_smpl_validation.pt`。当前转换脚本使用已有 `smpl_humanoid.xml` 和姿态参数做前向运动学；**不直接加载 SMPL_NEUTRAL/MALE/FEMALE.pkl，也不按 betas 重建个体骨架**。之前将这些 PKL 泛化为当前动作转换前提的说明，应以此处代码核对结果修正；可视人体 mesh 仍需要 neutral 参数。

脚本默认目标输出帧率为 30，但其抽样逻辑可能使用不低于目标的可整除帧率；后续应检查实际输出元数据。它还会遍历输入根目录的其他动作，所以首次转换宜只展开所需四包。

本轮没有解压到数据目录、没有转换或改动原始数据、没有加载未知 Pickle 对象。ZIP 未执行所有 payload 的 CRC 检查，其余 18 个原始归档仅做抽样。证据见 [motion 目录解析 JSON](resource_audit/motion_directory_analysis.json)。


## 十一、AMASS 解压与转换完成

三个项目清单共 11,786 段全部转换并通过当前 MotionLib 加载验证。原始 22 个归档保留，19 个项目所需包已解压。

| 数据集 | 动作数 | 帧数 | 实际时长（秒） | 大小 | 帧率分布 |
|---|---:|---:|---:|---:|---|
| validation | 345 | 307,046 | 9,892.113 | 646.60 MiB | 30 Hz × 307、50 Hz × 38 |
| test | 138 | 29,705 | 985.567 | 62.57 MiB | 30 Hz × 138 |
| train | 11,303 | 4,331,631 | 122,620.894 | 8.91 GiB | 30 Hz × 7385、50 Hz × 3918 |

结果位置为 `../Assets/motion/amass_smpl_{validation,test,train}.pt`。详细方法、零接触片段及严格论文复现边界见 [AMASS 数据准备记录](AMASS_PREPARATION_ZH.md)。本节更新前文历史快照中“尚未解压、尚无 MotionLib”的描述。
