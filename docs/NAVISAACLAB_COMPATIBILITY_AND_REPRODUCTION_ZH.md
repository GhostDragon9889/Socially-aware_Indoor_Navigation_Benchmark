# NavIsaacLab：代码兼容性复核与复现步骤

> **最新迁移状态（2026-10-01）：** SFM+SMPL MaskedMimic 已完成四人8秒共享 PhysX、真实蒙皮和 MP4 验收，SMPL neutral 已接入，相关接口与推理问题已修复。
> 当前运行命令、依赖锁、最终1,750项回归及论文资源缺口见 [Isaac Sim6 / IsaacLab3 迁移说明](ISAAC_SIM6_LAB3_MIGRATION_ZH.md)。本文以下内容为历史审查快照。

> 2026-10-01：最新代码复核见 [全面审查报告](CODE_REVIEW_20261001_ZH.md)，新增人体子集 reset、四元数和旧偏好奖励实证，并区分普通 PPO 与带 KL 更新的数值保护。以下保留此前审查记录。

日期：2026-09-30。审查当前工作区，基准提交 `7a8417a`，包含未跟踪的 CrowdSim、src、配置等代码。**本轮排除 NavDP 和 FLUX**。本报告补充此前报告，尤其新增跨仓库接口和指定 IsaacLab 工作区依赖冲突；旧报告的测试数量不是本轮结果。

## 最新状态更新：AMASS 目录与双仓库比较之后

2026-09-30 增量复核：

- AMASS 默认验证库已经补齐：`../Assets/motion/amass_smpl_validation.pt` 含 345 段，测试库含 138 段，训练库含 11,303 段，均通过当前 MotionLib 加载与逐动作插值验证。原始 22 个归档全部保留，完整数据处理详情见 [AMASS 数据准备记录](AMASS_PREPARATION_ZH.md)。
- 保留现有 `Assets/motion/raw/` 即可，无需迁往此前建议的另一 raw 路径。当前转换脚本使用已有 MJCF 做前向运动学，不直接需要 SMPL 参数 PKL；可视人体 mesh 仍需要 neutral 参数。
- SMPL neutral/UV、四套地图和四个场景依赖仍缺失。MaskedMimic 权重及推理配置存在。论文的 2,848 条动作子集不能由通用验证清单替代。
- 本轮检查的 openAIbase 解释器仍缺 IsaacLab、IsaacSim、Lightning、TensorDict、dm-control、OmegaConf、smplx；Torch 2.12.0 超出目标 extra 上限。不推断机器其他环境的安装情况。
- 加载兼容函数缺失、评估观测/reset 问题、PPO 单样本尾批方差风险仍在；采集入口三个 F821 再次确认。没有重跑历史 CPU 测试或 GPU 仿真。

### 新增重点：CrowdSim 的后端状态接口仍需迁移

当前核心 `simulator/isaaclab/config.py:64` 声明后端 xyzw 四元数，simulator 多处通过 `wp.to_torch()` 读取底层状态。`CrowdSim/tools/smpl_mesh_visualizer.py:163–170` 仍直接把 `_robot.data` 交给 Torch 运算，并将 `root_quat_w` 作为 wxyz 传入；`:536/574` 按 wxyz 旋转顶点。`CrowdSim/nav_manager.py:2345` 也按 wxyz 计算朝向。

源码可确认接口约定不一致，但未运行目标 GPU 后端确认实际首先出现哪种错误。需统一张量类型、关节顺序、四元数格式，并核验网格/骨架对齐和 RGB-D 时序。`nav_manager.py:471` 会忽略网格更新异常，所以主循环运行不证明可视人体正常。

另需验证 env_ids 子集 reset 与导航目标覆盖逻辑，以及 MaskedMimic 的配置类路径和模型 rollout 状态。A 版本中的 `apply_backward_compatibility_fixes` 没有实际迁移逻辑，复制回该函数不能解决上述接口差异。

资源证据见[资源报告第十节](NAVISAACLAB_RESOURCE_AUDIT_ZH.md)，接口变化见[双目录比较](PROTOMOTIONS_DIRECTORY_COMPARISON_ZH.md)。以下详细报告中的历史测试保留为既有证据。

## 0. 最新复核：当前距离论文复现还缺什么

复核日期：2026-09-30；基于当前工作区和本地 NavIsaacLab PDF。此次重读论文第 III–VI 节、核对主入口和关键代码、重新检查 26 个指定资源路径，并重新运行采集入口 F821 检查。未重复之前的全量 USD 扫描和 CPU 测试，未启动 GPU 仿真。此前测试结果保留在第 3 节，不能视作本轮再次执行。PDF 内容依据文本提取；图片查看受到本机沙箱错误影响，本轮未完成插图视觉复核。

**结论：目前已有可复用的仿真与 PPO 基础，但完整仿真闭环尚未验收，论文的关键算法和实验协议也未实现到位。补齐 AMASS 后仍不能直接复现论文。**

### 0.1 已有内容：无需重复准备

- 六个 DynBench 测试场景入口 USD 均存在，数据位于 `../Assets/DynBench/`；默认仓库通过 `../Assets/Simple_Warehouse` 链接访问。
- MaskedMimic SMPL 权重、配套推理配置、SMPL motion tracker 权重均存在。文件存在尚不证明能在目标后端成功加载。
- 默认 `CrowdSim/maps/newmap.yaml` 与 `World1.png` 存在，场景对齐仍待验证。
- 有 A*、SFM、人体与机器人连接、相机、PPO 训练和部分测试；ProtoMotions 有通用 AMP 实现（`protomotions/agents/amp/agent.py`）。
- 先前资源审查确认的 600 个 episode JSON 是 DynBench 测试任务，不能据此认定论文 30 场景任务集已经齐全。

### 0.2 第一优先级：运行当前实现缺少的资源

1. **SMPL 动作库（默认验证集已补齐）**：`../Assets/motion/amass_smpl_validation.pt` 已生成并验证，配置路径不需要再调整。测试库和训练库也已生成；当前通用清单产物不能替代论文 2,848 条精确子集，详见 [数据准备记录](AMASS_PREPARATION_ZH.md)。
2. **SMPL 参数**：`data/smpl/SMPL_NEUTRAL.pkl` 及其支持的嵌套候选路径均不存在；目录只有 `smpl_joint_names.py`。当前 MJCF 转换脚本不直接使用这些 PKL；人体 mesh 仍需要 neutral 参数。UV 与纹理也未齐备，影响人体外观；它们与物理骨架资产是不同资源。
3. **四套地图**：Full_Warehouse、Hospital、Office、Warehouse 的 `map/occupancy.yaml` 及对应地图图片。应由各自场景生成并校准；默认仓库的已有地图不能直接替代这四套。
4. **四个可达的场景依赖**：`DynBench/Props/Forklift/forklift.usd`、`DynBench/Props/KLT_Bin/small_KLT_visual_collision.usd`、`NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr`、`NVIDIA/Assets/Skies/Clear/sunflowers_4k.hdr`，均相对于 `../Assets/`。模型缺口可能影响碰撞，HDR 缺口主要影响渲染。
5. **待验证的远程资产**：本地 Nova Carter 缺失，但代码提供官方资源 fallback；需要验证实际解析和访问。不能直接判定机器人完全不可用。离线运行需本地化完整引用链。

逐文件状态见 [本轮路径复查](resource_audit/reproduction_recheck_20260930.json)，完整依赖证据与适用范围见 [资源审查](NAVISAACLAB_RESOURCE_AUDIT_ZH.md)。

### 0.3 第一优先级：环境与代码仍有阻塞

- **检查点适配缺函数**：`CrowdSim/protomotions_runtime.py:122/132` 仍导入并调用不存在的 `apply_backward_compatibility_fixes`；当前模块提供 `apply_all_inference_overrides`。需迁移实际配置流程，不能仅改名。
- **安装基线未统一**：当前主包要求 IsaacLab 12.0.0 / Sim 6.0.0.1 / Torch >=2.11,<2.12，相邻 IsaacLab 源码仍声明 6.1.11。目标提交工作区要求 moviepy <2，主包要求 >=2；本轮再次核对上游固定提交，冲突仍存在。这个版本组合是仓库的目标，不是论文原始环境的证明。
- **评估控制流仍有错误**：`eval_policy.py:360–404` 丢弃新人体观测、未消费人体 done；先传送再 reset 的顺序及未禁用课程会改变任务。修复前不能用其结果对照论文。
- **小批量 PPO 有 NaN 风险**：`ppo_policy.py:857/978` 仍使用 `returns.var()`，单样本尾批问题尚未修复；此前 CPU 复现见 C06。
- **采集入口仍有 3 个 NameError**：本轮 Ruff 再次确认 `collect_ppo_buffer.py:861–863` 的三个未定义变量；基础训练与采集受影响范围不同。

各问题修复方式和验证边界见 C01–C09。尚缺专用环境的依赖锁定、`pip check`、真实检查点加载与一步仿真验收。

### 0.4 第二优先级：论文方法尚未完成的部分

**A. 条件轨迹扩散模型（论文 IV-A，第 6 页）**

- 论文使用历史轨迹、邻居轨迹、占据地图作为条件，通过 U-Net 生成未来 5 秒轨迹，再转为人体控制目标。
- 当前 `CrowdSim/nav_manager.py:21/751` 使用 Social_Force 和 SFM 目标；未找到已接通主流程的对应扩散模型。
- 尚需：模型实现、训练/推理配置、轨迹数据与切分、权重或训练入口、重规划与控制时间同步。论文引用 TRACE/PACE 不等于仓库已经包含它们；轨迹数据来源尚不能确认为 ETH/UCY。

**B. 论文专用 AMP 人体控制器（IV-B/C，第 6–7 页）**

- 当前 `CrowdSim/config/env.yaml:9` 指向 MaskedMimic。仓库通用 AMP 可作为基础，但尚未形成论文的轨迹条件、障碍观测、任务奖励和训练配置。
- 尚需：论文筛选的 **2,848 段 AMASS 导航动作准确清单**、预处理与划分；论文 AMP 控制器权重或完整训练配方；轨迹、能耗、方向奖励及 `r_amp + 0.5 r_task` 的一致实现。
- 需统一验证控制频率、关节映射、状态和动作维度，完成闭环跟踪误差测试。已有 MaskedMimic 或 motion tracker 权重不能作为论文 AMP 权重的证据。

**C. 视觉行人观测（V-A，第 8 页）**

- 论文要求姿态检测 → MonoLoco++ 身体朝向 → 深度/跟踪估计位置与速度，并维护 **8 步历史**。
- 当前 `nav_manager.py:715/1477` 从 simulator root state 读取位置与速度；`:2186` 构造当帧 Top-K 邻居的距离、方位 sin/cos、相对速度。
- **当前方位 sin/cos 是邻居位置相对机器人方向，不是行人身体朝向。** 尚缺视觉估计、跨帧 ID、身体朝向、历史窗口、遮挡与丢失 mask、坐标对齐及相关权重/标定。

**D. 三层时空导航网络与训练配方（V-A/B/C，第 8–9 页）**

- 论文采用三层“时间因果注意力 → 行人交互注意力”，再接地图 CNN、目标查询 cross-attention 与门控。
- 当前 `ppo_policy.py:223–267` 的可选 FusionAttention 是一层模态 token 注意力，不能对应论文的时空结构；默认配置未开启该项。
- 尚需与上述历史观测匹配的网络、mask 和 rollout 存储。论文目标/碰撞奖励是 +20/-20，当前配置是 +50/-50，其他奖励项也有差异；并行实验的 nsteps/nepochs 为 32/6，当前配置为 256/4。
- 从头训练不要求事先存在导航权重；但用于复现推理时，需要对应模型权重及完整解析配置。

### 0.5 第三优先级：论文数据与实验仍未闭合

- **30 个论文场景**：论文在引言声明 30 个场景，并在 III-A 描述 OmniGibson 来源；尚缺可核对的场景 ID、资产版本、地图、训练/测试划分、起终点、人群配置和种子。现有 DynBench 不能自动算作该集合。
- **人体跟踪评测**：准确参考 clip 集、状态同步和误差脚本；对照平均位置误差 0.218 m、旋转误差 0.343 rad、论文所报最大位置误差 0.433 m、0.5 m 失败阈值下的 64.6% 成功率。最大误差的聚合范围等细节仍需作者配置确认。
- **表 II**：固定 8 场景、3–15 行人、每场景 200 s，在同一个 AMP 控制器上比较 Diffusion 与 SFM；计算 success/fall/map collision/agent collision/path error。
- **表 III**：IsaacSim Animated People 基线、AMASS 参考集、15 关键点映射、30 Hz 重采样、中心/朝向/尺度归一化、K=50/100 聚类与 Motion-MMD。尚缺统一实现和参数；精确带宽、采样和距离定义不能仅靠表格推断。
- **图 6**：1/8/16/32/64 并行规模；5000 policy updates 计时、16000 simulator steps 的末 10 episodes 平均奖励；区分 human-only / RL inference / RL training 的 env steps/s 与 agent steps/s。尚缺统一计数口径、计时脚本与原始日志；论文 batch size、rollout 与 update 的具体关系需要明确。
- **物理与视觉验收**：尚未完成实际 IsaacLab 启动、RGB-D 有效性、人车接触、地图坐标、多环境隔离、异步 reset、保存恢复和性能实测。单元测试通过不能替代这些验收。
- **实机部分（若复现第 VI 节）**：Scout Mini、双 D435i、Mid-360、FAST-LIO、标定、部署接口和实验日志；纯仿真阶段可后置。

论文没有提供足够细节的项目应记为“待取得作者配置/需约定复现口径”，不能把自行选择的参数描述成原论文设置。当前仓库中的偏好奖励扩展不是完成本论文方法的首要依赖。

### 0.6 建议执行顺序与完成标准

1. **环境和阻塞修复**：统一后端版本、解决依赖约束、修复配置加载；验收为依赖一致且真实检查点能够加载。
2. **单场景资源与闭环**：补 MotionLib、SMPL、默认仓库 KLT 资产，验证 Carter 和已有地图；验收为 1 人体 + 1 机器人连续物理/视觉运行并正确 reset。
3. **可信训练与评估**：修复评估、尾批 NaN 和采集入口；短训练 loss/参数有限，保存恢复一致，固定任务分布可重复评估。
4. **论文算法与数据**：取得动作子集、场景清单和轨迹训练资源，完成 Diffusion+AMP、视觉历史观测、三层时空策略与对应训练配方。
5. **论文实验**：依次产出人体跟踪、表 II/III、并行性能图及其原始配置/日志。数据与场景无法取得时，明确记录近似复现差异。

## 1. 结论

目前可以在已有 CPU 环境运行轻量 PointGoal PPO，并完成保存、恢复和评估。当前工作区尚不具备直接运行完整 CrowdSim 仿真和严格复现论文的条件：

1. CrowdSim 调用一个当前 ProtoMotions 不再提供的函数。
2. 指定 IsaacLab 工作区与本项目有 moviepy 直接版本冲突；本机相邻 IsaacLab 源码版本也不是指定基线。
3. 场景、动作数据与视觉资源未形成完整输入集合。
4. 当前评估入口有观测和 reset 问题，输出不能直接用作论文对比结果。
5. 论文的 Diffusion+AMP、视觉行人感知、时空交互策略和完整评测协议尚未在当前主流水线闭合。

本轮只审查、运行 CPU 验证和新增本报告，没有修改实现、安装依赖、启动 GPU 仿真或训练论文模型。

## 2. 兼容性问题与证据

### C01 / P1：运行时适配导入了不存在的函数

位置：`CrowdSim/protomotions_runtime.py:122`、`:132`。

`configure_from_checkpoint()` 导入并调用 `protomotions.utils.inference_utils.apply_backward_compatibility_fixes`。当前目标模块只提供 `apply_all_inference_overrides`，没有该名称或重导出。全量检查 CrowdSim 对 protomotions 的显式 from-import，这是一处确认缺失的导入目标。

影响：前面的依赖、资源和 torch.load 成功后，此处仍会阻断运行；更早的错误可能暂时掩盖它。

修复：依据当前官方 inference_agent 的配置恢复流程编写适配层，明确旧检查点需要迁移的字段；不能把新函数简单改名或添加空函数蒙混通过。验收需要真实配置恢复、构造组件、加载模型并运行一步。

### C02 / P1：文档指定的 IsaacLab 工作区与根包不能共同满足 moviepy 约束

本项目 `pyproject.toml`：`moviepy>=2,<3`（Python>=3.9）。

指定 IsaacLab 提交 `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8` 的根项目 `isaaclab-dev`：`moviepy>=1.0.3,<2.0.0.dev0`。其注释还解释了与预置 Pillow 版本及视频导出行为的关系。

这是当前指定安装路径中的工作区依赖冲突，不依赖其他导航子项目。单一满足两方约束的解析结果不存在。分步 pip 安装可能只报告已安装环境冲突并继续，不能以命令退出成功代替环境一致性。

证据：[指定提交的根 pyproject](https://raw.githubusercontent.com/isaac-sim/IsaacLab/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/pyproject.toml)。

修复：先确定视频导出实际使用哪套 API，选择兼容实现并修正相关声明，或者将相关工具及其依赖放入有明确安装契约的独立环境。未经 API 验证，不直接强制覆盖版本，也不以 `--no-deps` 作为完成修复的依据。

### C03 / P1：本地源码、测试环境和目标后端不是同一版本组合

| 对象 | 本次读取结果 | 判断 |
|---|---|---|
| 项目期望 | Python 3.12；IsaacLab 12.0.0；Sim 6.0.0.1；Torch >=2.11,<2.12 | 当前仓库声明，不等于论文原始环境认证 |
| 相邻 `../IsaacLab` | Git `28a37cecd`；extension.toml 为 6.1.11 | 与期望提交/包版本不同 |
| 默认 Python | 3.12.13；缺 Torch、Lightning、TensorDict、IsaacLab、IsaacSim | 不适合运行完整路径 |
| openAIbase | Python 3.12.13；Torch 2.12.0；NumPy 2.4.6；缺 Lightning、TensorDict、IsaacLab、IsaacSim、dm-control、OmegaConf、smplx | 可作部分 CPU 测试；Torch 超出根 IsaacLab extra 范围 |

使用实际 `_validate_isaaclab_version()`，仅将版本查询替换为本地源码读到的 6.1.11，确认它会拒绝该版本。这不是证明该源码已安装到当前解释器。相邻目录也未发现 `.venv/bin/python` 或 `_isaac_sim/python.sh`；不能据此推断整台机器没有其他可用安装。

上游指定提交的 `source/isaaclab/pyproject.toml` 确实声明 12.0.0，避免把发行版本、扩展版本、开发工作区版本混为一谈。[上游包元数据](https://raw.githubusercontent.com/isaac-sim/IsaacLab/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/source/isaaclab/pyproject.toml)。

修复：为复现建立专用环境与独立源码 checkout，记录解释器、commit、包版本和 CUDA wheel 来源；保留现有其他项目环境。

### C04 / P1：评估丢弃新的人体观测，且没有消费人体 done

位置：`CrowdSim/tools/eval_policy.py:360`、`:371`、`:403`；`protomotions/envs/base_env/env.py:332`、`:687`。

评估循环只在循环前 reset 得到 obs；每轮调用 `env.step()` 后丢弃新 obs，下一轮继续使用旧字典。核心 get_obs 对动态观测进行 clone，旧字典不会自动变成新状态。人体 done 被赋给局部 done_indices 后也没有实际调用对应 reset。

影响：评估中的人体闭环偏离正常仿真流程，使导航统计失去一致性。修复应接收最新观测并正确处理 humanoid reset，然后验证连续多步状态变化及完成/摔倒后的恢复。

### C05 / P1：评估任务分布与配置宣称不一致

位置：`CrowdSim/tools/eval_policy.py:229` 起；`CrowdSim/ppo/goal_curriculum.py`。

评估没有关闭训练课程。本轮用实际 YAML 和课程类复现：初始化前距离 7–10 m，初始化后变为 2–4 m。

此外，评估先手工传送机器人，再 `env.reset(None)`，首次完整 reset 可覆盖起点；手工设置未复用完整 reset/控制缓存/传感器同步流程。

修复：显式禁用评估课程，固定并保存最终解析配置；提供统一的指定起终点 reset 接口；断言首帧真实位置和目标与任务清单一致。当前评估命令即使成功，也不能直接证明论文结果。

### C06 / P2：单样本尾批会污染 PPO 参数

位置：`CrowdSim/ppo/ppo_policy.py:857`、`:978`，轻量 PPO 中也有对应实现。

`returns.var()` 使用样本方差，只有一个元素时产生 NaN。batch 迭代器保留尾批。

本轮 CPU 复现：3 个 rollout 样本、minibatch_size=2、1 个更新 epoch，得到 `value_loss=nan`，11 个参数张量出现非有限值。默认大批量配置不一定触发，但缩小环境做冒烟测试时要特别处理。

修复：使用合适的总体方差或 rollout 级尺度，检查非有限 loss/梯度；覆盖普通 update 和带 KL 的更新。仅选择整除的 batch 可以临时规避，不能视为修复。

### C07 / P1（采集路径）：采集 main 使用未定义变量

位置：`CrowdSim/flow/collect_ppo_buffer.py:861–863`。

repeat_robot_routes、target_episodes、route_repeats 没有在 main 的作用域中赋值。Ruff 在 Python 3.12 目标下再次确认。采集入口会触发 NameError。

修复：从 CLI 与 YAML 合并后的 collection 配置读取值，并验证 main 到 collect 的参数传递。它阻断采集，不直接阻断基础 PPO 训练。

### C08 / P2：初始化和推理加载行为不一致

- `eval_policy.py:28` 顶层先导入 torch，之后才通过 crowd_sim 导入 IsaacLab；与项目 `import_simulator_before_torch()` 的约定不一致。尚未运行后端来证明一定崩溃，应统一入口顺序并做子进程启动测试。
- `protomotions_runtime.py:238` 只传 `load_env=False`，保留 `load_training_state=True`；当前官方 inference_agent 同时传 `load_training_state=False`。纯推理恢复会额外依赖优化器/训练状态，需按 checkpoint 实际内容核验。
- CrowdSim 独立运行入口调用 agent.eval()，PPO train_loop 对人体 agent 未见等价切换。应明确预训练人体策略的推理模式，检查 dropout/随机性与评估路径一致。

以上是确认的接口使用差异；未把未经运行验证的影响写成必然失败。

### C09 / P2：打包、文档与资源预检不完整

- 根 setuptools 只安装 protomotions；不包含 `src/crowdsim_ppo`，不存在 `[test]` extra。轻量路径目前应使用 `PYTHONPATH=src:.`，开发 extra 是 `[dev]`。
- FULL_PIPELINE 引用的 check_isaaclab3、validate_isaaclab3、constraints_isaaclab3.txt 和迁移 Markdown 不存在。
- `_validate()` 当前只做 exists 检查；未检查推理配置是否存在、Git LFS 指针内容或所有 USD 外部引用，和文档宣称的检查能力不符。
- `--empty` 分支之前仍加载地图并检查 humanoid checkpoint / motion_file，不能作为完全无人体资源的场景验收入口。
- smplx/OpenCV 未形成 CrowdSim 独立直接依赖声明。mesh 初始化失败被捕获后继续，视觉输入可能已经变化。

低影响静态问题：train_ppo.py:312 的 CrowdNavigationManager 类型注解未导入；启用 postponed annotations 后通常不阻断函数定义，但类型解析仍需修正。不要与采集入口的三个运行时 NameError 等同。

## 3. 已验证与未验证

| 项目 | 本轮结果 | 范围 |
|---|---|---|
| AST 解析 | 515 个 Python 文件全部通过 | protomotions/CrowdSim/src/tests/examples/deployment/data/usd_convert；Python 3.12 |
| CrowdSim + 轻量测试 | 38 passed，2 warnings | warnings 来自不完整测试对象析构时缺属性 |
| 核心状态与 IsaacLab 摩擦测试 | 9 passed | 不需要真实启动 IsaacLab；另有 TorchScript 等 warnings |
| 其余选定核心测试 | 3 个收集错误 | 缺 OmegaConf / dm-control，未执行断言；不是 3 个业务断言失败 |
| 轻量 PPO 闭环 | CPU 训练到 64 步，恢复到 96 步，评估 4 episodes 成功退出 | 仅入口/保存加载可用；短训练成功率为 0，不证明策略质量 |
| 检查点配置静态检查 | 69 个 ProtoMotions GLOBAL 引用均可对应本地模块/名称 | 使用 zipfile + pickletools；未执行 pickle，未验证字段和 tensor shape |
| 单样本尾批 | 复现 NaN 及参数污染 | 真实 CPU 实现 |
| 课程距离 | 复现 7–10 m → 2–4 m | 真实配置与课程类 |
| IsaacLab GPU / 相机 / 接触 / 多环境 reset | 未验证 | 环境和资源未闭合 |
| 论文训练、指标、性能和实机 | 未完成 | 不能由单元测试替代 |

测试解释器：`/home/lfm/anaconda3/envs/openAIbase/bin/python`。轻量闭环临时产物位于 `/tmp/navisaaclab-compat-42f1w8wv`，该路径不是交付依赖。

## 4. 当前资源与算法完成状态

| 项目 | 状态 | 尚缺内容 |
|---|---|---|
| 核心模拟器接口、组件、通用 AMP/PPO | 有实现 | 完整目标环境验收 |
| CrowdSim 场景、地图、SFM、控制、相机连接代码 | 有实现 | 上述适配修复与物理/视觉实测 |
| MaskedMimic 权重和 resolved_configs_inference.pt | 文件存在 | 完整反序列化、字段、模型形状和后端兼容性验证 |
| 默认 AMASS 动作文件 | 345 段验证库已在配置路径生成并通过 MotionLib 校验 | 后续验证人体控制器和模拟器闭环 |
| 6 份 CrowdSim 场景 YAML | 迁移后引用的场景 USD 均存在 | 4 套地图、4 个可达依赖文件与几何对齐验证；详见资源审查 |
| data/smpl | 目录存在 | 验证实际模型、纹理、骨架映射与渲染 |
| Nova Carter | 支持本地路径及 Nucleus fallback | 验证实际可访问性、轮关节/相机挂点和动力学参数 |
| 论文轨迹扩散模型 | 未在 CrowdSim 路径发现完整实现 | 历史/邻居/地图条件、5 秒轨迹输出、数据/权重与控制连接 |
| 论文人体 AMP 控制器 | 通用 AMP 基础存在；默认运行 MaskedMimic | 对应训练配置、2,848 段动作清单、权重、任务奖励和跟踪评测 |
| 论文视觉行人感知 | 未见完整连接 | 关键点检测、MonoLoco++、跟踪、朝向/速度与标定 |
| 论文三层时空策略 | 当前为向量/CNN，另有单层模态 attention | 三层因子化时空注意力、地图交互、目标交叉注意力与门控 |
| 论文 30 场景 benchmark | 未形成完整集合 | 场景 ID、划分、任务种子、人群配置、统一评测 |
| 论文动作与性能指标 | 当前只有部分导航统计 | 摔倒、路径误差、动作覆盖/熵/MMD，吞吐和消融实验 |

“未发现完整实现”指当前审查范围与入口，不能推广为上游作者未发布。偏好奖励微调是本仓库扩展，不是本地论文 v1 的必要前置步骤。轻量二维 PointGoal 可以用于算法调试，不能替代 NavIsaacLab 的人体物理和视觉复现。

## 5. 按依赖顺序执行的复现步骤

### 阶段 A：冻结目标并修复兼容边界

1. 保存当前工作区版本，纳入 CrowdSim/src/configs 等必要未跟踪文件；记录提交和资源清单。
2. 区分“运行当前 CrowdSim”与“复现论文结果”：前者使用本仓库支持的现代后端，后者还需作者原始配置/数据/权重和实验协议；论文原始版本组合不能由当前根包声明代替。
3. 优先处理 C01、C02、C03；在正式评估之前处理 C04/C05；在调小训练 batch 之前处理 C06；采集前处理 C07。
4. 在独立 checkout 准备指定 IsaacLab commit，先核对其 Python/Sim/Torch 和 API。不要把现有相邻 6.1.11 目录直接当作该环境。
5. 修正 moviepy 及其关联 API 后，将主包与环境 requirements 一起求解，增加 CrowdSim 的直接依赖和目标平台锁文件。只在统一求解与 `pip check` 均通过后继续。

安装命令的形式应是：在专用解释器内安装主包 `[isaaclab,dev]`、同环境清单和 CrowdSim 直接依赖。**当前 C02 未修复，所以本报告不把现有安装命令写成保证成功的一键流程。** `--no-deps` 不能作为消除冲突的验收。

### 阶段 B：准备资源并做 CPU 验收

6. 配齐人体 checkpoint、配套 inference config、动作文件、SMPL、场景 USD、贴图、地图 YAML 与图片；检查所有 USD 引用和必要 LFS 对象。
7. 使用显式路径的专用 Python 检查包元数据、导入和资源；避免 shell 默认 python 指向其他环境。
8. 运行已有轻量测试及核心适配测试。下面的 `$NAV_REPRO_PY` 是阶段 A 创建的解释器绝对路径，由使用者设置：

```bash
"$NAV_REPRO_PY" -m pip check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. \
  "$NAV_REPRO_PY" -m pytest --import-mode=importlib -q tests CrowdSim/tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  "$NAV_REPRO_PY" -m pytest -q \
  protomotions/tests/test_simulator_state_and_config.py \
  protomotions/tests/test_config_builder_and_inference_utils.py \
  protomotions/tests/test_isaaclab_robot_friction.py \
  protomotions/tests/test_isaaclab_mjcf_usd.py
```

验收：无收集错误、核心断言通过；另加 C01 配置恢复、C04/C05 评估状态、C06 尾批更新的针对性回归，现有 38 项通过不包含这些保证。

### 阶段 C：最小仿真闭环

9. 复制默认配置为新配置（例如 `CrowdSim/config/repro_smoke.yaml`；本轮未创建），设 1 机器人、1 人体，修正资源路径；关闭自动录制/W&B，保留实际需要的深度相机。
10. 分层验证：场景/地图对齐 → 人体控制与 mesh → 机器人驱动 → RGB-D → 相互碰撞 → 到达/摔倒/reset。当前 --empty 有 C09 限制，修复后再用作最小场景入口。
11. 接通人体检查点：恢复配置，按真实 schema 修复 C01，按纯推理目的选择加载状态；检查动作/观测形状、设备、四元数约定和时间步。训练中的人体策略保持约定的推理模式。
12. 进行短 PPO 训练。以下命令仅在前述阻塞项修复、配置文件已创建后使用：

```bash
"$NAV_REPRO_PY" -m CrowdSim.ppo.train_ppo \
  --env-config CrowdSim/config/repro_smoke.yaml \
  --total-steps 4096 --headless --no-wandb
```

`total-steps` 可能按 rollout 批次越过阈值；同步选择 rollout/minibatch，避免单样本尾批。验收包括：loss/参数有限、深度不是恒零、人体每步使用新观测、动作实际执行、reset 后相机与状态一致、保存恢复一致。不能只以训练日志在增长判定成功。

### 阶段 D：修复评估并扩展并行规模

13. 创建独立评估配置，禁用课程并固定任务清单；修复 C04/C05 后再运行：

```bash
"$NAV_REPRO_PY" -m CrowdSim.tools.eval_policy \
  --ckpt /实际训练目录/robot_ppo_latest.pt \
  --env-config CrowdSim/config/repro_eval.yaml \
  --num-episodes 200 --seed 42 --headless
```

`repro_eval.yaml` 是待创建的建议配置，不是当前仓库文件。

14. 从 1 扩展到 8、16、32、64 个并行环境；分别验证状态索引、活跃人体/机器人数量、碰撞、异步 reset、相机和资源占用。不要仅修改 --num-envs 而忽略 YAML 中 agent 数量。
15. 如需采集专家数据，先修复 C07，再使用 `CrowdSim.flow.collect_ppo_buffer`；检查数据中的观测、动作、终止、时间戳和真实执行轨迹一致。

### 阶段 E：补论文方法与实验

16. 取得论文 2,848 段 AMASS 动作清单、30 场景和任务划分、训练配置、种子及模型权重；若无法取得，记录自建数据/场景差异，结果标为近似复现。
17. 接入条件轨迹扩散模型与对应 AMP 控制器；保留同一个底层控制器完成 Diffusion+AMP / SFM+AMP 消融。
18. 完成视觉行人状态估计、历史序列和三层时空策略；对齐论文奖励及训练设置。当前课程、奖励和 256/4 的 rollout/epoch 设置不能直接替代论文并行实验的 32/6。
19. 复现人体可靠性实验：8 场景、3–15 行人、每场景 200 s，统一成功/摔倒/地图碰撞/行人碰撞/路径误差定义。
20. 复现动作质量：15 关键点、30 Hz、归一化，计算覆盖率、姿态熵和 Motion-MMD，并记录参考数据与聚类参数。
21. 复现并行性能：固定硬件与同步计时，分别测 human-only / RL inference / RL training 的环境吞吐和 agent 吞吐；保存配置、日志与绘图输入。
22. 实机实验单列：Scout Mini、双 D435i、Mid-360/FAST-LIO、标定、控制与时序链路。纯仿真复现不需要先完成该项。

论文依据为仓库本地 NavIsaacLab PDF 的第 IV–VI 节；此前逐项论文差异已核对。上面步骤是实施与验收计划，不代表这些阶段已经完成。

## 6. 最短行动清单

- [x] 当前源码语法检查和跨模块显式导入核对。
- [x] 38 项轻量测试、9 项核心配置/状态测试。
- [x] 轻量 PPO 训练→保存→恢复→评估闭环。
- [ ] 修复不存在的兼容迁移函数与检查点适配。
- [ ] 解决 moviepy 冲突，建立专用的匹配 IsaacLab 环境。
- [ ] 补齐场景、动作、模型与地图并校验内容。
- [ ] 修复评估观测/reset/课程、尾批 NaN 和采集变量。
- [ ] 完成 1 人体+1 机器人真实物理、视觉、训练与评估闭环。
- [ ] 完成多环境隔离、重置及性能验收。
- [ ] 补齐论文扩散轨迹、AMP 配置、视觉感知、时空策略和评测数据。
- [ ] 复现论文各表格、消融与并行性能曲线。

相关背景：[依赖审查](DEPENDENCY_REVIEW_ZH.md)、[此前代码审查](REPOSITORY_REVIEW_ZH.md)。涉及兼容性的最新结论以本报告证据为准；资源状态以第 0 节及资源审查为准。
