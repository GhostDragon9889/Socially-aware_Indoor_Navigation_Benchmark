# 项目代码全面审查（AMASS 处理完成后）

> 本篇保留迁移前的代码发现。后续已修复行人控制链的配置、四元数、子集 reset、推理冻结、评估循环与采集入口等问题，并完成四人8秒真实模拟和1,750项回归。
> 修复范围、运行证据和保留问题见 [Isaac Sim6 / IsaacLab3 迁移说明](ISAAC_SIM6_LAB3_MIGRATION_ZH.md)。这不表示本文所有非行人问题或所有后端均已解决。

审查时间：2026-09-30 至 2026-10-01（Asia/Shanghai）。基准提交：`7a8417a`，审查对象为实际工作区，包含未跟踪的 CrowdSim、src、tests、configs 和数据准备脚本。沿用此前要求，不展开 FLUX、NavDP 内部实现。本轮新增审查证据与报告，没有修复业务代码或修改全局环境。

## 结论

**当前项目尚不能认定为可可靠运行的完整 NavIsaacLab 训练/评估系统。** AMASS 的三个项目通用动作库已补齐，主要阻塞已转向运行时配置迁移、四元数接口、人体异步 reset、评估循环和环境依赖。现有单元测试没有覆盖这些完整调用链。

本报告列出 18 项发现：7 项 P1、11 项 P2。P1 表示主要入口阻塞、关键状态/评估错误或凭据泄漏；P2 表示条件触发的正确性与交付问题。语法扫描覆盖全部范围内文件，语义审查集中在关键调用链，不能理解为逐行证明全部代码正确。

## 一、主要发现

### F01 · P1 · 运行时导入不存在的检查点兼容函数

位置：[protomotions_runtime.py:122](../CrowdSim/protomotions_runtime.py#L122)，调用在 132 行。

`configure_from_checkpoint()` 导入 `apply_backward_compatibility_fixes`，目标 `protomotions/utils/inference_utils.py` 没有此定义或重导出。其现有入口是 `apply_all_inference_overrides`，签名和职责不同。本轮静态核对 18 条 CrowdSim → ProtoMotions 的 from-import，此项为唯一未找到目标名称的记录。前面的资源/依赖问题解决后，这里仍会中止构建。

修复：按当前官方 inference_agent 的流程迁移配置；实际恢复现有 checkpoint 并构建环境。不要仅添加空兼容函数。[导入证据](code_audit_20260930/cross_imports.json)。

### F02 · P1 · 机器人朝向与人体网格仍按 wxyz 解释 xyzw

位置：[drive.py:149](../CrowdSim/control/drive.py#L149)、[nav_manager.py:2348](../CrowdSim/nav_manager.py#L2348)、[smpl_mesh_visualizer.py:168](../CrowdSim/tools/smpl_mesh_visualizer.py#L168)。

当前核心 `IsaacLabSimulatorConfig.w_last=True`；项目锁定的 IsaacLab 提交也明确将 `root_quat_w` 标注为 XYZW。然而机器人 yaw 把第一个分量当 w，网格适配器也将同一数据直接作为 `root_quat_wxyz` 传入。CPU 输入单位旋转 `[0,0,0,1]`，实际 `DriveController.yaws()` 返回 **180°，应为 0°**。这影响朝向观测、地图旋转和运动控制，并可能使人体网格与物理骨架不一致。

修复：在明确的后端边界统一格式，并同时覆盖读取、写入、reset、相机和网格；不能只改一个 yaw 公式。依据：[锁定提交的根姿态接口](https://raw.githubusercontent.com/isaac-sim/IsaacLab/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py)。

**类型问题的边界**：本地相邻 IsaacLab 的 ProxyArray 有 Torch 操作兼容桥，不能断言每处直接 Torch 操作都必然失败。锁定提交 ProxyArray 文件未成功下载；目标版本的具体桥接行为仍需实际环境验收。四元数顺序错误则有本地代码、锁定接口和数值复现三方面证据。

### F03 · P1 · 子集 reset 的人体目标使用了错误的全局索引

位置：[nav_manager.py:899](../CrowdSim/nav_manager.py#L899)、[nav_manager.py:938](../CrowdSim/nav_manager.py#L938)、[nav_manager.py:957](../CrowdSim/nav_manager.py#L957)、`_humanoid_future_targets_from_path():2537`。

核心 reset 会以 `env_ids` 构造子集上下文；原 MaskedMimicControl 正确选取这些 ID。CrowdSim 随后覆盖上下文时只使用子集张量长度，仍从 `self.reached[:N]`、`motion_times[:N]` 以及从 0 开始的路径/速度数组取值，没有使用 `ctx.env_ids`。

本轮使用未修改的生产方法 AST 和真实 `MaskedMimicContext` 复现：只重置行人 1，两个全局时钟为 `[10,20]`，目标偏移 0.5；目标时间实际为 **10.5，预期 20.5**。全局 reached 为 `[True,False]` 时，行人 1 的目标 mask 实际为 **0，预期 1**。路径生成器在该检查中返回零目标，仅隔离时间与 mask 的索引错误；没有模拟真实物理。

修复：显式建立局部行 → 全局 env ID 映射，统一索引路径、速度、时钟、reached 和冻结槽位；增加非连续 ID、单行 reset、含冻结槽位的回归测试。[复现](code_audit_20260930/subset_reset.json)。

### F04 · P1 · 评估循环持续使用初始人体观测且不处理人体 done

位置：[eval_policy.py:360](../CrowdSim/tools/eval_policy.py#L360)、371、403 行；`BaseEnv.get_obs():348`。

循环前得到 obs，循环内丢弃 `env.step()` 返回的新 obs；核心观测返回克隆，不会自动更新旧字典。人体 done 只写入局部 done_indices，没有在下一步消费。机器人虽读取新观测，人体控制已偏离正常闭环，评估中的人群分布和避障结果因此不可信。

修复：更新人体 obs 并处理对应 reset，验收连续多步状态变化及摔倒/终止恢复。本轮为生产调用链静态确认，未启动 IsaacLab。

### F05 · P1 · 专家轨迹采集入口有三个未定义变量

位置：[collect_ppo_buffer.py:861](../CrowdSim/flow/collect_ppo_buffer.py#L861)，861–863 行。

`main()` 传入 `repeat_robot_routes`、`target_episodes`、`route_repeats`，但该作用域没有给它们赋值。另一个函数的同名形参不能提供这些变量。Ruff 本轮再次报告三个 F821；执行到调用处会触发 NameError。

修复：从 CLI/YAML 合并后的 collection 配置取值，验证 main → collect 的参数传递。另一个 F821 位于 train_ppo.py:312 的延迟求值类型注解，影响不同，不应混算为第四个立即执行错误。

### F06 · P1 · 指定 IsaacLab 工作区与根包的 moviepy 约束不相容

位置：[pyproject.toml:43](../pyproject.toml#L43)，`protomotions/utils/simulator_imports.py:13` 的锁定提交。

根包在 Python≥3.9 要求 `moviepy>=2,<3`；指定 IsaacLab 开发工作区要求 `moviepy>=1.0.3,<2.0.0.dev0`。二者在同一环境中无版本交集。本轮重新读取了[锁定提交的工作区依赖](https://raw.githubusercontent.com/isaac-sim/IsaacLab/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/pyproject.toml)。这是同时安装该开发工作区与根包时的冲突，不是宣称所有安装方式都必然报同一错误。

修复：选定后端基线，核对实际视频 API 后调整声明或隔离工具环境；以依赖一致性和启动检查验收，不能把 `--no-deps` 当作解决冲突。

### F07 · P1 · SLURM 将 W&B API key 写入可持久传播的脚本与日志

位置：[train_slurm.py:830](../protomotions/train_slurm.py#L830)、1089、1094 行。

从本地/交互取得的 key 被拼成 `export WANDB_API_KEY=...` 放入 job_cmd；完整脚本随后打印、写入 `delete=False` 临时文件并上传 submit.sh。该分支会让凭据进入终端日志和持久作业文件。此次只读代码，没有读取真实 key，也没有提交作业。

修复：采用作业环境或集群秘密配置传递，日志脱敏，避免把凭据写入可复制作业脚本。

### F08 · P2 · 普通 PPO 的单样本尾批导致 NaN 参数污染

位置：[ppo_policy.py:857](../CrowdSim/ppo/ppo_policy.py#L857)，同样存在于 src/crowdsim_ppo/ppo_policy.py。

batch 迭代保留尾批，而 `returns.var()` 默认使用样本方差。3 个 rollout 样本、minibatch_size=2、1 个 epoch 的 CPU 复现得到 value_loss 非有限，**11 个参数张量非有限**。两份策略实现字节完全相同，需同时处理或去重。

**更正此前笼统表述**：`update_with_kl()` 的 978 行也产生单样本方差警告，但 1025 行有 NaN 梯度跳过逻辑。本轮两个副本的 KL 分支参数均保持有限；不能宣称它与普通 update 一样已复现参数污染。它仍会丢弃该批更新，且只检查 NaN、不检查所有非有限值。

修复：采用适当的总体方差或 rollout 级尺度，普通与 KL 更新统一检查有限 loss/梯度；覆盖余数为 1 的 batch。[复现结果](code_audit_20260930/reproductions.json)。

### F09 · P2 · 动作采样权重更新重新启用被长度条件排除的动作

位置：[motion_manager.py:108](../protomotions/envs/motion_manager/motion_manager.py#L108)、438 行；`mimic_evaluator.py:124`。

初始化将短于 truncate+env_dt 的动作权重置零，evaluator 的正数下限又将它们抬高。update_sampling_weights 仅重新应用显式排除 ID，不重新应用长度资格。未修改的 MotionManager AST 类体复现权重 **[0,1] → [0.5,1]**，不再满足排除约束。

修复：持久维护可采样 mask，在初始化、权重更新和恢复时统一应用。

### F10 · P2 · 环境数多于动作数时忽略显式动作 ID 列表

位置：[motion_manager.py:124](../protomotions/envs/motion_manager/motion_manager.py#L124)。

`num_envs > total_motions` 时提前返回，显式列表分支不执行。2 个动作、3 个环境、合法重复列表 `[1,1,1]`，实际 available_motion_ids=None，后续进入随机采样。

修复：先处理与验证显式列表，再处理自动子集模式。

### F11 · P2 · 评估任务的距离与起点不等于声明值

位置：[eval_policy.py:231](../CrowdSim/tools/eval_policy.py#L231)、340、360 行；`goal_curriculum.py:128`、`nav_manager.py:418`。

评估直接使用开启课程的默认 YAML。课程初始化应用第一阶段 2–4 m，覆盖配置中的 7–10 m。另一方面先手工传送机器人再 env.reset(None)，首次完整 reset 会写回初始 spawn poses；后续手工 reset 也没有统一走控制缓存和传感器同步。

修复：评估显式固定课程开关和任务分布，统一指定路线 reset，并记录最终配置及首帧真实起终点。本轮为源码与配置确认；未重新运行课程数值案例或 GPU episode。

### F12 · P2 · 轻量 PointGoal 的边界裁剪破坏 reset 分布

位置：[point_goal_env.py:75](../src/crowdsim_ppo/point_goal_env.py#L75)、85 行。

目标先采样再 clamp 到地图边界，实际距离缩短、方位改变，但 heading 仍用裁剪前角度。seed=42、10,000 个环境、距离 2–4 m、朝向偏差 0–15°：**1,875 个距离不足 2 m；1,956 个偏角超过 15°**，最短约 0.099 m，最大偏角约 93.18°。

修复：采样满足边界与距离约束的目标，并按最终目标方向生成朝向。

### F13 · P2 · 旧偏好训练将补齐帧纳入轨迹奖励

位置：[pref_train_reward.py:102](../CrowdSim/pref/pref_train_reward.py#L102)、119 行。

变长 episode 补零到 batch 最大长度，却不返回有效帧 mask；奖励直接对全部时间步求和。实际网络在 eval 模式下，同一条 2 帧轨迹单独评分为 **-0.525117**，与 4 帧轨迹同 batch 时变为 **-0.703892**。真实两帧的和仍约 -0.525117，差值来自补齐帧，不是 dropout。偏好标签比较因 batch 组成改变。

影响限定在旧 `pref_train_reward.py` 分支；不能直接套到固定长度的 `train_offline_reward.py`。旧训练分支的验证阶段也未切 eval，另有可重复性风险；本轮没有量化该风险。

修复：返回并应用有效帧 mask，确保同一 episode 的评分与 batch 伙伴无关，验证阶段切换 eval/train。[复现](code_audit_20260930/preference_padding.json)。

### F14 · P2 · latest 检查点直接覆盖，恢复可能读到不完整或混代数据

位置：[agent.py:484](../protomotions/agents/base_agent/agent.py#L484)、501 行；[ppo_policy.py:1048](../CrowdSim/ppo/ppo_policy.py#L1048)。

直接 torch.save 到目标路径，写入失败/进程中断会破坏原检查点。核心 agent 与 env 分开覆盖，也缺少同一保存代次的提交机制。本轮静态确认，未模拟磁盘故障。

修复：同目录临时文件完成后原子替换；多文件用版本目录和提交清单维护一致性。采集器已有原子保存实现可参考。

### F15 · P2 · SLURM 安装失败后仍发布 ready 标记

位置：[train_slurm.py:771](../protomotions/train_slurm.py#L771)、811–819 行。

pip 安装与 touch 用分号连接，没有按安装返回码决定是否生成完成标记。失败的 rank 0 仍可能通知其他 rank 继续。相邻 warmup 阶段有失败标记，但不能覆盖更早的 pip 失败。

修复：只在安装成功时发布 ready，失败传递退出码及失败标记；以模拟安装失败验证所有 rank 终止。

### F16 · P2 · episode 事件计数可能重复计入成功率分母

位置：[nav_manager.py:1936](../CrowdSim/nav_manager.py#L1936)、[train_ppo.py:632](../CrowdSim/ppo/train_ppo.py#L632)。

reached、collision、timeout 是独立 mask，同一步可以重叠；logger 将三者及 stuck 相加作为总 episode 数。例如一次同时 reached 与 timeout 的 episode，实际完成数为 1，日志分母却为 2。stuck 已排除其他 terminal，另外三种没有互斥处理；课程仍将 reached 当成功。

修复：区分互斥终止原因与可重叠事件，完成数使用 done 的计数，并明确发生碰撞时的成功定义。

### F17 · P2 · empty 模式仍要求人体资源和已有地图

位置：[builder.py:95](../CrowdSim/world/builder.py#L95)、100、121、300 行。

分支进入 empty_mode 前已读取地图并无条件验证 humanoid checkpoint/motion_file。单纯打开场景、准备地图仍受人体资源或地图缺失阻断。普通模式的预检也只检查 exists，未检查推理配置或 LFS 指针。

修复：按模式建立资源清单，empty 模式先走最小场景路径，完整模式检查需要的真实文件与配置。

### F18 · P2 · 当前工作区、安装说明与 CI 的交付范围不一致

位置：[FULL_PIPELINE.md:65](../FULL_PIPELINE.md#L65)、[pyproject.toml:149](../pyproject.toml#L149)、[public-ci.yml:194](../.github/workflows/public-ci.yml#L194)。

文档引用不存在的 check_isaaclab3、validate_isaaclab3、constraints_isaaclab3.txt，并称 `[test]` extra 安装独立 crowdsim-ppo；根包没有该 extra，setuptools 只包含 protomotions。`git ls-files CrowdSim src configs tests` 本轮返回 0，这些目录不属于当前提交。CI 只运行 protomotions/tests，且 job 限定 NVlabs/ProtoMotions，自己的 fork 默认不执行。

修复：明确源码运行与包安装契约，修正文档/缺失入口，将应交付源码纳入版本管理，并在 fork CI 中加入 CrowdSim 与轻量测试。本轮没有替用户暂存或提交文件。

## 二、结构与依赖覆盖

| 子系统 | Python 文件数 | 本轮检查重点 |
|---|---:|---|
| protomotions | 352 | 配置恢复、状态/上下文、动作采样、PPO、检查点、集群启动、后端接口与现有测试 |
| CrowdSim | 59 | builder/runtime、机器人控制、人体目标、step/reset、评估、采集、奖励模型 |
| src / tests | 7 / 3 | 轻量 PointGoal reset、PPO 数值与重复实现 |
| data | 37 | AMASS 脚本、配置、数据准备边界 |
| examples / deployment | 47 / 7 | 语法、配置/部署入口及核心已有相关测试；未实际 ONNX 部署 |
| pyroki / usd_convert / scripts / docs | 2 / 5 / 4 / 2 | 静态扫描；未运行重定向或完整 USD 转换 |

主链路：

```text
配置/检查点 → ProtoMotions runtime → BaseEnv + IsaacLab
                                     ↓
地图/A*/SFM → CrowdNavigationManager → 人体目标 + 机器人控制
                                     ↓
                         PPO / 评估 / 专家采集
                                     ↓
                          偏好数据 → 奖励模型 → PPO 微调
```

- 本轮扫描工作区原有 **525 个 Python、4 个 Shell 文件**，均通过 AST / bash -n；不包括审查过程中新增的复现脚本。18 条跨项目 from-import 另行核对。
- openAIbase：Python 3.12.13、Torch 2.12.0；缺 IsaacLab/Sim、Lightning、TensorDict、OmegaConf、dm-control。Torch 超出项目 IsaacLab extra 的 `<2.12` 范围。
- MuJoco 基础环境：Python 3.9.25、Torch 2.8.0、dm-control 1.0.34；不是项目目标 IsaacLab 环境。前一任务建立的 AMASS 临时 venv 用于本轮真实上下文类检查。
- marlbase：Python 3.10.16、Torch 2.4.0、OmegaConf 2.3.0，同样缺完整仿真训练依赖。这些检查只代表列出的环境，不能推断整台机器都没有其他安装。
- CrowdSim 使用 OpenCV/SMPL mesh 等额外组件，尚无完整独立依赖/安装契约。纯推理 runtime 仍采用 `agent.load(..., load_env=False)`，没有像核心 inference_agent 一样关闭 load_training_state；需用真实 checkpoint 验证，未将其直接判为必然崩溃。
- 导航 manager 的网格更新异常被静默吞掉（471 行）。即使主循环继续，也不能证明人体视觉正常。

## 三、AMASS 当前状态

三个通用 MotionLib 已生成：validation 345 段、test 138 段、train 11,303 段；合计 11,786 段、4,668,382 帧。默认验证库已在 CrowdSim 预期路径，不再列为缺失。

这沿用上一任务实际转换和加载校验的证据，本轮未重复读取全部 9.6 GB 级训练库。数据准备脚本使用原生转换器，保留来源、裁剪范围、源码哈希及输出校验；原始包保留。脚本不会覆盖现存最终 .pt，重复校验应使用 `--stage verify`。论文 2,848 段精确筛选清单仍未取得；通用库可加载不证明论文结果已复现。详见 [AMASS 记录](AMASS_PREPARATION_ZH.md)。

## 四、本轮验证结果与边界

| 验证 | 结果 | 解释 |
|---|---|---|
| AST / bash -n | 525 / 4 全部通过 | 语法检查，不是全语言版本或语义证明 |
| Ruff F821/F822/F823 | 4 个 F821 | 3 个采集执行变量，1 个延迟类型注解 |
| CrowdSim/tests + tests | 38 passed，2 warnings | warnings 为测试不完整对象析构缺属性 |
| 核心测试 | 297 passed、4 skipped、24 failed、86 collection errors | 使用 --continue-on-collection-errors；没有把未收集部分算作通过 |
| 24 个执行失败分类 | yaml 15、dm_control 4、typer 3、trimesh 2 | 均为 ModuleNotFoundError，不是已证明的业务断言失败 |
| PPO 两副本 | 普通 update 均污染 11 个参数张量；KL 分支本例未污染 | 真实 CPU 实现 |
| 四元数 | 单位 xyzw 被计算为 180° | 实际 DriveController + 合成状态；未模拟物理 |
| 人体子集 reset | 时间 20.5→10.5，mask 1→0 | 原样方法 AST + 真实上下文类，路径生成器隔离 |
| 动作权重 / 显式列表 | 均复现 | 原样 MotionManager AST 类体，隔离缺失导入 |
| PointGoal reset | 距离/角度违反率 18.75% / 19.56% | 真实 CPU 环境，固定 seed |
| 旧偏好奖励 padding | 同一轨迹评分随 batch 改变 | 实际网络 eval 模式 |
| IsaacLab GPU、渲染、物理接触、完整训练恢复 | 未运行 | 依赖和接口仍未闭合 |

审查材料：

- [逐文件清单与哈希](code_audit_20260930/inventory.json)
- [导航测试日志](code_audit_20260930/crowdsim_tests.log) / [JUnit](code_audit_20260930/crowdsim_tests.xml)
- [核心测试日志](code_audit_20260930/core_tests.log) / [失败分类](code_audit_20260930/core_test_failures.json) / [86 个收集错误的缺依赖分类](code_audit_20260930/collection_error_categories.json)
- [CPU 复现脚本](code_audit_20260930/reproduce_findings.py)
- [子集 reset 复现脚本](code_audit_20260930/reproduce_subset_reset.py)

证据目录沿用开始审查时的 20260930 日期，报告完成于 20261001。

## 五、建议修复顺序与验收

1. **构建能运行的目标环境**：解决依赖冲突和 F01；真实检查点恢复、组件构造通过。
2. **统一状态与 reset**：修 F02/F03，验证多步朝向、人体骨架/网格对齐、非连续 env_ids reset。
3. **恢复可信评估和采集**：修 F04/F05/F11，固定任务清单与首帧状态，验证人体 done 和机器人终止统计。
4. **稳定训练**：修 F08–F10/F12/F13/F16，覆盖尾批、动作资格、变长奖励和互斥终止口径。
5. **可靠交付**：修 F07/F14/F15/F17/F18，验证失败退出、原子恢复和 fork CI，再进行短训练→保存→恢复→评估闭环。

在上述问题修复及真实仿真验收前，不应将现有评估输出作为可靠论文 benchmark，也不能用 AMASS 已生成或单元测试通过替代完整闭环验证。
