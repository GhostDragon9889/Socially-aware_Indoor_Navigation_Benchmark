# AMP 专家动作、时间与旋转误差：只读结论

2026-10-03。本项只读生产185及冻结q935/q936源码、此前已经完成的诊断文档、作者论文和源码。没有导入Torch/SDK、加载权重、重算指标、读取正在运行或未来capture，也没有修改生产、训练bundle、数据、默认模型。下列机制效果仍需真实训练及完整评估验证；`future_effect = false`。

## 1. 论文与作者实现的边界

[NavIsaacLab IV-B/IV-C](https://arxiv.org/html/2606.26265v1#S4.SS2)写状态历史与当前动作的判别器，并报告按参考序列同步的身体误差。论文没有公开专家控制动作如何从AMASS恢复、PD动作缩放、判别器历史布局，以及旋转指标的身体/坐标/失败帧聚合实现。因此本地 `.7829 rad` 与文中 `.343 rad` 的差距是真实待解释的数值差距，不能直接归因于某一实现bug或认定两套指标完全同口径。

[AMP §5.1/§5.3](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)明确mocap只观测状态、控制动作未知，所以采用状态转移判别器。其运动奖励不指定某一参考clip；判别特征在人物朝向坐标中编码。公开[PACER任务源码](https://github.com/nv-tlabs/pacer/blob/main/pacer/env/tasks/humanoid_amp.py)在`post_physics_step`更新AMP历史，[作者agent](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/learning/amp_continuous.py#L124)在`env_step`之后保存`infos['amp_obs']`。两者提供可核查的state-only、post-step实现依据，不能把它冒称为Nav私有完整训练实现。

本地作者源码副本及固定commit逐字节来源已在[既有commit记录](../paper_fidelity_research_v1/pacer_commit_binding.json)保存：PACER `baeb6241f03fc9d78370a64e3c2dcef9d830d97f`，agent `f88962cf…`、motion lib `ff5b793b…`。本轮没有重新建立来源框架。

## 2. 专家动作并没有错配一帧，但并非真实控制标签

| 核查项 | 具体源码 | 已确认结论 |
|---|---|---|
| 普通专家窗口 | `CrowdSim/humanoid/amass_corpus.py:172–177` | newest-first历史`t,t-1,…,t-9`，动作代理来自`dps[t+1]`；窗口不越clip边界。 |
| q分层专家窗口 | `training/expert_sampling.py:409–430,518–527` | 同样`absolute-arange(10)`及`absolute+1`，不是把`t+1`状态放进`t`历史。原corpus draw仍发生以保留shared reset RNG消耗，实际expert由独立private RNG抽取。 |
| 代理动作 | `amp_world.py:268–279` | `a*_t=clip(q_mocap[t+1]/scale,-1,1)`；hand/toe目标零。`scale`实际来自同一`action_to_pd(ones)`，按PACER/实际DOF顺序重排。 |
| 正向动作解码 | `amp/controller.py:24–61` | `q_target=offset+scale*a`；本配置offset=0、球关节范围1.2×max limit且capπ、膝y scale5。输出关节目标而非扭矩。 |
| 实采动作 | q935 `scripts/train_paper_amp.py:723–750` | clone pre-step `h[t]`，采样及clamp/mask后的`applied a[t]`用于D，然后执行physics。D/replay使用同一记录，未发现动作与历史错配一帧。 |
| D归一化 | `training/trainer.py:290–303` | 历史按1950个坐标RMS；追加69 action保持归一化PD数值，不套AMP RMS；12冻结列必须零。 |

`inverse_pd_target_from_mocap_pose`实际是**运动学PD目标反解**。它不使用当前关节状态/速度、惯性、接触、kp/kd或必要扭矩；同一下一帧姿态可以由多个控制目标达到。因而D可能区分“运动捕捉下一帧姿态代理”与“真实物理控制指令”而形成额外分类捷径；这是有源码依据的风险假设，尚未读取D权重或证明它导致旋转误差。不能把代理改名后称作者inverse-dynamics标签。

单纯把代理从`t+1`改`t`会改变标签意义，不能当作已发现的一帧bug修复。增加PD提前量也属于待验证的工程动作代理，不是公开Nav专家标签恢复。

DOF速度也有明确合同：q935/q936 profile指定`child_joint`，本地`amass_corpus.py:151–165`把已声明parent-frame速度按当前local quaternion逆旋转到child frame。作者motion lib的`local_rotation_to_dof_vel`（`pacer_motion_lib_smpl.py:44–50`）采用`Log(q_t^-1 q_{t+1})/dt`，同属child-frame角速度；本轮未发现把两者误当exp-map分量导数的源码证据。实际backend关节速度与其对应仍须区分于作者运动捕捉速度的定义，不从此宣称全物理状态完全一致。

## 3. q935 / q936 的准确机制

父q935：`output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1/code`；manifest `2e14550a64a5c6d091d64312e2799eb3854e88e9dacdeffacd9990297ced8359`；选用`isolated_terrain_state_prior_stratified_absolute.yaml`。

matched q936：`output/isolated_training/mixed_state_terrain_state_prior_stratified_matched_v1/code`；manifest `5c2a22098bf8df60530b5f53963c041a221cc17e62759284d7586f6ea6a05f50`；选用`isolated_state_only_matched_q_absolute.yaml`，不能误选bundle内保留的旧profile。

| 路径 | 判别器输入 | 奖励/rollout/replay时刻 | 专家动作是否进入D |
|---|---|---|---|
| q935 | 1950历史+69 action=2019 | pre-step `h[t]` + `a[t]` | 是，下一帧PD代理 |
| q936 matched ablation | 1950历史 | pre-step `h[t]`，有意与q935一致 | 否 |
| 原PACER式state分支 | 1950历史 | post-step `h[t+1]`，在reset前 | 否 |

q936直接移入公共1950 D及history RMS，沿用q分层采样、奖励内2/外2、task `.5`、PD、seed7和全部物理设置；没有动作扩展69列。初始化/审计已经另有冻结证明，本轮不重复加载。此表是机制对照，不包含当前新模型质量结论。

pre-step state奖励在当前步对`a[t]`没有直接依赖；动作后果会影响后续状态奖励和价值，因此不能说“没有学习信号”。post-step会改变信用分配与rollout奖励时刻，这是应独立检验的变量。

## 4. 为什么低XY/MMD误差不能保证低world旋转误差

1. **任务缺少唯一参考姿态/相位。** `amp/observations.py:132–172`的task只有10个未来XY点和terrain；`benchmark_paper_motion_tracking.py:307–336`每步只传参考root XY，不传参考全身rotation/DOF/phase。当前自状态可以间接推断自身动作相位，但同一XY轨迹并不唯一确定目标上身姿态或参考相位。D在整体expert分布中奖励动作，未按该参考clip同一帧监督。
2. **训练坐标与主指标观察不同。** `amp/observations.py:175–192`从root旋转移除heading，用局部根速度、19个关节旋转/速度、4个末端位置共195维。`motion_tracking.py:86–116`却对24个body的世界XYZW quaternion做`2 acos(abs(dot(q,q_ref)))`，不做yaw/root alignment或相位匹配。根yaw或上游姿态误差会传播到多个身体；这是不同目标，不是quat符号错误。
3. **四元数约定自洽。** 同一common state使用XYZW、显式`w_last=True`；主指标归一化且用abs(dot)，对`q/-q`不敏感。`rotations.py:704–740`heading为旋转后+X轴在XY的投影；`amp_world.py:221–225`planar yaw公式对单位quat给出相同投影heading。terrain中心专用Z/W投影只用于采样高度，不能与全身误差混为一谈。未发现本轮quat顺序或统一yaw差90°的源码证据。
4. **已有同帧诊断支持上游问题。** [已完成的诊断说明](../state_prior_pacer_reward_pose_diagnostics_v1/README.md)记录旧19f wrist world旋转大，而hand相对wrist旋转极小；不能把hand world误差直接当成hand控制失败。该诊断亦说明90D使用位置/速度、没有身体quaternion，且无Hand点；它不能替代world旋转指标。这里只引用旧报告，没有重新计算或消费新capture。

另一个需要区分的表示问题：作者motion lib从local quaternion取exp-map DOF（`pacer_motion_lib_smpl.py:605–608`），作者`dof_to_obs_smpl`和本地D都再用exp-map编码；实际IsaacLab则读关节标量并原样写PD目标（`simulator.py:661,691–695,817–825`），MJCF声明三个XYZ hinge。**这些来源事实不足以证明PhysX D6标量就应按某一Euler顺序转换，也不足以证明这是新引入bug。** PACER原码同样采用这种exp-map观察约定；更改live或expert表示会改变公共D/RMS输入意义，不能悄悄改观测后沿用原公共模型并声称等价。若进一步调查，应先核实实际D6关节到body-local quaternion的表示关系，仅用现有状态/模型无关的校验；本轮未执行。

## 5. 推荐下一单变量：state-only奖励时间

等当前q936完整22/fixture结果闭合后，若root决定继续，首选独立fresh试验：**1950 state-only D的AMP reward、rollout history、history_episode和replay一起由pre-step改post-step（reset前）**。使用同q936父源、公共D/RMS初始化、q expert sampler、seed7、奖励比例、模型/PD/physics/trajectory/budget；不改专家action代理，不增加3Dpose条件，不同时改D损失或镜像项。

所需改动就是把q936原三runner片段恢复为父q935已有state-only post-step分支，再准确声明timestamp；无需新网络、输入维度、数据或推理观测。AMP history始终newest-first10×195，不跨reset；rollout和replay必须消费奖励对应的同一post-step历史。以fresh run比较完整22主world旋转/位置/失败clip、原同帧root/parent decomposition、fixture全6000物理与路线；全部失败/idle/reset/unknown bins保留，MMD支持桶差异另列，不作启动过滤。

该实验回答“去掉动作代理后，奖励是否应及时观察动作后果”，属于公开PACER机制时间消融；仍不是严格Nav Eq2最终模型，也不是保证达到论文旋转误差。若相位/上身误差仍大，再决定是否需要独立的姿态控制研究；未来3Dpose输入会改变部署问题，不能冒称原2D Nav方法。
