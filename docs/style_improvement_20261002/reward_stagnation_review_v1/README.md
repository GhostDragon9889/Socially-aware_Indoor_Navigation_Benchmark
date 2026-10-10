# 奖励、弱参考与停滞：独立 CPU 数学核对

**结论范围：只读分析与未实施的单项候选。** 未修改生产181/隔离931代码、模型、AMASS或评估公式，未启动CUDA/Kit，未读34个test序列。数值分析执行时当前保护统计模型的导航尚未终态，本报告不将其进度或旧模型停留认作本轮原因。随后root通知本轮真实导航已closed0：6000控制步/200s、0摔倒0超时、顶点[7,6,7,5]共25；这些是root终态通知，原始时序/style由其他CPU读者审核，本工具没有重复测量或提前授予质量通过。**目前暂缓候选实施。**

## 1. 当前公式允许哪些行为

核对的是实际保护统计训练的冻结931源码，而非当前可变工作树：

- [settings.py](/project/EXP/SangforAndSaut/ProtoMotions/output/isolated_training/mixed_state_action_terrain_protected_v1/code/CrowdSim/humanoid/application/settings.py:109)、实际profile：`amp_reward_scale=2`、`amp_reward_weight=2`、`task_reward_weight=.5`、`direction_mode=backwards_penalty`。
- [trainer.py](/project/EXP/SangforAndSaut/ProtoMotions/output/isolated_training/mixed_state_action_terrain_protected_v1/code/CrowdSim/humanoid/training/trainer.py:276)：style映射为 `2*min(softplus(z), -log(1e-4))`；外层再乘2。有效style为 `4*min(-log(1-D), 9.21034)`。这是已声明的PACER映射及权重适配，**不是新发现的未声明bug**。
- [rewards.py](/project/EXP/SangforAndSaut/ProtoMotions/output/isolated_training/mixed_state_action_terrain_protected_v1/code/CrowdSim/humanoid/training/rewards.py:43)：task为 `exp(-2*||root-target||²) -.0005*P + exp(min(v_forward,0)/1.8)-1`；`P`为已同步4子步的signed power。
- [实际控制循环](/project/EXP/SangforAndSaut/ProtoMotions/output/isolated_training/mixed_state_action_terrain_protected_v1/code/scripts/train_paper_amp.py:721)：D2019使用pre-step历史和同次applied action；位置/能量/身体前向速度在post-step求值；目标是本次消费参考的同帧XY。

因此身体前向速度为0或任何正值时，direction均为0；它不直接奖励期望速度、路径进展或身体朝向误差。原地朝向错误但不后退不会被该项直接惩罚。XY未来点也没有规定必须朝向路径的body yaw，不能给“转弯路径率”冒充身体yaw目标。根速度为0不等于所有关节速度为0；只有后者与有限effort结合才保证P=0。signed P<0时energy为正，站姿是否实际占优还须看记录。

只比较固定D、P=0的合成数学情形：目标误差0.083m时，位置项仍为0.986316，外层贡献0.493158；D=.15时style贡献0.650076，合计1.143234。这里D=.15是**假设数值**，不是导航记录中的实测站姿D。

这符合[AMP原论文的任务/风格分工](https://xbpeng.github.io/projects/AMP/)，不能从style目标自动推出前进。[NavIsaacLab IV-B](https://arxiv.org/html/2606.26265v1#S4.SS2)给出history/action D与位置、signed能量、方向项；本地softplus映射、负方向适配及额外候选均不声称作者精确实验。

## 2. 实际旧导航能区分什么

对已封存旧state-action模型四人200s的原始physical_states/reference_plans重新读取：6000帧全保留，仅控制差分排除2个跨episode间隔；frame0之前的4个actor区间未保存，不伪造初态。通过实际replan步骤重建pre-action `p(t+.5)`，所有post-step目标与原保存值逐位一致（误差0）。请求速度由 `[p(t+.5)-root_pre]/.5`得到，是现有raw task前2项的世界坐标等价量；**这是CPU重建，不是新保存的真实pre-action observation**。

| 固定旧记录分组 | actor-s | 实际区间速度 m/s | 请求速度 m/s | 位置奖励均值 | 69DOF native速度RMS rad/s |
|---|---:|---:|---:|---:|---:|
| 同episode全部 | 799.800 | .174976 | .239367 | .982621 | .319120 |
| idle：实际速度<.1 | 591.733 | .012855 | .101917 | .989352 | .216762 |
| idle且请求<.1 | 327.600 | .011947 | .084184 | .989228 | .206300 |
| idle且请求≥.1 | 264.133 | .013982 | .123911 | .989506 | .229737 |
| 非idle | 208.067 | .636040 | .630268 | .963478 | .610221 |

idle时实际位置奖励接近1，signed energy均值约-.009025；低根位移同时仍有关节运动。因此可以明确区分**参考请求弱**与**非零请求未跟上**，但不能把所有idle叫完全静止姿态或认定D认为其优秀。旧导航没有保存实际AMP reward、applied action或完整D历史输入，不能从这些文件复算真实style奖励，更不能用训练平均D概率替代。没有创造接触点滑动或AMASS真实力GT。

训练metrics保存整batch的mean/min/max（[trainer.py](/project/EXP/SangforAndSaut/ProtoMotions/output/isolated_training/mixed_state_action_terrain_protected_v1/code/CrowdSim/humanoid/training/trainer.py:450)），未保存按请求速度/idle/real-procedural分组的style与task分量；平均forward速度不能证明没有原地模态。实际2000行逐行验证总reward均值等于 `2*amp_mean+.5*(trajectory+energy+direction)`，最大浮点差<1e-6。

保护统计模型1901–2000均值：style外层.684617、task外层.420002、位置.974978、forward .706146m/s。旧state-action对应.660552/.422567/.979855/.710367。这是训练池边际统计，不能据其推断导航停滞原因。

本轮已完成full22为15/22成功、位置.247776m、旋转.775494rad、90D MMD² .054529；旧state-action为17/22、.208382/.725853/.062530。风格汇总略好而跟踪退步，根XY/yaw误差也增加，heading-local身体位置与parent-relative关节旋转接近。它们不证明同一因果链，也不替代尚未结束的导航记录。

## 3. 仅一个可注册候选：命令相对速度成本

**未合入、未训练，需当前导航终态后决定。** 保留D2019、专家均匀抽样、原reward各项、AMP2×2、保护terrain统计、PPO/PD/mixed课程和预算；仅新增一个可关闭的task成本。

```text
v_cmd_local = raw_pre_action_task[:2] / 0.5
v_actual_local = R_heading(pre_root_quat)^T * (root_post_xy-root_pre_xy) / dt
r_v = -lambda * min(||v_actual_local-v_cmd_local||² / (||v_cmd_local||²+0.1²), 1)
r_new = r_old + 0.5*r_v
```

预登记lambda=1、speed_floor=.1m/s；.1来自既有idle诊断尺度，属于工程假设，不是作者参数或根据新模型试验调出的最优值。默认lambda=0必须直接走旧算式，不乘ones、不增加随机抽样或额外物理步。`v_cmd`用actor已有的**未经RMS归一化**输入，训练real/procedural和Diffusion部署均走同一raw task路径。它是+.5秒catch-up平均速度，不是瞬时轨迹导数；加速、转弯或参考不可达时可能有偏差，必须以命令分组诊断，而非把偏差藏在heading/clock变换内。

可证明：

- 精确匹配速度成本0；零请求且停住成本0，不用远goal强制伪造行走。
- 同D/同旧task其他分量时，非零请求下站住严格比匹配运动低分，补充了现有direction的平坦区；不只缩放已饱和的距离项。
- 旋转不变，约束forward和lateral的实际响应，不添加无依据的body yaw目标。
- 站住面对.106m/s请求的外层差额约-.264551；面对.5m/s请求约-.480769。
- 固定旧capture重放成本算术：idle新增外层均值-.238047，moving -.149256。**这是离线成本，不是重训后效果预测。**

限制：style仍可能补偿成本（最多.5）；例如站姿D=.3、移动D=.2，style优势 `4*log(.8/.7)=.5341` 已超过最大成本。该项不保证自然摆臂，也无法修复真正零请求、误导参考或穿障碍命令；不能宣布解决停滞。增加项后保留原D风格目标，并用冻结inclusive风格/foot/support检查是否出现急走、抖动或风格退化。

必要接线合同：在pre-action已有state/task处冻结root XY/heading/command，post-step且reset前算控制区间平均速度，所有fall/terminal样本保留；新增成本及请求/实际速度按同episode分箱。保存coefficient、floor、window、frame/timestamp/source身份，旧checkpoint缺字段仅legacy；显式新实验采用完整public warm start，禁止静默改旧checkpoint。4+4必须证明新cost配置与RNG/物理/课程恢复；与保护统计baseline相同2000预算/seed/256env，完整22、同四人200s、冻结c274参考，先不读34。验收同时看idle-with-moving-command、routes/falls、接触和inclusive速度条件风格，不因单一MMD下降晋升默认。

## 4. 全局专家软权重是否足够

[已封存prototype](../staged_command_expert_weights_v1/README.md)只按所有命令的边际亲和重权positive BCE。D输入仍只有history/action；同一个姿态与action在stop和move任务下得到同一D输出。等量stop/move请求下，两个匹配专家可获同权；站立actor无法收到属于自身moving请求的条件风格信号。已有37测试明确覆盖此反例。请求已弱时，重权还可能让站立更匹配；实际512样本run损失质量反而降低。故它可作为独立的风格分布实验，**不足以提供逐actor反停滞保证，当前不与速度成本同时更改**。

## 证据

[actual_CPU_review.json](actual_CPU_review.json)保存精确文件SHA、2000行reward身份、旧原始capture分组、full22旧对象及合成反事实；[可复现CPU脚本](analyze_reward_contract.py)不读取当前进行中的导航终态。[registration_proposal.json](registration_proposal.json)是尚未授权执行的实验身份，不是训练结果。
