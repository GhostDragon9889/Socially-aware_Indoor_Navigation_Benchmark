# AMP 动作风格：真实专家、判别器与任务分布审计

**只读 CPU 审计；未改变保护统计的 full2000 训练。** 结论是：目前最值得验证的是**专家运动与实际导航指令的相关性，以及判别器能否约束时序全身风格**。没有证据把停留主要归因于 action 代理、AMP clipping 或“判别器不看手臂”。当前四人200秒结果仍以 selected negative-guide20 为默认，新的训练完成与动作自然程度必须分别验收。

## 真实输入与范围

采用旧 state-only mixed2000、旧 state-action2000 和新保护地形统计的 sealed step4/step8，分别为2000/2000/4/8更新；前两者各16,384,000真实转换，pilot为32,768/65,536。用 `weights_only=True`、CPU mmap读取各真实 bundle，只构建 D 或 actor/task子网络，**没有恢复优化器、没有新训练、没有推进物理**。每个 D 使用同一组512个真实 v7专家窗口，以及该模型实际保存 replay 的512个等距位置；后者没有live/recency/课程分组身份，因此不冒充最近在线样本。

完整当前证据：[report_final.json](report_final.json)、[D探针](actual_cpu_probe_v2.json)、[专家朝向与actor探针](heading_actor_probe_v2.json)。初始工具/输出保留，初始reset行求和错误已纠正：训练metrics的reset字段在进程内累计，**不能逐行相加**，resume段计数也不能当作全部累计。最终报告只使用真实最后进程计数，不使用错误初始和数。

| checkpoint | 实际SHA256 | D输入 / 初始化 |
|---|---|---|
| state-only mixed2000 | `28aed874464379f0b0628719290d36af0a8b0c53691c0dd51f56102c5f428f72` | 1950；公开PACER D及AMP统计转移 |
| state-action2000 | `eeac037798d2244e0cd4f0a81b40f51020c2aa3320106c7f10620e8c2ddde526` | 2019；D随机，不转移公开D/AMP统计 |
| protected step4 | 见最终JSON，实际restricted读及完整SHA | 2019；同随机D路径，地形suffix保护 |
| protected step8 | `d3c7b193225c800f4f0d46ef37afa8b6bc998a6ee163362da0ff66807bc66c23` | 同上；真实4→8续训，非新8更新独立实验 |

旧state-only与state-action保存config有两个差别（D输入、公开D初始化），来源epoch还分别为138/isolated146，奖励使用历史时间边界也不同；**不是只改变一个 action 输入的消融**。新保护pilot与旧state-action仅增加统计模式字段，正在执行的完整试验应保持这一单变量。

## 1. 专家 action 是姿态代理，时间配对正确

来源931的 `amp_world.expert_batch` 先采样 newest-first `h[t],...,h[t-9]`，再把 **q[t+1]除以实际PD scale**，clamp至[-1,1]作为专家action。实际decoder的offset为0，三轴scale为`min(1.2*最大关节限幅,π)`，两膝y scale为5；12个Hand/Toe坐标为0。没有测量的专家控制、逆动力学、重力补偿或接触力。原标识`inverse_pd_target_from_mocap_pose`应理解为**目标位置归一化代理**，不是求解了动力学逆控制。

Runner的state-action分支确实在 `world.step` **之前**保存历史并计算D(history,本步effective action)；state-only在步进后使用post-step历史。没有证据支持“state-action把旧动作和下一步历史误配”的bug。10帧历史覆盖0.3秒，reset历史只来自同一动作合法边界；新统计策略没有改变这个合同。

同512真实窗口，next-pose代理与current-pose代理差的有效坐标RMS为 **0.0069511**。将真实AMASS self状态和源root未来0.5…5秒、零训练地形重建为离线actor输入后，旧state-action actor decoded目标与下一mocap姿态的差为 **0.15928rad RMS**（24个arm坐标0.15260rad）。这是**离线姿态/指令探针**，不是当次在线观察或真实tracking误差；2686/5120个未来查询使用已披露的clip末尾线性速度延拓，也没有把这些末尾当作新动作数据。

## 2. 目前 D 并不主要依赖 action 代理

| 同一个 D 的离线干预 | 旧 state-action2000 平均P(real) |
|---|---:|
| 实际保存 replay history +原action | .153347 |
| 相同history，action全部置0 | .153672 |
| 相同history，action在512个样本中反序打乱 | .153350 |
| 真实专家history +next-pose代理 | .808628 |
| 相同专家history +current-pose代理 | .808633 |
| 相同专家history +实际policy replay action | .808519 |
| 相同专家history +离线actor预测action | .808564 |

当前归一化输入的平方梯度占比：action仅 **0.1073%（replay）/0.1279%（expert）**；arm history占 **26.41%/34.69%**。梯度比例受坐标尺度影响，不是跨尺度的重要性定理；反序/置零是离分布反事实，也不能证明整个训练过程中action永远无用。但这些真实结果**不支持action-shortcut为此模型停留的主要成因**，更不支持直接放大action来加强风格。放大代理可能反而惩罚物理控制所需的target偏置。

AMP frame195包含root旋转/速度/角速、19关节旋转与速度，以及两脚踝/两腕相对位置；肩、肘、腕、胸锁运动都进入D。只排除terminal Hand/Toe的12个关节轴；并非丢失整个手臂。D未接收任务20维未来点、行走速度指令、style latent、真实脚接触或显式相位。当前0.3秒历史可以判断短时姿态，却不直接保证完整摆臂周期、左右协调或长时间风格覆盖。

## 3. AMP归一化、奖励尺度和 replay 的实际情况

D使用1950维逐历史坐标mean/variance；history normalize后clip±5，action仍是dimensionless [-1,1]。当前512样本里history超过5的比例仅：旧state-action replay **.1304%**、expert **.0970%**。因此此probe未显示全体AMP特征大规模饱和。AMP moments每次更新含8192个rollout和4096个expert，比例2:1；D分类batch是live/replay各2048与expert4096。**replay使用当前统计，但不加入统计更新。** 旧成熟两模型最后100更新live/replay分类概率接近，没有证据仅凭buffer存在就断言“replay导致idle”。

| 最后100更新（pilot使用全部已完成更新） | weighted AMP平均贡献 | 位置task平均贡献 | D专家 / policy概率 |
|---|---:|---:|---:|
| state-only2000 | .69155 | .49122 | .84779 / .15197 |
| state-action2000 | .66055 | .48993 | .80793 / .14587 |
| protected step4 | 2.60365 | .46137 | .48408 / .47684 |
| protected step8 | 2.54236 | .45746 | .48616 / .46867 |

实现为 `AMP=2*softplus(logit)`，总奖励再乘AMP weight2，即 **4*softplus(logit)**，上限36.8414；位置奖励的最大贡献是0.5，任务还含signed-power与方向项。成熟state-action weighted AMP/位置均值约1.35，初始pilot约5.56；这不是PPO advantage或梯度贡献的比例。pilot D接近随机.5时reward仍较大，所以**pilotreward高不能当风格更好**。与论文外层`r_amp+.5*r_task`相比，当前采用显式PACER变换和额外AMP权重假设，且所有比较模型同样使用此设置，不能用它单独解释两模型差别。原论文没有给出完整D输出映射与训练参数。[NavIsaacLab IV-B](https://arxiv.org/html/2606.26265v1)

## 4. 指令相关性是可验证的剩余瓶颈

这次固定512专家窗中393个root速度≥.12m/s；**47/393=11.96%**身体朝向前向速度<0，37/393=9.41%<- .12m/s。walk/turn整clip代理类中的移动窗口分别9.09%/8.51%后退，standing类也含运动窗口。这里的forward严格为root heading与真实root世界速度的点积；不是rootyaw朝向目标的误差，也不是全bank比例估计。当前任务负backward项与这些正专家窗口存在局部目标张力，但可能也对应合法转身/后退，不能据此删除所有此类数据或认定存在坐标错误。

同时D对真实expert的stationary/fast样本平均P分别 .83134/.74078（119/10个窗口，**fast样本很少**），目前无任务条件的D承认站立是真实动作。只要上游给出的近时域参考也很慢，task误差小和真实站立奖励可以同时成立。旧真实state-action导航actor0从约50.83–200s的执行参考速度均值只有.1059m/s、目标误差.0831m、5秒raw计划位移.2138m；这提示需要区分“controller拒绝有足够速度的命令”和“planner本就给弱命令”，不能把低速度全部归给D。

实际完整200秒对照仍是：selected **0F/21vertices/194.77 idle actor-s**；mixed state-only **11F/19/71.77**；state-action **2F/2/591.73**。state-action腕部heading-local速度仅约.074/.075 chain-unit/s，mixed约.296/.257，selected约.345/.353；但速度分布不同，低腕速不是同速摆臂缺失的单独证据。[既有failed-inclusive完整报告](../state_action_navigation_style_v1/README.md)

## 5. 真实2848数据可用，但类别与窗口不能等同

v7是2848个明确AMASS-derived segments、1853父源，不是作者2848原始ID。wholeclip代理类turn1568/walk720/stand427/side116/run17；KIT1233（43.29%）。已有实际coverage审核指出：fast短窗6063/494792=1.23%、low-speed22.31%；427 stand/look中180个文件含Push线索，多数并非经接触测量证明的安静站立。仅699段有CMU官方trial描述，其余仍主要靠filename，不能把数据集名称或kinematic turning类直接当语义风格。

当前uniform clip→uniform legal-frame意味着每个短clip和长clip总权相同；多段同父源也带来相关性。它不是均匀人物、均匀时间或每类20%。先前均匀五类式分层先验已经没有取得主验收提升，因此**不建议再次无条件提高17个run片段到20%**。现有真30Hz/child-joint速度、neutralFK与parent-heldout门禁已验证；本次没有把速度坐标问题重新臆测为当前原因。[既有覆盖审计](../amass_style_coverage_v7_read_only_v1.md)

## 6. 下一独立实验建议（没有实施、没有效果承诺）

首先完成目前的**地形suffix统计保护单变量**full2000。真实pre-action observer只验证过selected baseline：2061/122880个terrain坐标clip（1.677%），raw−10来自2m obstacle的高度差而非地下；旧saved pose地形重建仍是post-step诊断，不能冒充在线测量。保护pilot已真实保留public参考mean/variance及reference count61931520001，task全局warm count200000继续增长，但4/8更新不能证明idle或手臂改善。[真实pre-action记录](../pre_action_actual_observation_reader_v1/actual_cuboid30_v1/report.json)

**保护试验终态之后，首选一个有界的“指令相关专家”独立实验：**

1. 保留全部2848资源、uniform reset、fixed-half AMASS/procedural课程、PD/PPO/负方向项与完整预算；只改变D positive专家窗口的软相关权重。使用真实 **pre-action未来点所要求的**heading-local速度、侧向分量和转向作为context；不能用已经停留的实际root速度自我匹配站立。
2. 窗口类别依据真实history10+next边界和局部kinematics，而不是wholeclip turn/run标签。`q(window|command) ∝ p_uniform_clip_frame(window)*w`，建议首个实验预登记weight范围[.5,2]并归一化；不硬删stand/backward，不给稀缺bucket固定大概率，没有支持就报告unsupported。权重范围只是候选、不是论文参数复现。
3. 最小可审接口：`sample_experts(batch, command_context[B,3], CPU_generator, manifest/index_SHA)` 返回history、有效action、父源/帧/权重/期望概率；context来源、归一化及边界必须同policy侧可核。独立expert RNG与一次旧draw配对控制、完整checkpoint恢复沿已有明确合同；改变训练后的fall时点不可能声称仍保持所有reset逐位相同。
4. 在同预算前先验收短窗速度/转向支持、parent heldout、nonzero有限概率、同seed/resume、真实body/DOF order、history chronology；之后固定完整22 validation与同一200秒导航，失败帧不删。新增按**指令速度/实际速度分别分箱**的腕速、arm-swing幅度/频率、左右时滞和foot native支持诊断；旧c274 90D/45D主数据不重调。不能仅降低MMD就推广。

该建议基于“无任务条件先验可承认与指令无关的真实动作”的可测风险，而非已经证明它导致此idle。[CMP primary paper](https://arxiv.org/html/2608.03234v1)研究任务context与motion兼容性、参考软权重及residual prior；它支持此研究方向，但作者系统/指标/仿真器不同，不能宣称现有工程已实现CMP。检索到的是**CMP**，未核实同名“CMD”方法。

若此单变量仍没有改善同速arm周期，再分别考虑更长时域/显式style条件；不要一次混入动作缩放、新损失、课程重采样和新架构：

- [AMP原论文](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)因为mocap没有专家控制，采用state transitions；其LSGAN与[0,1] style reward不同于当前PACER BCE/softplus。可作为后续随机D初始化、统计相同的公平state-only对照，不能拿现有public-D vs随机-D比较直接当纯action消融。
- [CALM](https://arxiv.org/html/2305.02195v1)共同学习motion encoder、latent-conditioned policy/D来指导不同风格。当前模型没有该latent，无法靠加载既有权重或命名一个标签获得CALM控制。
- [SMP](https://arxiv.org/html/2512.03028v2)训练full-body motion diffusion作为style reward。现有XY轨迹Diffusion不是该身体先验；需要在本地真实AMASS上新训练body prior并验收，不存在把当前XY checkpoint直接当自然全身prior的合法捷径。

## 验证边界与复跑

四bundle完整SHA、shape/finite、57有效坐标/12冻结轴、各D严格权重加载；4×512 expert与replay输入梯度/反事实，另4×512离线actor输入探针全部CPU实际完成。Python/NumPy/Torch CPU调用者RNG逐次保持，model param.grad未填充，CUDA未初始化。没有新仿真、没有34 test读取，也没有改生产、931冻结业务、bank、权重、旧评分公式或训练参数。

[复跑脚本](audit.py)、[heading/actor脚本](heading_actor_probe.py)、[最终reader](finalize.py)。初始工具Ruff失败和源副本保留，最终只针对三个实际工具的Ruff gate通过；不是整个repo的质量测试。本报告为**诊断与下一实验计划**，不把机制假设或pilot称为动作效果改善。
