# 未合入候选：按真实导航请求软加权专家正样本

**docs-only 数学prototype与CPU验收，不是已部署功能或训练效果。** 生产181、隔离训练931、v7数据、D2019/PD代理动作/PPO及正在执行的地形统计保护full2000完全不改。该候选没有有效生产CLI/profile，不自动启用，也未启动训练。

## 能力边界

这仍是 **D(history, action) 的无任务条件判别器**。只把当前batch全部请求汇总成专家正样本的边际损失权重；不在D输入添加context、不逐actor生成条件风格、不改变参考抽样。若batch同时等量要求站立和前进，对各自匹配的两个专家会给相同权重——测试明确覆盖此反例。它无法识别“同一姿态在actor A合适、actor B不合适”，也不是CALM、条件D或完整CMP。

如果上游近时域参考本来很弱，这个方案仍可能认为站立匹配请求；它无法替代导航规划/远目标进展要求。**目前只是一个较小、可分离的实验方向，不承诺改善停留或摆臂。**

## 1. 实际字段与两种运动量

`command_context(raw_task[B,1044], capture_phase='pre_action', observation_space='raw_task_1044') -> [B,3]`：

- 仅用未经RMS归一化的 `task[:20]`，即当前root位置为原点、当前body heading局部坐标系下的10个未来XY（+.5,+1,…,+5s）。取首+.5秒点除以.5作为要求的forward/lateral **catch-up速度**，不是已停留root的实际速度，也不是Diffusion生成器内speed字段。
- 第一段速度 `v0=p(.5)/.5`，第二段 `v1=(p(1)-p(.5))/.5`。路径转率为两段方向的shortest wrapped差除以.5；任一段速度<.12m/s时置0，防止近零点方向噪声。
- XY未来点不提供body orientation目标。因此第三维只能叫 **path turn rate**，不能假称root/head的期望yaw rate。prototype的phase标签是调用者合同；仅一个字符串不能认证不诚实调用者的时刻。

`expert_context(root_xy[E,11,2], root_xyzw[E,11,4], motion_ids[E,11], local_frames[E,11], clip_counts[E], history_episode_ids=optional) -> (descriptor[E,3], body_yaw_diagnostic[E])`：

- 读取真实v7同clip连续 `t-9,…,t,t+1`，与D的newest-first history10及next-frame PD proxy完全对应；明确重排只是几何计算，不改变D历史顺序。
- 平移速度来自10个实际间隔总位移/.3333秒，并旋到t时刻的真实root heading。path turn来自前后各5个间隔的位移速度方向差，除以两个段中心的间距5/30秒。
- **body yaw** 另由真实quaternion的t→t+1 wrapped差/dt返回，仅供诊断，不进入当前weight。这样不会把轨迹转弯当作身体朝向转弯。
- 硬拒绝非连续帧、float IDs、越界、跨motion/episode、非unit/非finite XYZW、退化heading或计算overflow。所有window由真实source局部帧边界约束，不能靠clip尾复制/跨reset拼接制造专家。

命令用未来.5秒，专家用近时.333秒，两种窗口不同且可能有加速/转弯偏差；这是候选描述符近似，必须保留该局限。尚未把参数称为论文参数或重新解释wholeclip turn/walk标签。

## 2. 权重数学：固定抽样，只改变positive BCE

设已采样专家descriptor为`e_i`，真实pre-action batch请求为`c_j`。候选核尺度 `s=(.5m/s,.35m/s,1rad/s)`：

```text
A_i = mean_j exp(-0.5 * sum_k ((e_ik-c_jk)/s_k)^2)
raw_i = 0.5 + 1.5*A_i
w_i = clip(lambda*raw_i, 0.5, 2.0), 选择lambda使mean(w)=1
```

参数仅是显式prototype假设，尚未调参、没有实际效果选择。64次float64二分同时满足上下界与mean1，输出与输入FP32/64一致；不能先clip再除mean（那会越界）。helper仅接受已声明raw范围[.5,2]，并末端验证有限性/bounds/mean；peer发现的subnormal输入bracket overflow已以最小范围拒绝修复，并有真实回归。

**最终权重不是新抽样概率。** 输入仍由原uniform clip→uniform合法帧抽样产生；`w/sum(w)`只是这个已抽batch中positive损失的相对质量。报告的ESS亦仅为batch loss权重ESS，不是全bank有效clip/人物数，不用于重要性`p/q`校正。

```text
L_class = 0.5 * [ mean_i(w_i * BCE(D(expert_i),1))
                  + mean_k(BCE(D(policy_live_or_replay_k),0)) ]
L_D = 原disc_coefficient * [ L_class + 原GP*原UNWEIGHTED_GP
                            + 原logit_reg + 原weight_decay ]
```

- 仅positive real/fake **binary** CE加权；不是新增五类style分类CE。
- negative live/replay混合、replay保留/随机索引/容量，gradient penalty（历史及动作坐标）、logit/L2正则、D更新次数与学习率均原样。
- AMP normalize/history shape1950与action69、12个冻结坐标、奖励softplus/scale/weight均原样；不放大action代理、不修改body回馈或方向项。
- 原`ei`随机选择/排列仍恰好执行一次。一般子batch选出后必须显式 `renormalize_bounded(w[ei])`，在该selection再保证bounds/mean1并记录；不能偷偷让positive整体尺度随selection平均权重改变。
- weights在`torch.no_grad`下产生；不让D/actor通过改command优化自己的权重。当前数学prototype不会保存/修改模型参数梯度。

## 3. Legacy、RNG和未来接线范围

`classification_loss(pos_logits, neg_logits, weights=None)` **直接使用原BCE算式及运算顺序**，不是“先生成ones再乘”的替代。实际FP32 loss及输入梯度逐位等于原计算。所有prototype函数无任何sampling/random draw；原expert IDs/帧/顺序/次数、共享corpus RNG和Torch randperm次数保持。

如果后续决定实施，需要单独注册新实验，并最小保存：版本/descriptor窗口/核尺度/bounds、基准profile与source SHA、v7 manifest/hash、mode、positive-only与unweightedGP合同、实际command来源字段、每次selection权重和支持统计。模块是stateless，**没有新RNG**；现有corpus/PPO/replay RNG完整保存恢复。enabled resume必须有匹配policy配置，缺失不能静默默认/reseed；旧checkpoint缺字段只能legacy。当前prototype没有实施这个集成或创建checkpoint。

未来最小接点是现有rollout保存的真实pre-action `task_obs`，取first20；原专家sample只执行一次，同时返回其真实motion/local-frame root几何metadata，不能为描述符另抽一次动作。预先计算batch marginal权重，D实际selection后显式mean-one投影，再仅替换positiveCE项。没有GPU版本、没有hook或profile autoactivation。根据相同调用序列保留RNG，不意味着改变训练后fall/reset时点仍完全一致。

## 4. 已实际执行的输入证明

[actual_CPU_proof_final.json](actual_CPU_proof_final.json)读取 selected baseline **真正的30次pre-action sample（4actor，共120请求）**，逐个NPZ SHA与raw task array SHA核验，并使用前次冻结的512个真实v7 legal窗口。不是post-step重建，不是当前保护full2000的训练分布。真实yaw/frame来自同bank root，不生成替代数据。

| CPU实物结果 | 值 |
|---|---:|
| 最终weight min / mean / max | .539997 / 1.0000000015 / 1.488818 |
| batch loss ESS | 485.17 / 512 |
| 请求在该sampledbatch中至少一个expert核亲和≥.1 | 120/120（诊断阈值，不是所有资源支持保证） |
| 原成熟D分类项 / weighted分类项 | .197979 / .196838 |
| unweighted GP | .023931684，逐位保持 |

positive损失质量中walking25.78%→26.70%、turn52.34%→53.43%、stand16.21%→15.61%、side5.08%→3.94%、fast.586%→.319%。这是**仅512样本权重**，并未重采样或移除17个run/427个stand；在这个低速请求分布中run权重反而降低。因此不能宣称“自动增强跑步/动作丰富度”，也不能据分类loss的小幅降低推断质量改善。

若当前sampledbatch没有合适专家，核亲和近零时mean-one权重会退化为ones；输出明确 `no_support_in_sampled_batch`，不造数据、不无限重抽稀有片段。实际120请求的支持结果不能推导整个2k训练课程或所有新导航指令都有支持。

37 CPU专项测试通过，覆盖真实方向/坐标变换、path-turn≠body yaw、非法时刻/空间、源/reset窗口、mean/bounds反例、subnormal bracket与计算overflow、unsupported batch、marginal能力反例、positive梯度解析值、legacy逐位loss/grad、CPU Python/NumPy/Torch RNG及input不变。初始测试的一处FP32 logits/FP64 weights fixture不匹配已纠正并保留失败日志；不是放宽dtype合同。Ruff通过。实际成熟D2019 strict加载、57有效动作/原proxy、GP不变、parameter.grad未填充、caller/corpus RNG不变、CUDA未初始化。

## 5. 预登记效果验收（尚未执行）

先完成当前保护统计full2000并冻结完整指标，才能决定是否比较本候选；保留所有现有2848、D2019、PPO/PD/replay/初始化/负方向项/课程和等预算，唯一新增变量是positive专家权重。完整22 validation、同四人200秒protocol、failed-inclusive c27490D与既有45D/速度bins不修改；不读34、不按新结果调参考/阈值。除fall/routes/idle外，单列按**请求速度和实际速度**分箱的腕部周期/左右协调和nativefoot支持；不能只用MMD下降或pilot高reward推广。

研究动机参考 [CMP原论文](https://arxiv.org/html/2608.03234v1) 对task context与motion兼容性的讨论，但CMP还包含学习的context relevance和条件residual。这个prototype只做固定核、global batch边际重权，**没有复现该完整方法**。[NavIsaacLab](https://arxiv.org/html/2606.26265v1) 提供history/action目标形式和AMASS动作来源，未提供当前kernel/weights，因此此候选属于明确的本地改进实验。

[数学prototype](prototype.py)、[37项测试](test_prototype.py)、[真实CPU读者](actual_CPU_proof.py)、[候选配置身份](candidate_recipe.json)。它们只位于docs新目录，生产和隔离训练均不会导入。
