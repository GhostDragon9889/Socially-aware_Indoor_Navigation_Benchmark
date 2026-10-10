# AMP 迁移诊断与显式状态判别器对照

## 已完成

新增 [amp_pacer_adaptation.yaml](/project/EXP/SangforAndSaut/ProtoMotions/configs/diffusion_amp/amp_pacer_adaptation.yaml)：`discriminator_input: state`、`amp_normalization: per_history_coordinate`、`warm_start_discriminator: true`。它严格转入公开 PACER 的 actor、critic、task encoder、sigma、状态判别器及全部配套统计；判别器输入为 1950 个状态特征，不消费由下一帧姿态反解的动作代理。**这是明确的实现对照／改进模式，区别于 NavIsaacLab Eq2 的状态加动作判别器，不声称作者实验复现。**

旧默认为 `state_action / per_frame / false`，保持 2019 维判别器、195 维归一化、12 个固定动作坐标与旧 checkpoint 结构。实际旧 corrected-power `step_1000.pt` 用新代码安全载入，模型、优化器和统计的 **101 个 tensor 逐一完全相等**；旧文件未修改，见 [legacy_checkpoint_compatibility.json](legacy_checkpoint_compatibility.json)。

修复两处明确缺陷：公开 warm-start 原来允许关闭观测／值归一化或改变 clip，造成预训练网络的输入／值单位失配；恢复原来遍历已有统计字段，漏掉一组时会静默保留新建的 identity 统计。现在分别拒绝不兼容 warm-start，及缺失／额外／非有限／负 variance/count／错误 shape/dtype 的统计。`TrainingConfig` 移入仅依赖标准库的 [settings.py](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/application/settings.py)，原 trainer 导入 API 保留，pre-Kit gate 可复用同一 validator。

CPU 门禁 **49 项通过，Ruff 通过**：[amp_adaptation_tests.xml](amp_adaptation_tests.xml)。其中真实公开 checkpoint 的 actor、归一化前 critic、D logits、逐历史坐标 mean/variance/count 数值对照通过；反复 inference 不改变统计。root 新 runner 的实际 AST 响应测试也通过：旧状态动作模式保持 pre-step `h[t]+a[t]`；状态模式在 physics 后、reset 前读取 `h[t+1]`，同时将该 history 写入 D 的训练批次。CPU 响应源测试不等于真实物理仿真。

## 论文与公开实现的差别

| 合同 | 原始 AMP | PACER 公开实现 | 旧本地重建 | 新显式对照 |
| --- | --- | --- | --- | --- |
| expert 输入 | 观测状态转移，无 demonstrator action | 10 帧状态，1950 维 | 10 帧状态加 69 维 PD 代理动作 | 10 帧状态，1950 维 |
| D 目标／style reward | least squares；有界二次 reward | BCE；负 log 概率 reward | BCE；等价 softplus reward | 保留 PACER BCE／softplus |
| AMP RMS | 明确特征规范化 | 每个历史坐标独立 | 195 维跨历史帧共享统计 | 1950 维历史坐标统计 |
| D 初始化 | 新训练 | 公开完整 checkpoint 可用 | 新随机动作判别器 | 严格转入公开 D＋RMS |
| reward 取时刻 | 状态转移包含动作后的结果 | post-physics AMP history | pre-action 状态加当前动作 | post-physics、reset 前 |

原始 AMP 对 motion clips 中缺少动作的问题采用状态转移判别器，并用 replay 稳定 D；其 least-squares 目标不能与 BCE 混称同一训练公式。[AMP 论文 §5／§6](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)

PACER 论文写出状态加动作记号，并有 motion-symmetry loss；其公开 no-shape checkpoint 对应的代码实际只输入 AMP 状态 history。论文设置还包括地形／社会／体型条件，不能从单一 no-shape checkpoint 推断已覆盖所有变体。[Trace and Pace §3.2／Appendix B](https://arxiv.org/html/2304.01893v1)

固定官方源码 revision `baeb6241f03fc9d78370a64e3c2dcef9d830d97f`：本地 [amp_models.py](/project/EXP/SangforAndSaut/ProtoMotions/third_party/pacer/pacer/learning/amp_models.py:26)、[amp_continuous.py](/project/EXP/SangforAndSaut/ProtoMotions/third_party/pacer/pacer/learning/amp_continuous.py:536)、[no-shape config](/project/EXP/SangforAndSaut/ProtoMotions/third_party/pacer/pacer/data/cfg/pacer_no_shape.yaml:30)、[训练 config](/project/EXP/SangforAndSaut/ProtoMotions/third_party/pacer/pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml:115)，来源为 [官方 PACER 仓库](https://github.com/nv-tlabs/pacer)。

## 当前数据说明与归因边界

读取现有 corrected1000 指标：2,048,000 次训练交互，83 次 fall reset、6750 次 timeout；最终 D expert/policy 概率约 .865/.116，AMP reward .255、forward velocity .636m/s、backwards fraction .0962。该分离值得做动作输入消融，但不能据此证明 D 只利用了动作 shortcut。

公开 checkpoint 的 observation RMS count 为 61,931,520,001，value RMS count 为 6,881,280,001，AMP RMS count 为 15,482,880,001。公开 value mean=50.0823、variance=153.0001；当前训练首／末 return mean=69.0532/69.2563，value loss=1.0566→.00766。未观察到值学习的数值发散；计数巨大意味着新样本对旧统计的权重很小，不代表已证实训练错误。新对照保留原计数，不隐藏 reset；如要 actor-only／重设统计先验，必须另建明确 stage。当前 trainer 反归一化值不做公开 player 的 ±5 normalized-value clip，因此仅宣称 critic 网络／统计完整转入，不将所有值推理边界声称完全相同。

官方 PD 生成链的 stiffness/damping、armature=.02、统一 motor gear500 与当前 XML 一致；`GAINS` 第四项没有用于该 actuator 生成，不能据其 150/250 数值猜测 force-limit 不匹配。明确不同的是旧 asset angular damping=.01／max angular velocity100 与当前0／1000、旧 Gym 和现代 PhysX 的时间与 solver 参数；影响需真实测量。此前有限地面直线诊断的跌倒被 root 定位在地面外，不作为 AMP 自身不稳定证据。

## 下一步对照

先在足够大地面与居中起点运行同 seed、同命令速度的公开 actor／corrected actor 控制诊断，再用同一干净 planner 比较旧动作 D 与新状态 D。新模式的长期训练与共享障碍效果由 root 实测，本文不提前判定改善。

保留奖励、物理、数据和 planner 固定以便归因；`backwards_penalty` 与 signed→absolute power 应另做显式对照，不能静默改变旧论文式奖励。公开 PACER 的 replay、整关节 dropout、mirror-symmetry、地形课程、接触终止及作者 motion manifest 未在本次状态 D 对照中补齐，均仍是后续可验证项。
