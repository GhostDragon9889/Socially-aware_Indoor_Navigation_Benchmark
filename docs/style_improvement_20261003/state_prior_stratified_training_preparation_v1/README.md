# 下一轮：仅改变 AMP 专家采样先验

## 当前状态

这是待训练的本地工程实验。GPU 尚未启动，未来 checkpoint 的 SHA/bytes 仍为 null；CPU 准备通过不代表动作质量提高。生产 185 源码、已有 934/935 实验、默认配置和封存的 34 条 test 输出保持原状。

基线是完整评估的 `19f3341b…`、outer AMP weight 2、coupled public state prior、fresh 2000 的配置。新的 `.5` 模型 `09e802…` 在全 22 条 validation 上退化：成功数 17→15，平均位置误差 .209859→.251094 m，平均旋转误差 .810943→.891657 rad，inclusive90 MMD² .042092→.110080。该模型及 `.5` 配方不用于本轮初始化。

## 唯一数学变量

新 source bundle `output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1` 包含旧 `19f` 的 934 份源码逐字节副本，以及一份新绝对路径配置，共 935 份。旧 runner、模型、AMP/PPO、PD、镜像系数 50、历史 RMS、回放、任务奖励、protected terrain、public 初始化及 mixed command 配方完全相同；没有加入 `.5` 实验的两份 resume reward guard 文件。

`training.expert_sampling` 从 null 改为仓库已经实现的映射：

| 窗口类别 | 原 nominal uniform clip/center 质量 | 新目标质量 | 合法窗口 | parent 支持 |
| --- | ---: | ---: | ---: | ---: |
| low_speed | .238952 | .10 | 110,378 | 1,674 |
| fast | .015801 | .05 | 6,063 | 168 |
| sideways | .112666 | .10 | 62,693 | 1,412 |
| turning | .225018 | .25 | 114,472 | 1,441 |
| moving | .407564 | .50 | 201,186 | 1,765 |

类别内先按 dataset 的 parent 支持数平方根取样，再 uniform parent，再 uniform legal center；seed 7，dataset cap 为 null。源训练库仍是 2,848 条动作，validation 22、test 34 清单不变。窗口有 10 帧历史和 1 帧未来，中心范围为 `[9, frame_count-2]`。

类别由窗口内根部速度中位数、身体坐标下横向速度占比、约 .3 秒净航向转角计算。阈值依次为 .12 m/s、1.8 m/s、.55、.2 rad；冲突优先级为 low_speed→fast→sideways→turning→moving。这些是本地根部运动代理标签，无法等同于作者的语义风格标签或未公开动作清单。没有 p/q importance correction，因此本轮确实改变判别器的专家先验。

AMP 内部 scale 2、外层 weight 2（基础系数 4）、task weight .5 不变。公开 actor/critic/1950-history state prior 到 2019 state-action 的初始化、12 个冻结 hand/toe 坐标和 next-frame mocap PD target 代理动作均沿用基线；AMASS 不提供测量专家力矩。本轮是基于已实现采样方法的工程消融，不是作者公开超参数复现。

## 实际 CPU 证据

[真实训练库检查](actual_CPU_readiness.json) 的进程 `64779` 已观察到 exit 0，耗时 6.63 s。真实读取 train bank，重新得到全部 494,792 个合法中心；registry `a42c2455…`、概率 `427ea2d0…` 与既有 v7 验证相同。所有资源完整 SHA 由原 launcher 实际重算，实际 owned imports 来自新 bundle；没有反序列化神经网络权重、启动 Kit 或初始化 CUDA。

原 nominal uniform clip/center 离散窗口 ESS 为 **425,972.12**，新先验为 **73,696.15**，保留 **17.30%**。该量是 `1/Σq²`，不是实际训练 batch 的有效样本量或独立样本数。集中度明显增加，短窗口/少 parent bucket 会更频繁重复，不能据此承诺效果更好。

实际检查还验证了原 reset payload 完全相同、4096 个新专家窗口边界和 next-frame DOF 正确，以及恢复 private sampler 和 shared corpus 两个 RNG 后下一批专家/下一批 reset 完全相同。sampler 构造不消耗 global Torch RNG；每次专家取样仍调用一次原 legacy corpus sampler 并丢弃结果，以保持共享 reset RNG 在**相同调用日程**下相同。未来动作或摔倒改变 reset 日程时，跨模型随机序列可以自然分叉。

[源码与原 runner 分支检查](actual_source_and_resume_branch_CPU.json) 实际执行了原 runner 的四条采样配置检查语句：基线 null→新映射的 resume 被拒绝、新映射→null 被拒绝、相同新配置 resume 通过、fresh 新配置通过。新训练应 fresh 初始化；resume4to8 只能恢复本轮 pilot4。原模型 `.5` 的 reward guard 不移入旧源码，因此本轮不把该源码作为任意配方恢复器。

[CPU 真实闭合记录](actual_CPU_process_closed.json) 保留了首次 exit 1：准备脚本误把 2,848 写为三个 split 的总数，而实际是 train 2,848 加 validation 22、test 34。仅修正 docs reader 的计数断言，并保留首次 reader/导入日志。另一个 prepared plan 的 `CPU_watch_env` 原路径继承问题已在启动前修正，初始计划与精确修正记录保留。

## Root 的具体命令

[命令登记](root_GPU_launch_commands.json) 提供绝对路径 argv/env，以下只是索引，不由本准备脚本启动：

1. `commands.pilot4`：本轮独立 pilot 输出目录，fresh 4 updates。
2. `commands.resume4to8`：只恢复该 pilot 的 `step_4.pt`。
3. `commands.fullfresh2000`：独立完整输出目录，**无 `--resume`**，256 actors ×32 rollout ×2000 updates =16,384,000 transitions；初始化是与 19f 相同的 public PACER，而不是 19f/09e 已训练 checkpoint。
4. `watch_command`：原 watcher 源字节不变，独立 snapshots 目录，500/1000/2000/final2000。

完整训练继续使用 fixed-half AMASS/procedural、replan15、episode300、terrain192、spawn48、smooth speed2/accel1.5、synchronized replanning、projected joint-effort substep mean、physics-only、seed7。全部数值与基线完整计划相同，只有 bundle/profile/独立输出路径变化。

## 评价和待完成项

先完成独立 source/profile/实际 CPU 证据审查，再由 root 串行执行 pilot/resume/fresh2000。完整 payload reader 还需要只做 935 来源身份和现成 sampler state 的薄适配：保留原 optimizer/replay/RMS/RNG 数学审查，并核对 q 配置、registry、概率 fingerprint、private RNG 与采样计数；不把 prepared CPU 证据当作真实训练闭合。

训练真正完成后，评价固定全 22 条 validation/11,982 帧，保留每个失败，先验证同帧 reference/phase/initial state 与 19f 一致，再用原 inclusive90 和关节误差/固定 reference toe 诊断。fixture 使用原 4 actors ×6000 controls/200 秒、guide20/eta0，保存真实 contacts/falls/ground；actual terminal 与完整 fixture 零 fall/static/peer 才能进入 native4，native4 完整零物理失败后才能 native15。MMD 不筛选是否启动物理评价，封存 34 条 test 不参与优化。

仍待实际训练和评价后才能判断更集中的专家先验是否提高运动风格或导航。当前不修改默认、不推广候选，也不声称已达到或超越论文效果。
