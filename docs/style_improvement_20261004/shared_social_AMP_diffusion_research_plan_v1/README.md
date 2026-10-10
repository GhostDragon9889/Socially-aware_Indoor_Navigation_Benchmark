# SharedSocialAMP + Diffusion：下一轮三个优先实验

2026-10-04，**可修改研究计划，尚未实现或训练**。仅核对论文、源码与已有小型结果记录；未导入 SDK/Torch/NN，未打开权重、AMASS/参考 bank、NPZ 或 Test34 数据。原封存父源和默认模型保持原身份。共享入口的生命周期运行门目前仍为 False；执行安排由 ROOT 在真实进程退出检查通过后串行进行。

## 已有证据决定优先级

同 AMP62、同四人 200s/6000 controls 的 OLD63e 与正式 EMA40000 对照已封存，不能把多个训练 recipe 的变化归因于某个参数：

| 完整已有记录 | OLD63e | EMA40000 |
| --- | ---: | ---: |
| 90D speed-bin macro MMD²（共同 5 bins） | .07238060 | .05592538 |
| 45D pose MMD² | .02015677 | .02297688 |
| K50 / K100 coverage | .86 / .72 | .86 / .71 |
| supported foot-origin 切向速度代理均值，m/s | .07313597 | .08532028 |
| fall / static actor-controls / peer pair-controls | 0 / 0 / 0 | 1 / 0 / 45 |

来源：[封存对照](../formal_EMA40000_style_historical_comparison_v1/README.md)。近地 toe 速度代理也增大；但上述足速度不是 contact-point slip。90D 中 [.1,.5) 与 [1,1.5) 两个 bin 退化；[2,3) 和 [3,+inf) 的参考支持不足，保持 unknown/null。这些事实支持以下优化假设，尚不能证明退化原因。

原论文使用导航动作子集和分层 Diffusion/AMP；固定 AMASS 数量不等于恢复作者动作 ID。Nav 的任务奖励公开为 trajectory、energy、direction 三项，未公开名为 `social_collision` 的额外行人奖励。[NavIsaacLab IV-B/IV-C](https://arxiv.org/html/2606.26265v1#S4.SS2)

## 1. 让低层学到真实同行人避让，再接闭环 value guidance

**优先理由：**现有 AMP62 的 1044D task 只有轨迹/静态高度；新的 12308D SocialCNN、共享碰撞 World 已有源码，但没有已完成的共享训练权重。高层邻居条件不能补出低层当前同行人反馈。PACER 将同行人写入高度/速度栅格，并通过碰撞终止学习避让；TRACE/PACER 也提出使用已训练 critic 引导高层轨迹。[作者项目](https://research.nvidia.com/labs/toronto-ai/trace-pace/)、[固定 PACER 源](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/tasks/humanoid_pedestrain_terrain.py#L711)

**可实施顺序：**

1. S0：先完成现有 SocialAMP fresh 基线，固定 neutral SMPL、69 输出/57 有效动作、state_action2019、BCE/PPO、mirror50、120Hz 物理/4子步/30Hz 控制、mixed 两 AMASS+两 procedural、uniform expert、同预算。2000 updates 只是已登记本地 pilot，不能称作者完整预算。
2. S1：只增加一个独立 procedural encounter provider，覆盖迎面、交叉、超越；保持 AMASS expert/物理 reset 采样，保留原 5s/51 knots 时序。记录实际 encounter 暴露、计划/root 速度、最小间距、接触、跌倒、终止及 collateral reset，不将没有交互的 frames 删除。先验证此变量的作用，再决定是否增加 soft clearance cost。
3. 后续单独分支才添加 `social_clearance_cost` 或 measured `social_collision_cost`。两者分别是几何代理和真实 pair-force 成本；明确公式/系数/分母，不把全身体净法向力或仅 >50N 终止标志命名成同行人碰撞。它是本地优化，不冒称 Nav 原式3。既有 >1N pair 接触指标与 >50N 终止不同，净力相消也可能掩盖接触。
4. 完成训练后，固定 Diffusion40k，以新 critic 的当前 self+raster 评估候选 local20；先核对冻结 RMS、反标准化 value、成本符号/尺度/梯度，再做独立 alpha0→非零消融。选/引导**实际将执行的适配后 reference**，raw guidance clearance 不能代替 causal adapter 后的 clearance。更多候选不是自动安全保证。

**源码挂接点 / 当前缺口：**

- `shared_group_world_owned_lifetime_revision_v2/shared_group_world.py:353,406,445`：当前社交观察及四子步；contact 仅控制末态净法向力终止，缺少逐子步/逐 peer 的训练暴露记录。
- `social_world_lifetime_dependency_migration_v1/social_rollout_collector.py:291`：reward 仍只合并原三项；额外 cost 应在**新 collector 分支**单独保存/合并。
- `CrowdSim/humanoid/training/mixed_trajectories.py:184`：当前 procedural 命令独立生成，没有专门 encounter 配对/难度账本。
- `social_value_adapter_staging_v1/value_adapter.py:169,187,200`：已有可微冻结 critic ABI；未获得已训练共享 critic，未接入实际 Diffusion guidance。
- `CrowdSim/humanoid/training/reference.py:92`：适配器没有地图/peer 参数；不得将它的输出说成碰撞安全。

## 2. 固定 2848 内做短窗口风格/速度分层，补足中速和转向反馈

**优先理由：**改善 macro MMD 可能来自速度 mixture 改变；现有两个支持充分的速度 bin 仍退化。clip-uniform 不是 window-uniform，也不能保证转向/侧移/长尾动作被充分训练。分层需要同时看动作样本暴露和真实控制响应，而非只提高运动速度。

**最小实现：**新实验入口允许已有 `ExpertReferenceSampler`，显式传入 `SocialAMASSExpert(..., expert_sampler=..., sampling_config=..., sampling_provenance=...)`。仅从原 train2848 生成 legal history10+future1 窗口；五类 low_speed/fast/sideways/turning/moving 先按 train 支持登记权重。保留其**每批一次原 legacy draw 消耗共享 corpus RNG，再以独立 CPU generator 替换 expert 窗口**的控制设计，避免无意改变 reset 随机流。显式记录 q 改变 expert prior；当前代码没有 p/q importance correction，不得声称目标分布仍是原 p。

五类训练 proxy 的 .12/1.8m/s 阈值不等于评估 speed bins，也不是 BABEL 真正动作语义。若要专门补 [.1,.5) 与 [1,1.5)，下一版本增加**train-only**中速窗口子分层，并更新 registry/probability/schema/source 身份；不能修改既有评估 bins 或用 Test34 选权重。先固定 q 配方，再评估；不按 MMD 挑帧、删未知或重拟合 reference clusters。

**源码 / 缺口：**

- `CrowdSim/humanoid/training/expert_sampling.py:66,206,492,518`：已有五类窗口、层级采样、独立 RNG/legacy draw；尚无上述两个精确中速子 bin。
- `social_world_lifetime_dependency_migration_v1/expert_batch.py:81,129`：已有可选 q admission。
- `shared_social_owned_lifetime_entry_revision_v3/train_shared_social.py:122,271`：CLI 明确拒绝 q，实际只构造 uniform expert；需新分支接线，不能只改 JSON。
- 新 ledger 保存 batch/window/stratum/dataset/parent/segment、实际 q、expert/live/replay 暴露及 sampler RNG sidecar；checkpoint 的 `runner_state=None` 仍不表示物理可 resume。

验证保持原 transition/expert 预算、initialization recipe、PPO/D/mirror 和 planner；报告同相位 root position/velocity、低速 undefined heading、转向/侧移 bin、pose45+coverage、完整 fall/peer/static。分层采样改善的是一个待检验假设。

## 3. 先接真实脚部支撑与时序，再做接触感知动作优化

**优先理由：**已有足部代理退化，同时 pose45 变差。不能把 body-origin 运动或 ankle 命名成 heel，更不能通过让脚不动获得“低滑移”。现有 D195 排除 toe/hand body 旋转，动作 mask 冻结 toe/hand；这些既有适配不是本轮应隐式解除的设置。

**最小实现：**独立评估/训练记录适配器复用 `FootSupportMonitor` 的四脚 Ankle/Toe-to-floor 法向力，接入每个实际已完成 120Hz 子步、reset 前完整归档。记录左右支撑相、double/no support、持续支撑中的足速度、root target 跟踪及真实接触；把 unsupported/airborne、terminated、reset 首帧及 unknown 留在总分母。对速度/转向分层报告，防止静止比例或 support 分母改变造成虚假改善。

之后才另写 contact-point 观测：同一物理子步的点/法向、body 线/角速度与坐标基准、地面运动速度。按刚体点速度减地面速度后取切向分量；仅在真实支持上解释 slip。缺这些通道继续标 unknown，不能由 normal force 推出 friction/impulse/COP。验证测量后，另一个单变量分支尝试有界 support-slip cost；保存原 reward 和新增 cost、权重，防止惩罚 swing、自然 foot rolling 或鼓励僵硬/跳跃。系数不能由当前代理均值推导。

**源码 / 缺口：**

- `CrowdSim/humanoid/foot_support.py:114,152,265,309`：已有真实 floor support 和 body-origin 切向速度代理；没有实际接触点 slip。
- `scripts/probe_paper_foot_support.py:272`：已有独立采集与 true COM 对照；不是 SharedGroupWorld 的逐子步训练 hook。
- `shared_group_world_owned_lifetime_revision_v2/shared_group_world.py:406`、`social_rollout_collector.py:298`：需要新独立 hook/归档分支，保存与 actions/next-value 同期数据。
- `social_training_settings_staging_v1/social_training_settings.py:30`：57 mask 保持；本分支不把 toe/hand 解锁、PD 或 AMP loss 改动混入脚接触优化。

新增足部 native views 必须有**独立明确的拥有/释放 receipt**，不能仍用原 N 个同行人 views 的弱引用数宣称所有 views 已释放；本计划不修改 ROOT 正在诊断的生命周期源。

## 共同评价与结果解释

- train 固定 2848 segments / 1853 parents；manifest SHA `68d41be2…`、child_joint velocity 合同不变。validation22 / Test34 parent-disjoint 不变；本轮没有打开 Test34。新 q、encounter/cost 均是可审阅独立分支，不改默认。
- 开发比较使用完整 validation22 与固定同源四人200s，原所有 failure/reset/idle、控制/子步分母及来源保留。现有本地 native 扩展门仍要求 fall0/static0/peer0；没有因风格分数改善降低此门。
- 每次只改变一个变量：S0共享基线→S1 encounter；q 与 support-cost 各同基线独立对照，不将三项一次混合。新控制器固定同一个 planner 后比较；SFM+AMP 对照保持相同新 AMP，沿 Nav 的高/低层分离原则。
- 45D/90D 原 body、heading/scale、reference selection、bandwidth、caps 和 K50/K100 centers 保持；macro/每 bin/总体 mixture、有效+unknown 数量并报。最高支持 bin 的历史 balanced counts 不同，不能称完全同样本数。
- 本地 MMD²、coverage 或更低脚速度不能直接对比论文 Table III：作者完整参考/聚类、scale、核和聚合尚未恢复。只有本地相同协议下的改善证据，不能由 .02 < .327 宣布超越作者。

AMP 原论文使用 LSGAN 和有界奖励；固定 PACER 源使用 BCE/负 log 概率。当前 SocialAMP 保留后者；LSGAN 切换是另一个算法实验，不是本轮兼容性修复。[AMP §5.2](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)、[固定 PACER 判别器](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/learning/amp_continuous.py)

[TRACE 官方实现](https://github.com/nv-tlabs/trace)区分 clean guidance 与 filter-only sampling。这里的训练/guide 都必须记录实际执行 source/config/weights 和真实进程结果；原论文/仓库的展示不证明本分支已实现同等效果。
