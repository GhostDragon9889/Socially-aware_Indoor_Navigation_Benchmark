# 论文 motion / navigation 指标与本地协议映射

只读源码与 cached primary paper；未运行指标、Torch/NN/SDK/GPU，未读任何 sealed34 或新 capture，未改生产185、默认或旧 frozen 产物。论文来源为 [NavIsaacLab IV-C / Tables II–III](https://arxiv.org/html/2606.26265v1)，本地缓存 `docs/full_audit_20261001/navisaaclab_paper.txt:475–506`；来源 bytes/SHA 见 `source_bindings.json`。

## 已知的对应关系与未披露部分

| 项目 | 论文明确披露 | 本地22动作协议 | 仍不能证明同口径的部分 |
|---|---|---|---|
| Position | 闭环模拟 rigid-body 与参考序列逐帧同步；报告均值0.218m | 24个 SMPL body origin，世界XYZ欧氏距离；每帧均匀平均24body，再按全部实际 metric frames 平均 | 作者 body集合、body origin/COM、坐标对齐、clip/frame权重、是否只取成功或全部clips均未披露 |
| Rotation | rigid-body rotation mean0.343rad | 单位XYZW quaternion、`2*acos(abs(dot))` 的SO(3)测地角；24body/frame平均，再frame-weighted | 作者角度公式、body集合、heading/local/world定义和聚合权重未披露；不能证明0.343与本地world角严格相同 |
| Tracking failure | 任一步 mean rigid-body position error **>0.5m** 即整段失败 | 严格`>`，失败latched；成功仅给eligible、full requested且完整实际完成的clip；跌倒单独报告 | 成功规则的已披露阈值逻辑吻合；作者具体评估clips/分割/最短长度/起始history/reset/中断策略未知 |
| Population | 2,848 AMASS clips用于训练参考style；报告评估locomotion clips成功64.6% | 固定parent-disjoint validation22，逐clip一次initial reset，frame9之后到真实source末帧；全部失败clip/帧保留 | 论文没有明确64.6%的分母、clip ID或是否为held-out；不能把2,848训练数当其评估分母。22的成功率每clip变动4.545个百分点 |
| Maximum | 报告 maximum rigid-body position error0.433m | 每clip已存`maximum_step_mean_body_position_error_m`；当前aggregate只报告frame-weighted means，没有统一maximum字段 | 作者“maximum”的axis、reduction及population未知；本地step mean maximum不等于individual body/frame maximum或maximum clip-mean |

本地明确的reset/phase合同在 `scripts/benchmark_paper_motion_tracking.py:241–350,383–390`：一次source pose/速度初始化，加常量初始translation，initial frame不入指标；没有逐帧yaw/pelvis最佳对齐，没有fall/time-limit reset，后续仍保留失败帧；actor只收到参考未来XY。论文只说明逐帧同步，不能由此确认作者的初始化、输入或frame排除完全一致。现有同帧root/heading/parent decomposition是描述性secondary；不能用它替换原world primary以声称rotation已达到论文数值。

## “0.433 maximum”需要明确scope，不能据此改失败规则

这是**从论文文字和非负误差推导的歧义**，不是作者实现bug结论。若0.433是同一评估population中所有body/frame误差的全局max，或所有per-step body-mean的全局max，那么每步body-mean都≤0.433，不会触发>0.5的失败，不能同时产生64.6%成功。它也可能指最大clip-mean、成功subset或另一个统计范围；论文没有说明，不能任选一种重现。

因此应保留原threshold、全22和全部失败帧，并在目标对照中将作者 maximum reduction 标为unknown。不要用0.433截断、挑样本、改success denominator或拟合对齐。

## Fixture / native 与 Table II 的明确缺口

| 论文导航评估 | 现有 fixture/native记录 | 缺口与正确报告 |
|---|---|---|
| 8 scenes、每scene3–15人、200s | 已有单fixture4人和部分本地Warehouse4/15，完整6000control/200s | 时长可对应，场景人口/布局ensemble没有覆盖同一作者population；历史原source与current-source对照需各自说明 |
| Success rate | `rollout_metrics.success_rate=None`；route vertices/cycles是循环路径进展，reset不算arrival | 循环arrival没有一次query的总分母、deadline/成功/失败/右删失定义，不能将vertices/cycles当88.67%成功；当前保持None是正确的 |
| Fall rate | pelvis-ground/up-Z阈值、fall frames、连续event、episode受影响比例、runner fall resets均可记录 | 作者fall判据和rate分母未披露；frame fraction、episode fraction、resets/actor-time不等价，必须保留字段的真实单位/分母 |
| Map/Agent collision rates | PhysX normal force、严格声明阈值、4substeps最大body-pair力、floor/self排除；actor-frame和unordered-pair-frame分别统计 | 作者collision判据/分母未知；不可将contact incidences、actor-frame百分比或pair-frame百分比静默映射为Table II Col. |
| Path Error0.153m | 当前`plan_tracking`对同时间物理root与明确target，提供mean Euclidean/RMSE/max；target可为causal speed/accel适配后的controller reference | 作者path error的point-time/最近path/global route/desired DP含义未定义；必须标明本地target身份，不能把controller-reference tracking与global path距离混称 |

对应源码：`CrowdSim/humanoid/evaluation.py:334–442,494–504,513–532`；`scripts/train_paper_amp.py:273–277,817–826`。本地primary contact定义已明确`author_metric_equivalence=False`，不需要新碰撞数学。fall/contact/frame/idle/reset全保留；native启动条件继续使用既有完整物理零门，不从本文添加MMD或新paper分数gate。

## 统计scope已实现：单次实际CPU执行

[薄汇总入口](summarize_tracking_scopes.py)只读取已封存的22条validation结果JSON，不读取capture、不加载模型。实际退出0，工具块`a05881`；[闭合记录](saved_scopes_actual_CPU_receipt.json)、[完整scope报告](state_only_matched_q_saved_scopes_actual.json)均已只读封存。

状态判别器试验保留原primary位置 **.21896992m**／旋转 **.75101782rad**；由全部22条sequence scalars所得clip等权secondary为 **.23121842m**／**.85676704rad**。全部11,982个metric帧的step-body-mean最大值 **.68853303m**，最大clip时间均值 **.39762771m**，四条失败均保留。individual-body/frame全局最大值不能从这些摘要推出，保持unknown。不同reduction产生不同数字，因此任何一个都不能直接替代论文未声明scope的“.433最大”。

## 其余最小报告改进

优先只增加一个metadata/summary小节，不动现有数学、actor或capture流程：

1. **Tracking统计scope卡片**：body names/24、world Euclidean与SO3、30Hz、frame9初始化排除、constant translation、全失败帧、fullclip denominator、clip数/metric frames。原frame-weighted两均值保持primary。
2. **从现有sequence JSON增加三项secondary reductions**：`clip_equal_mean_position_error_m`、`clip_equal_mean_rotation_error_rad`、`maximum_over_clips_step_mean_position_error_m=max(sequence.maximum_step_mean...)`。配明确axis/popu­lation标签；并列作者maximum定义unknown。它们只用已保存的sequence scalars，不重新计算capture、不加载模型，不声称作者采用clip均值。
3. **导航表使用原字段和真实分母**：success保留None及“循环route未定义query denominator”；并列vertices/cycles、fall事件/actor-frame、static actor-frame和peer unordered-pair-frame；path栏明确mean/RMSE与target选择（controller reference / raw DP / global path中实际保存的那个）。作者rate/path定义缺失列为unknown，保留200s结束的censored段。

以上改进可以解释“成功数提高但frame-weighted误差未同向”“局部path tracking好但route进展变少”等已有现象；它不证明训练机制有bug、不承诺未来效果，也不使本地22或fixture成为strict author复现。无需新NN或扩大到sealed34。
