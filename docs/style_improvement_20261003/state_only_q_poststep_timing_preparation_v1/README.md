# 单变量：state-only AMP 奖励时间

2026-10-03。**只完成独立源码准备和一次有界 CPU timing 验证，未构建新隔离 bundle、加载模型或运行 GPU/SDK/训练。** 当前生产185、q936历史、默认权重和所有数据保持原样。

## 直接证据与待检验问题

q936 的[完整 fixture 结果](../state_only_matched_q_fixture_metrics_actual_v1/summary.json)保留200s/6000controls全部帧：3跌倒、静态/peer接触0、路线31，90D/45D分别 .06529463/.03235954。q935 state-action 稳定结果作为上下文；本项不重算任何指标。

q936为了匹配q935动作输入消融，特意让1950维state D使用pre-step历史。此状态不含当步动作执行后的结果；动作仍通过后续奖励/价值影响学习，不能称没有学习信号。[Nav IV-B式2](https://arxiv.org/html/2606.26265v1#S4.SS2)写历史+action；本项是公开AMP/PACER状态判别器时间实验，不冒称严格Nav式2或作者私有参数。[AMP §5.1/式5](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)使用状态转移，说明mocap控制动作未知。固定[PACER agent源码](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/learning/amp_continuous.py#L123)先env_step，再保存infos的amp_obs，后计算style reward；[既有逐commit bytes绑定](../paper_fidelity_research_v1/pacer_commit_binding.json)为来源依据。

**假设**：去除动作输入后，style reward及时观察当步动作后果可能改善物理稳定或动作风格。当前没有证明它导致3跌倒，也没有保证降低旋转误差。

## 最小改动

[runner三片段patch](runner_three_fragments.patch)仅恢复原有state分支：

1. 正确记录post-step timestamp。
2. 动作执行前只给state-action分支计算D奖励。
3. `world.step`后、reset前clone新的history/history_episode，再给state-only分支计算D奖励。

同一history进入原transition dictionary、D更新和policy replay。terminal状态保留后才reset；policy/value/self/task observation仍是原时间关系。[逆向来源证明](source_derivation.json)确认3片段逆转后全文bytes/AST恢复q936；新runner恰与原q935 runner全文相同，**不导入更多当前185变化**。PD目标/动作/专家state-action代理、网络、q sampler、PPO/GAE/Adam/replay数学、RMS、奖励内2/外2与task .5、terrain、physics、mixed commands、seed7及fresh2000×256×32均不改。profile仅改标签和解释文字。

[源计划](source_plan.json)固定q936父源和全部来源。必须fresh公共actor/critic/state-D/RMS初始化；1950输入形状不能区分pre/post时间，不能resume旧q936，不能接q935或新Diffusion权重作为本次机制对照。相同seed不保证成对随机轨迹或物理bitwise一致。

## 已实际验证与执行边界

[CPU checker](check_timing_CPU.py)执行实际runner所选AST语句、完整原transition dictionary及原trainer的replay.add调用，用stdlib spy对象代替张量/物理/模型。实际tool `55af30` exit0，[报告](actual_timing_CPU.json)/[闭合记录](actual_timing_CPU_closedreceipt.json)：

- post-state非终止与终止：reward/rollout/history_episode/replay均取同一动作后history，reset不会改已clone记录。
- 原q936 pre-state对照和state-action分支均仍取动作前history。
- pure evaluation不调用AMP reward。
- 内存删除post-step history赋值，必要negative确实失败。

这验证源码时间和handoff，不是张量、真实D/PPO/replay ringbuffer、模型或物理证明；没有重跑旧suite、指标、语料或sealed34。

[未来命令源计划](future_commands_source_plan.json)仍**不可启动**：新bundle/sourcecount/source manifest绑定未生成。后续root若执行，保持原fresh4与独立fresh2000预算，并与q936同完整22/同旧Diffusion fixture6000controls对照，保留跌倒/idle/reset/unknown bins。q935动作输入结果作为机制上下文；不以MMD筛选启动，不提前更改默认。正在运行的正式Diffusion40k保持GPU独占。
