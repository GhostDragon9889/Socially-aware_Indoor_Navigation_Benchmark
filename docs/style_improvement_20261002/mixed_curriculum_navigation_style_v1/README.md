# Mixed1000/2000：完整200秒导航的冻结风格与停留对照

**CPU 工件/测量合同通过，导航稳定性未通过。保留 negative1000 guide20 的既有 0 跌倒部署基线。** full22 的 19/22、0 fall tracking 成绩不能直接推广为多人闭环稳定性。

4人、6000步/200秒、120Hz物理/30Hz控制、同 Diffusion5000、guide20/eta0、1.5m/s因果reference、15步同步replan、同场景/接触/ground设置。新的两个 run 全时段已完成；没有提前停止、删跌倒/低姿态/低速帧。`result.status=passed` 表示执行完成，不表示质量达标。

|指标|negative1000 guide20|AMASS-root1000|mixed1000|mixed2000|
|---|---:|---:|---:|---:|
|训练 transitions|8.192M|8.192M|8.192M|16.384M|
|fall|0|13|13|11|
|路线累计顶点|21|5|15|19|
|inclusive MMD²|0.072381|0.078196|0.072470|0.116486|
|upright-only MMD²（次要）|0.072421|0.050850|0.065134|0.110717|
|upright retained|99.9542%|99.0535%|99.5413%|99.2620%|
|速度分布 W1 m/s|0.353368|0.519750|0.360403|0.306563|
|实际位置差分速度均值 m/s|0.560623|0.393868|0.553193|0.607087|
|idle总actor秒 / 800秒|194.766667|251.266667|144.300000|71.766667|
|idle / 合法同episode区间|24.3499%|31.4306%|18.0503%|8.9764%|
|A*查询失败|0|173|0|7|
|start recovery|0|38|0|10|
|最低pelvis m|0.713017|0.390584|0.379404|0.395899|

新模型路线分别 `[6,2,5,2]` / `[5,4,4,6]`，均为实际累计到达顶点；有 fall 重启，不是无中断路线成功或作者八场景结果。mixed2000 的 15 个 before-frame reset mask 包含 4 个初始 reset + 11 个 fall 后 reset，不能误报 15 次 fall/reset 终止。mixed1000 同理为 4+13；两个 run 的 time-limit 均为0。

## 评分合同与分母

固定 c274 reference、90D pose+velocity、sigma1、原速度 bins、每bin最少30/最多256。没有改变 evaluator、特征、内核、参考、阈值或指标。原 negative 与 AMASS-root evaluator/comparison/stagnation 字典逐值保留，没有重算旧评分和旧停留对象。

24000 actor-frame全部保留；仅导数的跨reset区间排除。合法区间为 negative23996、AMASS-root23983、mixed1000 23983、mixed2000 23985；合法加速度triplet为23992/23966/23966/23970。不同分母来自实际 reset，未截短或删除失败段。upright-only 只是敏感性结果，不能取代 inclusive 主评分。

|物理速度bin m/s|negative MMD²|mixed1000 MMD²|mixed2000 MMD²|
|---|---:|---:|---:|
|0–.1|.162037|.111445|.095694|
|.1–.5|.045008|.104319|.115020|
|.5–1|.026610|.013651|.047209|
|1–1.5|.051330|.039082|.125343|
|1.5–2|.076919|.093851|.199164|
|≥2|unsupported|unsupported|unsupported|

mixed1000 的平均正式MMD与基线近似，但在 .1–.5 m/s 段明显变差。mixed2000 的 idle 减少、速度W1改善并没有带来更接近参考的动作形态；五个支持bin平均MMD明显恶化。unsupported 表示真实样本不足，不表示零误差。训练预算不等且只有一个seed，不给显著性/收敛判断。

## 停留、旋转和上肢描述

停留复用原 `summarize_navigation_style.stagnation`：同episode位置差分速度<0.1m/s，原时间 t=(i+1)/30；逐replan执行target/velocity/raw target phase均核验。全failure帧保留，只排除reset跃迁。

|actor|negative idle秒|mixed1000 idle秒 / 最长连续秒|mixed2000 idle秒 / 最长连续秒|
|---:|---:|---:|---:|
|0|116.2000|3.7000 / .7000|34.8667 / 24.8000|
|1|10.0667|111.1667 / 51.5333|17.4333 / 6.0333|
|2|30.2000|18.6667 / 1.9333|7.8000 / 1.2000|
|3|38.3000|10.7667 / 1.9333|11.6667 / 3.9667|

mixed1000 大部分停留集中 actor1，不能只展示 actor0 的改善。mixed2000 整体停留下降，但仍有 actor0 的24.8秒连续段。

导航没有逐帧 AMASS body orientation 真值，不能计算 tracking 的 root/body rotation geodesic 误差。新[描述性yaw/wrist](descriptive_yaw_wrist.json)仅报告实际 same-episode root角速率、moving时heading与位移关系及原heading-local wrist位置/速度，不增加评分或更改MMD：

|描述量|negative|mixed1000|mixed2000|
|---|---:|---:|---:|
|绝对root yaw rate均值 rad/s|.547305|.527626|.618719|
|moving位移与朝向相反比例（speed≥.1）|10.4996%|8.7870%|2.1436%|
|L/R wrist local速度 chain/s|.344912/.353484|.245020/.247269|.295770/.256504|

这些统计不等于身体旋转真值误差、上肢自然度或论文阈值。moving选择仅用于此描述，不过滤主评分；更低腕部速度不自动代表更好风格。

## 足部代理与真实测量边界

原 toe近地body-origin代理均速 negative .397334、mixed1000 .385981、mixed2000 .401470m/s；超过原.1m/s阈值的比例20.7178%→23.2858%/24.4298%，没有代理防滑改善。代理有效foot-interval分别38059/32698/31875。它不是原生contact-point slip，不能混用原生foot/COM/contact支持统计作为新的AMASS真值。

新 ground/COM/contact 源的独立2259项验收分别见 [mixed1000](../mixed_step1000_navigation_ground_actual_verification_v1.json) / [mixed2000](../mixed_step2000_navigation_ground_actual_verification_v1.json)；本分析只复用其工件绑定和原停留公式，不重算原生测量内核。场景/120/30Hz/PD/contact/power/适配参数逐值相同，训练controller和课程/统计/权重不同；不能仅据一个静态差异归因。

证据：[完整比较](comparison.json)、[原style输出与历史字典](evaluation.json)、[停留与phase](stagnation.json)、[真实新run原style](new_evaluation/evaluation.json)、[固定参考](heldout_reference.json)。CPU-only，没有启动SDK/GPU/Kit、修改业务/数据/模型或读取sealed34输出；反复使用validation已明确披露。
