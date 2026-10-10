# AMASS-root1000：四人200秒闭环迁移失败，固定风格与停留对照

实际run完整执行、数据验收通过；模拟质量未通过。新控制器13次fall、5个路线顶点，旧negative guide20为0 fall、21顶点。保留现有0fall部署基线；不因full22的17/22成功将此权重推广为稳定导航。

固定参数：4人/6000控制步/200秒、guide20/eta0、同Diffusion5000权重、相同场景/120Hz物理/30Hz控制/1.5m/s causal reference/15步同步replan/1N原生contact及ground记录。仅trained controller/profile和output改变；旧eta键缺失对应显式0。

|指标|旧negative guide20|AMASS-root1000|
|---|---:|---:|
|fall|0|13|
|路线累计vertices|21|5|
|inclusive MMD²|0.072381|0.078196|
|upright-only MMD²|0.072421|0.050850|
|upright retained|99.954159%|99.053496%|
|速度W m/s|0.353368|0.519750|
|实际位移速度mean m/s|0.560623|0.393868|
|idle actor秒（四人总分母800秒）|194.766667|251.266667|
|idle/合法同episode区间|24.349892%|31.430597%|
|A*查询失败|0|173|
|start recovery|0|38|
|最低pelvis m|0.713017|0.390584|

路线计数旧[2,8,6,5]→新[2,2,1,0]；是累计实际到达顶点，不是episode完整成功或作者8场景标准。新0time-limit不是稳定性证明，13fall全保留。

## 风格/速度

冻结c274参考、90D特征、sigma1、速度bins/最少30样本/最多256均不变，旧guidance20 evaluator dictionary逐值一致。新inclusive MMD0.078196略差于0.072381；upright0.050850较低但排除了部分低姿态帧，不能作为替代正式分数。速度W和实速明显退步，不能以更低近地toe代理宣称姿态更自然。

toe近地body-origin均速 0.397334→0.263013m/s；超阈比例 0.207178→0.179660。有效区间38059→34749、实际速度减少且有falls，这不是原生contact-point防滑改善。

所有24000 actor-frame保留；只排除跨reset导数，新23983（旧23996）同episode interval、新23966（旧23992）加速度triplet。不同分母是实际13次reset的结果，不为分数截短。

## 按原停留公式的逐人结果

同episode30Hz rootXY位移速度<0.1m/s；沿用原source time t=(i+1)/30，未更改阈值、特征或插值。旧对象直接复用历史报告，新仅调用一次unchanged helper，并严格验证每个saved replan与实际target phase。

|actor|旧idle秒|新idle秒|旧最长连续秒|新最长连续秒|
|---:|---:|---:|---:|---:|
|0|116.200000|73.533333|38.266667|27.366667|
|1|10.066667|50.000000|1.766667|23.666667|
|2|30.200000|56.033333|5.233333|12.200000|
|3|38.300000|71.700000|8.600000|12.800000|

actor0 idle减少，其他三人增加，总停留增多；未因局部actor0好看而改变整体结论。停留包含fallen低速段，AABB仅root点几何诊断，不用它冒充碰撞或证明因果。原生ground/COM/contact完整性另见独立审计，未重新计算其核。

## 迁移原因与下一步（已在34测试开封前冻结）

[只读checkpoint诊断](../amass_root_transfer_static_review_v1.md)：两控制器terrain1024 RMS mean/variance/count逐元素完全相同，不能直接归因“terrain RMS改坏”。source-phase真未来XY/短source-end episode与独立程序化命令/长episode的分布明确不同，self/trajectory/AMP/value RMS与network也同时变化，单静态差异无法分离因果。平均episode298.411→102.085步；55.628%未来XY knot是声明的command-only线性尾部。

下一候选仅建议fresh同公开PACER warmstart、显式程序化/AMASS混合命令课程，并记录真实transition比例；不在此改业务、权重或指标。这里没有读取34测试capture；固定validation已反复使用、只有一个seed，不是盲test或统计显著性。full22 tracking improvement与闭环质量不同，论文多人导航效果尚未达标。

证据：[evaluation](evaluation.json)、[comparison](comparison.json)、[verification](verification.json)、[原停留公式输出](../amass_root_navigation_stagnation_v1.json)、[原生ground独立审计](../amass_root_step1000_navigation_ground_actual_verification_v1.md)。完整历史与新原始结果全部保留。CPU only，没有更改业务/配置/权重或启动GPU/Kit。
