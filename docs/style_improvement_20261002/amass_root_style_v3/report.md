# AMASS root 课程：九模型固定风格 / full22 对照 v3

新增真实500→750续训分支：16/22 success、2 fall，位置误差0.401208m；inclusive MMD²0.085542。原1000为17/22、0 fall、0.229298m、MMD²0.056075，仍是更稳候选。750直立帧子集MMD²0.037172不能替代包含全部失败帧的正式评分。

旧八份 evaluator 与 comparison run dictionary 逐值一致，c274冻结reference未变；所有22段/11982 frame与442个750 fall frame完整保留，未调公式、bins或门槛。

|模型|success/22|fall|位置 m|姿态 rad|inclusive MMD²|速度W m/s|
|---|---:|---:|---:|---:|---:|---:|
|legacy1000|6|1|0.377293|0.859479|0.093668|0.257637|
|amass2848_child500|1|0|0.441347|1.027874|0.050227|0.213351|
|amass2848_child1000|5|2|0.430993|1.254071|0.206480|0.092944|
|backward_penalty1000|11|0|0.300708|0.861454|0.046821|0.143767|
|stratified_experts1000|8|2|0.352941|0.791256|0.089115|0.200738|
|stratified_experts500|1|0|0.460473|0.869972|0.055555|0.305356|
|amass_root1000|17|0|0.229298|0.831882|0.056075|0.065651|
|amass_root500|12|1|0.357862|0.875744|0.038601|0.047519|
|amass_root750|16|2|0.401208|0.805335|0.085542|0.053742|

## 750真实续训身份

来源：sealed500 SHA `8db53ffad6b60782a7e73dd924de675f0caaa941d951efb6806fad22cd092314`；独立output新增250更新/2048000transition，累计750/6144000。终点SHA `1e87b82b643a1d94bf2117de5a57a27ebb1e0e9811a4cabf66ab1588fcaf9592`，652093749B。

它是验证/风格取舍之后提出的新续训实验，不是原连续1000训练中遗漏的750快照。续训result的46fall/19971time-limit只属于新增250更新，未伪称完整750累计值，也未声明PhysX cache逐位连续。配置/资源保持，预算仍不同；未读取sealed34测试输出。

## 750与1000的诊断

750相对1000无新成功，丢失motion14；位置误差只有4/22段改善，姿态10/22段改善。摔倒发生于SFU BackwardsWalk001和Moonwalk001，原1000这两段虽未达到tracking success但没有摔倒。

rootXY漂移0.325017m（1000为0.142768）；yaw误差0.501213rad（1000为0.551314，部分改善）。去平移/heading后的局部pose位置0.135398m（1000为0.125401）；L/R wrist parent-relative rotation0.674780/0.643413rad（1000为0.553035/0.513716）。综合跟踪稳定性未优于1000。

750速度W0.053742较1000的0.065651略低，实际速度均值0.908380较参考0.930120接近；但低速bin受倒地长期帧影响，0–0.1m/s MMD²升至0.272022。750upright仅保留96.112%生成样本，inclusive0.085542与upright0.037172分离；正式比较坚持inclusive。

toe近地body-origin代理均速0.615793m/s、超阈比例0.290459看似降低；其中含倒地、代理有效区间不同，不能视作真实脚部接触或更少物理滑动。原生ground/COM/contact需独立测量，90D风格指标不包含这些。

## 速度分层 MMD²

|实际速度 m/s|AMASS root500|750新续训|原1000|
|---|---:|---:|---:|
|0.0–0.1|0.043297|0.272022|0.088217|
|0.1–0.5|0.089308|0.083483|0.092030|
|0.5–1.0|0.009179|0.010974|0.014914|
|1.0–1.5|0.019880|0.014456|0.034496|
|1.5–2.0|0.031341|0.046773|0.050717|
|2.0–3.0|unsupported|unsupported|unsupported|
|3.0–∞|unsupported|unsupported|unsupported|

全部沿用固定sigma1、速度bins、最少30样本/每bin最大256；≥2m/s仍不足，不重新调参。三对750与negative/500/1000的初始physical/AMP与完整reference/clock/target数组均22段逐元素一致。

现结论只适用于反复暴露的validation低层真未来XY跟踪；不是盲test、多seed显著性或作者8场景结果。未因upright子集分数改选，也未把更低速度误差等同完整动作风格改善。

证据：[evaluation](evaluation.json)、[comparison](comparison.json)、[verification](verification.json)、[750同帧分解](../amass_root_from500_to750_tracking_decomposition.json)。[八模型历史v2](../amass_root_style_v2/report.md)保留原样。CPU only；没有修改业务/配置/资源/权重或启动Kit/GPU。
