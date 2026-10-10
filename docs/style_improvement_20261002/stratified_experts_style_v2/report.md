# 专家窗口分层采样：六模型与学习阶段诊断

核验时间：2026-10-02T06:18:53.956139+00:00。仅CPU，4 threads，未启动GPU或Kit；没有修改任何模型/数据/业务/配置/evaluator。

## 判断

**当前仍保留 negative-only /1000。** 相同预算的新 stratified /1000 改善 yaw 与腕部旋转，却在跟踪成功、位置、跌倒和固定风格距离上退步。真实第500更新也没有解决总体问题；它的零跌倒伴随1/22成功与更慢的实际动作。此事后阶段诊断未筛出可替代参考模型。

本报告新增真实500阶段并保留 [前五模型快照](../stratified_experts_style_v1/report.md)。原五个 run 字典逐值相同；同一 fixed reference SHA256 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`。所有22原动作和11,982 metric frames（含失败/跌倒）原样评估。

## 固定结果

| 模型 | 训练 transitions | 成功 /22 | 跌倒序列 | 位置 m | 旋转 rad | inclusive MMD² | speed W m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧319clip /1000 | 2,048,000 | 6 | 1 | 0.377293 | 0.859479 | 0.093668 | 0.257637 |
| v7 positive /500 | 4,096,000 | 1 | 0 | 0.441347 | 1.027874 | 0.050227 | 0.213351 |
| v7 positive /1000 | 8,192,000 | 5 | 2 | 0.430993 | 1.254071 | 0.206480 | 0.092944 |
| negative-only /1000 | 8,192,000 | 11 | 0 | 0.300708 | 0.861454 | 0.046821 | 0.143767 |
| stratified expert /1000 | 8,192,000 | 8 | 2 | 0.352941 | 0.791256 | 0.089115 | 0.200738 |
| stratified expert /500 | 4,096,000 | 1 | 0 | 0.460473 | 0.869972 | 0.055555 | 0.305356 |

negative-only 与 stratified1000 训练配置唯差 `expert_sampling`，各1000更新/8.192M transitions，初始化、任务、物理、精度、预算一致。采样器仅改变 D 专家样本先验；旧共享 RNG 支出按匹配调用保留，实际训练跌倒/reset时序与物理轨迹不保证一致。

## 500→1000：并非单纯晚期才退化

| 诊断 | negative1000 | stratified500 | stratified1000 |
|---|---:|---:|---:|
| root yaw rad | 0.600987 | 0.598754 | 0.530187 |
| capture实际 root planar velocity m/s | 0.777697 | 0.616616 | 0.722394 |
| 去平移与heading局部位置 m | 0.119189 | 0.126039 | 0.133597 |
| 父子相对旋转 rad | 0.301823 | 0.296816 | 0.291410 |
| L_Wrist父关节相对旋转 rad | 0.798023 | 0.602781 | 0.575697 |
| R_Wrist父关节相对旋转 rad | 0.782541 | 0.597975 | 0.570617 |
| 近地 toe body-origin 速度代理 m/s | 0.725802 | 0.479005 | 0.649377 |
| 同代理 >0.1m/s fraction | 0.351220 | 0.277638 | 0.317288 |

500 的 tracking 位置 .460473m 和 speed W .305356 均较差；500→1000成功1→8、位置改善，却新增2个跌倒序列。1000两fall均为 SFU DanceTurns001/002；相比negative没有新增成功，丢失 IDs11/23/244。500零跌倒不足以作为接受标准。

toe代理变低可以来自动作更慢；它不是真实脚掌接触或支撑点滑移，也不是自然步态保证。500→1000的改善/恶化不能用此单代理概括。

## 物理速度分层 MMD²

| 速度区间 m/s | negative1000 | stratified500 | stratified1000 |
|---|---:|---:|---:|
| [0.0, 0.1) | 0.051339 | 0.062997 | 0.205671 |
| [0.1, 0.5) | 0.087036 | 0.135848 | 0.090303 |
| [0.5, 1.0) | 0.016421 | 0.023814 | 0.026302 |
| [1.0, 1.5) | 0.023122 | 0.020475 | 0.039210 |
| [1.5, 2.0) | 0.056186 | 0.034639 | 0.084088 |
| [2.0, 3.0) | unsupported | unsupported | unsupported |
| [3.0, ∞) | unsupported | unsupported | unsupported |

1000每个有支持的速度bin都比negative变差；upright-only .091907 vsnegative .046821，不能将差异完全归因于fall。500 upright=1.0，MMD² .055555仍未优于negative .046821；2m/s以上支持不足，未强评分。

## 配对与范围

22条初始physical state、AMP history、完整reference/clock/target数组在 negative、stratified500、stratified1000之间完全一致，无时间相位搜索或删掉失败帧。各宣告训练父源与旧完整val/test源 disjoint；未知PACER预训练名单/真实人身份独立性未作保证。

90D风格指标为局部位置+速度、speed-bin Gaussian biased MMD²，sigma=1；作者精确实现/阈值未知。单seed、重复暴露validation、500半预算限制结论。这里使用truefutureXY oracle低层跟踪，不等同于 learned Diffusion 人群导航或八场景论文复现。

## 文件

- [完整六模型 evaluation](evaluation.json)
- [机器比较及逐数组配对证明](comparison.json)
- [冻结参考](heldout_reference.json)
- [1000分解](../stratified_experts_tracking_decomposition.json)
- [500分解](../stratified_experts_step500_tracking_decomposition.json)
- [前五模型不可覆盖快照](../stratified_experts_style_v1/verification.json)
- [六模型CPU运行 log](../stratified_experts_style_v2.log)

第200训练检查点已被替换，明确不可得；500/1000模型均为按真实更新数保存的原完整权重，未改名伪装或重构。
