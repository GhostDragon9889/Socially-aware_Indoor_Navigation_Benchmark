# 专家窗口分层采样：固定留出动作比较（五模型）

核验时间：2026-10-02T06:13:21.286334+00:00。CPU 4 threads，无 Kit / GPU；业务代码、数据、配置和 frozen evaluator 均未修改。

## 判断

**保留 negative-only /1000 为当前参考模型。** 新采样模型的上肢与 yaw 误差改善，但跟踪成功、位置、跌倒与 inclusive 风格距离退步，不能当作整体动作改善。

两轮各真实训练 1000 更新 / 8,192,000 transitions；training config 唯一差异是 `expert_sampling`。初始化、任务、物理、精度、预算相同；采样器保留匹配调用时的 corpus RNG 支出，但真实 fall/reset 发生时间与训练轨迹不保证相同。

## 固定指标

| 模型 | 成功 /22 | 跌倒序列 | 位置 m | 旋转 rad | inclusive MMD² | speed W m/s |
|---|---:|---:|---:|---:|---:|---:|
| 旧319clip /1000 | 6 | 1 | 0.377293 | 0.859479 | 0.093668 | 0.257637 |
| v7 positive /500 | 1 | 0 | 0.441347 | 1.027874 | 0.050227 | 0.213351 |
| v7 positive /1000 | 5 | 2 | 0.430993 | 1.254071 | 0.206480 | 0.092944 |
| negative-only /1000 | 11 | 0 | 0.300708 | 0.861454 | 0.046821 | 0.143767 |
| stratified expert /1000 | 8 | 2 | 0.352941 | 0.791256 | 0.089115 | 0.200738 |

22 条原 validation 动作、11,982 个 post-step metric frames 全部保留，含所有跌倒帧；失败没有裁掉。原四模型的完整 run 字典与前报告逐值相同。固定 reference SHA256：`c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`。

新1000相对 negative-only：无新增成功，丢失 motion IDs 11、23、244；位置误差仅 5/22 clips 改善，旋转误差 16/22 改善。两个跌倒源均为 SFU0018 DanceTurns001/002。

## 动作风格与速度分布

| 物理速度区间 m/s | negative MMD² | stratified MMD² | negative / stratified 样本数 |
|---|---:|---:|---:|
| [0.0, 0.1) | 0.051339 | 0.205671 | 300 / 407 |
| [0.1, 0.5) | 0.087036 | 0.090303 | 1555 / 2075 |
| [0.5, 1.0) | 0.016421 | 0.026302 | 7861 / 7542 |
| [1.0, 1.5) | 0.023122 | 0.039210 | 2010 / 1715 |
| [1.5, 2.0) | 0.056186 | 0.084088 | 230 / 205 |
| [2.0, 3.0) | unsupported | unsupported | 4 / 16 |
| [3.0, ∞) | unsupported | unsupported | 0 / 0 |

每个有支持的速度区间均变差；2m/s 以上参考支持不足，未强行评分。新 upright-only MMD² .091907，negative .046821，因此 inclusive 退步不能完全归因于两个跌倒。新 upright 保留 .989716，negative 1.0。

## 同帧姿态分解与脚部代理

| 指标 | negative | stratified |
|---|---:|---:|
| root yaw rad | 0.600987 | 0.530187 |
| 去 root 平移与各自 heading 的局部位置 m | 0.119189 | 0.133597 |
| 父子关节相对旋转 rad | 0.301823 | 0.291410 |
| L_Wrist 父关节相对旋转 rad | 0.798023 | 0.575697 |
| R_Wrist 父关节相对旋转 rad | 0.782541 | 0.570617 |
| 近地 toe body-origin 速度代理 m/s | 0.725802 | 0.649377 |
| 同代理 >0.1m/s fraction | 0.351220 | 0.317288 |

toe 代理改善只能说明该高度/垂速筛选下的 body-origin 运动变化，不是真实脚掌接触或支撑点滑移。没有为这些 tracking captures 推断真实足接触或 COM。

## 证据与范围

- [完整五模型 evaluation](evaluation.json)
- [机器比较及22逐数组输入/初始状态校验](comparison.json)
- [冻结 reference](heldout_reference.json)
- [新1000同帧分解](../stratified_experts_tracking_decomposition.json)
- [CPU运行 log](../stratified_experts_style_v1.log)
- [summary Ruff](ruff.log)

新/旧1000各 clip 初始 physical state、AMP history、完整参考与 clock/target 数组均完全相同。全部模型的声明训练父源与固定 validation/test 源无交集；不据此声称 PACER 预训练名单、真实人物身份相互独立。

90D 指标使用局部位置+速度、speed-bin Gaussian biased MMD²、sigma=1。作者精确统计与阈值未知。单 seed、已反复使用的 validation、不同比较模型的预算差异均限制结论。该任务是 truefutureXY oracle 的低层独立跟踪，未据此声称 learned Diffusion 避障、人群导航或作者八场景复现。

第500更新将另建六模型报告作为事后学习阶段诊断，不覆盖此五模型结果，不将早期权重筛选说成盲测。
