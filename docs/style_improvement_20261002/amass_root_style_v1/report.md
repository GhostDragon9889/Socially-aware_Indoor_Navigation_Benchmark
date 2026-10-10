# AMASS root 课程：七模型固定风格与 full22 对照 v1

新1,000 checkpoint：17/22 success、0 fall，位置误差0.229298m。低层跟踪明显改善；冻结风格 MMD² 从0.046821升至0.056075，不能声称动作风格全面改善。

所有22段、11,982个真实 post-step frame 保留；11960个同源导数 interval，不跨 clip/reset 求差。参考 artifact SHA `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564` 完全不变，旧六份 evaluator run dictionary 和 comparison run dictionary 逐值未变。

|模型|success/22|fall|位置 m|姿态 rad|inclusive MMD²|速度W m/s|
|---|---:|---:|---:|---:|---:|---:|
|legacy1000|6|1|0.377293|0.859479|0.093668|0.257637|
|amass2848_child500|1|0|0.441347|1.027874|0.050227|0.213351|
|amass2848_child1000|5|2|0.430993|1.254071|0.206480|0.092944|
|backward_penalty1000|11|0|0.300708|0.861454|0.046821|0.143767|
|stratified_experts1000|8|2|0.352941|0.791256|0.089115|0.200738|
|stratified_experts500|1|0|0.460473|0.869972|0.055555|0.305356|
|amass_root1000|17|0|0.229298|0.831882|0.056075|0.065651|

## 与同预算 negative-backward 基线的变化

新成功 IDs：4, 14, 242, 254, 259, 260；丢失成功0。位置误差19/22段改善、姿态误差11/22改善。两个模型均0 fall，全新模型也100% upright，因此此处 inclusive/upright MMD 相同。

实际 root planar speed 均值 0.777697→0.896613m/s，参考同段0.930120m/s；速度W 0.143767→0.065651m/s。

rootXY漂移 0.228346→0.142768m；yaw误差 0.600987→0.551314rad。去掉平移/heading 后 local pose 误差 0.119189→0.125401m，略退步，说明整体位置改善主要来自 root 跟踪。

腕部 parent-relative rotation L/R：0.798023/0.782541→0.553035/0.513716rad，改善；腕部去heading位置 L/R：0.225962/0.219637→0.256107/0.227967m，退步。

脚部近地 body-origin 代理：滑动比例 0.351220→0.320159，均速 0.725802→0.696977m/s。代理有效区间18692→14321，不能把该变化当作真实接触、防滑或 COM 改善。原生足部/地面接触需独立记录验证。

## 冻结速度分层 MMD²

|实际速度 m/s|negative baseline|AMASS root|新生成/参考样本数|
|---|---:|---:|---:|
|0.0–0.1|0.051339|0.088217|289/148|
|0.1–0.5|0.087036|0.092030|1180/2251|
|0.5–1.0|0.016421|0.014914|6020/6664|
|1.0–1.5|0.023122|0.034496|4148/5324|
|1.5–2.0|0.056186|0.050717|323/969|
|2.0–3.0|unsupported|unsupported|0/4|
|3.0–∞|unsupported|unsupported|0/0|

低速0–0.5与1–1.5m/s的 MMD 退步，0.5–1与1.5–2m/s改善；≥2m/s样本不足，未人为补齐或改变门槛。

## 可比较范围

新旧1,000模型均8,192,000 transition、相同公开PACER warm start、reward/PPO/D/RMS 配置和物理参数。旧训练 config 缺少后来可选 `expert_sampling` 键，新为显式null；按既有 disabled 默认语义一致，raw JSON 不同已保留。AMASS-root 改变真实根轨迹命令、legal reset frame 与 clip-end truncation 分布，是声明的 curriculum intervention，不是只修改专家分布或单项损失。初始24体/AMP history 与完整 reference/clock/target 数组22段逐元素相等。训练 fall313→157但训练episode长度/截断分布不同，不能直接作等机会fall率比较。

当前新模型是此 full22 低层跟踪验证的较好候选；仍剩5段失败（HumanEva两段Jog、SFU BackwardsWalk、DanceTurns002、Moonwalk）。尚未证明新控制器的实际多人 Diffusion 导航、真实地面接触、8场景或作者效果。验证集反复用于迭代、只有单训练seed；不是独立test或多seed显著性结果。

完整机器证据：[evaluation.json](evaluation.json)、[comparison.json](comparison.json)、[verification.json](verification.json)。同帧诊断：[decomposition](../amass_root_tracking_decomposition.json)。CPU工具只读原资源/权重/结果，未启动 Kit/GPU 或修改训练代码。
