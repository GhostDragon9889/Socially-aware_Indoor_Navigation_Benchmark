# Mixed500 完整22验证风格对照

2026-10-02。只用 CPU 对已完成的真实 capture 评估；没有再训练 / 模拟、修改指标或读取封存34测试。冻结工具 SHA `d19c9cea62839792c85fccc92cdefd5bf9a3f1d27c3c39c7b0b7f6d0d4783612`，参考 SHA `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`。

## 实际结果

| 模型 | 更新 / transition | 完整动作成功 | 跌倒 | 位置误差 m | 旋转误差 rad | inclusive MMD² | 速度 W1 m/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| negative-backward1000 | 1000 / 8,192,000 | 11/22 | 0 | 0.300708 | 0.861454 | 0.046821 | 0.143767 |
| AMASS-root1000 | 1000 / 8,192,000 | 17/22 | 0 | 0.229298 | 0.831882 | 0.056075 | 0.065651 |
| mixed500 | 500 / 4,096,000 | 16/22 | 0 | 0.271462 | 0.924934 | 0.062570 | 0.090754 |

全部 22 个动作、11,982 个 metric 帧保留；仅去除每个动作的初始 reset 帧及跨 clip 速度区间。mixed500 的初始物理状态 / AMP history、所有实际 reference / 时间数组与 negative 基线逐位一致。旧九模型的 evaluation / comparison 字典原值完整保留；本次总计十模型，旧模型没有重算。

mixed500 在 0.5–1 m/s 与 1–1.5 m/s 两个速度桶的 MMD² 为 `0.011267` / `0.019625`，低于 negative 的 `0.016421` / `0.023122`；其 0–0.1、0.1–0.5、1.5–2 m/s 桶分别为 `0.101492`、`0.105889`、`0.074578`，均高于 negative。因此整体风格 MMD² 尚未改善。速度分布 W1 优于 negative，但低于 AMASS-root1000；不同更新预算不能视为最终候选等预算胜出。

## 姿态与位移分解

mixed500 的 root XY 漂移均值 `0.190086 m`，低于 negative 的 `0.228346 m`；heading 对齐后的 body position 误差 `0.123739 m`，高于 negative 的 `0.119189 m`。yaw 误差 `0.626893 rad`、parent-relative 关节旋转误差 `0.321446 rad` 也高于 negative 的 `0.600987` / `0.301823`。L/R Wrist 的 parent-relative 误差为 `0.860032` / `0.894665 rad`，仍是具体短板，不能把成功率提高等同于全部姿态更自然。

近地 toe body-origin 滑动代理比例 `33.4625%`（negative `35.1220%`，AMASS-root1000 `32.0159%`）。这只是冻结的 body-origin / 高度代理，没有 native 接触或真正 sole / COM 测量，不应解释为足部物理支持率。

## 实验边界与证据

mixed500 checkpoint SHA `42939bf6a120e478541eb7ea7209d72158acb075fec30f86931341c8044d084b`。metadata 证明两组各 `2,048,000` 个真实 actor-transitions；AMASS / procedural reset 数为 `20,024` / `6,931`，50% transition 暴露不等于 reset 或 plan 数各半。checkpoint 完整 metadata 已以安全 CPU mmap 核对，但本轮没有恢复 optimizer / PhysX。

所有模型用相同 frozen 90D pose/velocity 特征、sigma1、固定256×61 reference windows、速度桶与 inclusive 规则。upright 是次指标；本次 mixed500 无跌倒，因此与 primary 数值相同。完整22属于已多次使用的 validation，不能声明 blind test、多 seed 显著性或达到作者论文标准。动作跟踪使用真实未来 XY；它还不是 learned Diffusion 闭环导航表现。

[完整机器对照](comparison.json) / [完整 evaluation](evaluation.json) / [新增模型 evaluator 日志](evaluation.log) / [同帧分解](mixed500_decomposition.json) / [冻结参考](heldout_reference.json) / [交付 SHA](delivery.json)。
