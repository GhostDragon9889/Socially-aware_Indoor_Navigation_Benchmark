# Mixed1000 完整22验证风格对照

2026-10-02。沿用冻结工具 `d19c9cea62839792c85fccc92cdefd5bf9a3f1d27c3c39c7b0b7f6d0d4783612`、参考 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564` 和原 90D / sigma1 指标。只进行 CPU 工件分析；未改业务、指标或参考，未读取封存34测试。

## 完整实际结果

| 模型 | 更新 / transition | 完整动作成功 | 跌倒 | 位置误差 m | 旋转误差 rad | inclusive MMD² | 速度 W1 m/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| negative-backward1000 | 1000 / 8,192,000 | 11/22 | 0 | 0.300708 | 0.861454 | 0.046821 | 0.143767 |
| AMASS-root1000 | 1000 / 8,192,000 | 17/22 | 0 | 0.229298 | 0.831882 | 0.056075 | 0.065651 |
| mixed1000 | 1000 / 8,192,000 | 19/22 | 0 | 0.203485 | 0.875661 | 0.054961 | 0.056390 |

全部22动作 / 11,982 metric 帧保留，3 个失败动作没有删除。本次十模型机器字典保留旧九模型原值，仅新增 mixed1000。与 negative 基线的22个初始物理状态 / AMP history、全部 reference 与时钟数组逐位一致。

相同预算下 mixed1000 的成功率、world body position 误差和速度分布 W1 均优于表内两基线。旋转误差仍高于两者，整体 MMD² 高于 negative；因此可以说动作跟踪改善，不能说整体风格已优于全部基线。[先前 mixed500 独立报告](../step500_comparison_v1/README.md) 的成功率16/22、MMD² `0.062570`、速度 W1 `0.090754`，原报告保持不变。

## 速度与姿态短板

- mixed1000 的 0–0.1 / 0.1–0.5 m/s 桶 MMD² 是 `0.097318` / `0.100821`，高于 negative 的 `0.051339` / `0.087036`。0.5–1 / 1–1.5 / 1.5–2 m/s 桶为 `0.015156` / `0.018093` / `0.043414`，低于 negative 的 `0.016421` / `0.023122` / `0.056186`。低速动作仍拖累整体风格；≥2 m/s 样本不足，不计为已支持的桶。
- root XY 漂移均值 `0.115572 m`、yaw 误差 `0.534792 rad`，低于 negative 的 `0.228346` / `0.600987`；heading 对齐后 body position 误差 `0.122856 m`、parent-relative 关节旋转误差 `0.321271 rad`，仍高于 negative 的 `0.119189` / `0.301823`。位移与朝向改善不能替代姿态质量。
- L/R Wrist 的 parent-relative 误差 `0.866274` / `0.844557 rad`，高于 negative 的 `0.798023` / `0.782541`，腕部仍是具体风格短板。手部冻结与 wrist 错误应分开解释。
- 近地 toe body-origin 滑动代理比例 `33.4908%`（negative `35.1220%`，AMASS-root1000 `32.0159%`）。该代理未测 native contact / sole 支持或真实 COM，不代表足部物理验收。

## 工件与边界

真实 checkpoint `step_1000.pt` 为 `652,179,357` 字节，SHA `76708bfa8cc2a55d621f8c3b39a864b66a3b2522eb15d2ca7f451583db165d7d`。实际两组各 `4,096,000` actor-transitions；AMASS / procedural reset 为 `40,025` / `13,735`，并非各半。安全 CPU mmap 核对 format、计数、v7来源、assignment、专家采样 disabled 与实际组统计；没有执行 optimizer / PhysX 恢复。

这是已经多次使用的完整22 validation 跟踪结果，使用真实未来 XY；不是盲测、不是 learned Diffusion 闭环导航、不是作者八场景复现通过。当前尚未据 mixed1000 的新四人闭环捕获证明导航迁移效果。

[完整机器对照](comparison.json) / [evaluation](evaluation.json) / [冻结参考](heldout_reference.json) / [同帧姿态分解](mixed1000_decomposition.json) / [CPU执行日志](execution.log) / [交付校验](delivery.json)。
