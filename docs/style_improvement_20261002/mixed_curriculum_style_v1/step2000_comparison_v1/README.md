# Mixed2000 完整22验证风格对照

2026-10-02。只用 CPU 对已完成的真实22动作捕获分析；冻结工具 `d19c9cea62839792c85fccc92cdefd5bf9a3f1d27c3c39c7b0b7f6d0d4783612`、参考 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564` 与指标均未改变。未读封存34测试。

## 完整结果

| 模型 | 更新 / transition | 完整动作成功 | 跌倒 | 位置误差 m | 旋转误差 rad | inclusive MMD² | 速度 W1 m/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| negative-backward1000 | 1000 / 8,192,000 | 11/22 | 0 | 0.300708 | 0.861454 | 0.046821 | 0.143767 |
| AMASS-root1000 | 1000 / 8,192,000 | 17/22 | 0 | 0.229298 | 0.831882 | 0.056075 | 0.065651 |
| mixed2000 | 2000 / 16,384,000 | 19/22 | 0 | 0.200096 | 0.768363 | 0.060441 | 0.045834 |

保留全部22动作、11,982 metric 帧和3个跟踪失败。旧九模型字典原值不变；本次十模型结果只新增 mixed2000。全部22初始物理状态 / AMP history、reference 和时钟数组与 negative 基线逐位一致。

mixed2000 位置、旋转误差和速度分布 W1 优于表内基线，但其训练 transition 为基线两倍，不能忽略预算差异。inclusive 风格 MMD² 仍高于 negative 与 AMASS-root1000；19/22 成功并非完整效果通过。

## 与 mixed1000 的具体权衡

[mixed1000](../step1000_comparison_v1/README.md) 同为19/22成功、0跌倒，位置 / 旋转误差 `0.203485 m` / `0.875661 rad`、MMD² `0.054961`、速度 W1 `0.056390 m/s`。2000 的旋转与速度分布改善，primary MMD² 反而较高，不能只按更晚的更新数判断全面胜出。

- 2000 的 0–0.1 m/s 桶 MMD² `0.136809`（1000 `0.097318`），拖累冻结的等支持桶平均；0.1–0.5 / 0.5–1 / 1–1.5 m/s 桶分别为 `0.093377` / `0.010854` / `0.012663`（1000 `0.100821` / `0.015156` / `0.018093`）；1.5–2 m/s 为 `0.048500`（1000 `0.043414`）。≥2 m/s 样本不足，不能宣称支持。
- 2000 的 parent-relative 关节旋转均值 `0.256404 rad`，L/R Wrist `0.490408` / `0.458698 rad`，明显低于1000的 `0.321271`、`0.866274` / `0.844557`。heading 对齐后 body position 为 `0.116580 m`（1000 `0.122856`）。
- root XY 漂移 `0.118753 m` 略高于1000 `0.115572`；yaw 误差 `0.514223 rad` 低于1000 `0.534792`。near-ground toe body-origin 滑动代理 `31.2891%`（1000 `33.4908%`），不是 native sole 接触验收。
- 2000 的失败 motion IDs 是 `[245,257,259]`；1000 为 `[20,245,257]`。相同成功率包含一次新失败与一次恢复，不能认为失败动作完全相同。

## 工件与限制

checkpoint `step_2000.pt` 为 `652,179,357` 字节，SHA `28aed874464379f0b0628719290d36af0a8b0c53691c0dd51f56102c5f428f72`。实际 AMASS / procedural 各 `8,192,000` actor-transitions；reset 数 `80,017` / `27,420`，50% transition 不代表 reset / plan 各半。CPU mmap 仅核对 checkpoint metadata，没有恢复 optimizer / PhysX。

90D pose/velocity、sigma1、256×61 reference windows、速度桶、inclusive失败帧规则保持冻结。22 validation 已反复使用，无 blind test / 多 seed 显著性或作者八场景通过声明。该协议提供真实未来 XY，是低层动作跟踪；还没有验证 mixed2000 的 learned Diffusion 闭环导航。后续导航候选登记由主任务独立决定。

[完整对照](comparison.json) / [evaluation](evaluation.json) / [分解](mixed2000_decomposition.json) / [冻结参考](heldout_reference.json) / [执行日志](execution.log) / [交付校验](delivery.json) / [三检查点证据汇总](../milestone_comparison_v1/README.md)。
