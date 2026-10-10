# 实际 100 场景配对数组核查

2026-10-03。核查已经执行，CPU exit0，约 1.35 秒；未运行模型、重跑 DDPM、启动 SDK/GPU，未读取 sealed34。完整证据为 [报告](actual_array_peer.json)、[100 场景 CSV](all100_scene_metrics.csv) 和 [实际 CPU 闭合记录](actual_CPU_closure.json)。

## 比较身份与公平性

比较旧扩散权重 `63e7cad0…` 与新权重 `b56346fb…`，绑定实际父进程记录 `ac6a2b7f…`。13 对保存批次覆盖全部 100 个场景：`window_indices`、损失计算的 `timesteps` 和 `noise` 在 OLD/NEW 中形状、类型、数组值和原始字节均一致。

单份 `common_inputs.npz` 的全部 13 个参考／上下文字段，使用冻结的原 NumPy 数据读取器和登记索引重新读取后完全一致。两模型使用该共同样本切片；各模型预测归档没有分别复制 GT。

**随机性证据范围：**保存的是损失计算噪声及其时间步。DDPM 初始／反向采样噪声没有单独保存；相同的 `plan_seed` 有实际元数据及调用源码依据，不能声称逐字节检查了未保存的 DDPM 噪声。

## 全部场景结果

直接复用原 `offline_plan_metrics` 和 `polygon_surface_gap`，13 对原批次指标均重现；逐场景沿用同一函数和定义，未增加指标或删去碰撞样本。

| 指标 | OLD 均值 m | NEW 均值 m | 改善场景 | 退化场景 |
|---|---:|---:|---:|---:|
| ADE | 1.132818 | .852463 | 65 | 35 |
| FDE | 2.636335 | 1.879224 | 73 | 27 |

全部 100 场景和最坏碰撞保留在 CSV。最坏障碍表面间隙在 `scene_000843`（窗口 7286）由 **−2.628902 m → −2.720159 m**，穿透加重；该场景 ADE 同时改善，说明均值轨迹误差不能替代碰撞检查。NEW 最坏邻人间隙为 `scene_000898` 的 −.732448 m，该场景 OLD 为 −.405210 m。

这些是保存轨迹的离线比较，不证明真实物理闭环已安全，也不构成作者场景的复现结论。[根进程绘制的全部场景图](../orca_maps_diffusion_paired_actual_runs_v1/paired100_scene_comparison.png)仅展示上述原指标。

## 封存身份

| 文件 | SHA256 |
|---|---|
| actual_array_peer.json | `01c03c26df4e1766469968351da4043cbae6b6f03d3dd2322974850f0bdc0be8` |
| all100_scene_metrics.csv | `ca61f4d7e4690aaa2ff0cd329706cf164b7a0d98eaa75679b0d1307a3e488a80` |
| actual_CPU_closure.json | `7a9df8cc1e79adeedcb9f1e071ba120bf1c011721618c18b4bbda5e180abc7ca` |

执行源码与最终源码分别保留。最终源码只拆开三个分号语句，整体 AST 相同、Ruff exit0；没有把最终格式版冒充实际执行版本。

下一项动作风格实验建议见 [独立建议](NEXT_STYLE_EXPERIMENT.md)，没有修改训练源码或启动新训练。
