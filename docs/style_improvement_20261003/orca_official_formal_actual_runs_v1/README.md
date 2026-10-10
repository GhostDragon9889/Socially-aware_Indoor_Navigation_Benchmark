# 官方 ORCA 两域 Diffusion：实际显存与恢复验证

使用本地按官方流程生成、全量检查并封存的 Maps／Interact 数据，共 228,588 个训练窗口。独立 Nav 五状态模型保留原网络与 loss；未知地图以 TRACE 的 .5 raster 编码，训练保存原始模型、EMA、Adam、完整 RNG 与两域采样状态。未替换默认部署权重。

| 实际运行 | 结果 |
| --- | --- |
| microbatch400 × accumulation1 | session33525，`c88759`→`5a99cf`，实际退出1；第一轮训练地图编码 CUDA OOM，失败日志与配置保留 |
| microbatch200 × accumulation2 | session40507，`ee42f4`→`5a4d53`，实际退出0；4更新／8microsteps／1,600窗口，PyTorch峰分配8.160GiB、峰保留10.733GiB |
| 从真实GPU step4恢复一轮 | session75232，`bd04d8`→`7f61e5`，实际退出0；原完整恢复函数包含CUDA RNG，计数4→5／10microsteps／2,000窗口；未声称配对物理或逐位续训 |

四轮试训训练阶段耗时14.506秒，含验证与保存；第3至第4更新间隔约.993秒，不能作为40k完整训练耗时保证。四轮和恢复均明确省略DDPM轨迹采样，不是ADE／碰撞或物理效果证明。EMA尚处于前2,000更新的复制预热期。

正式候选选择200×2（effective400）；BatchNorm每次训练只使用200，区别于官方配置的整批400。40,000更新对应1,600万窗口暴露，计划从原本地AMASS初始化 `4b6…` 独立重新开始，不接续四轮／五轮试训权重。正式训练目前尚未运行；终态和闭环效果仍需验证。

[400失败记录](pilot4_micro400_accum1_actual_root_owner.json) · [200×2成功记录](pilot4_micro200_accum2_actual_root_owner.json) · [GPU完整恢复记录](resume5_micro200_accum2_actual_root_owner.json) · [恢复登记命令](resume5_registered_command.json)。各产物是独立文件、实际退出后只读封存；不声称作者数据、模型或论文八场景协议已经匹配。
