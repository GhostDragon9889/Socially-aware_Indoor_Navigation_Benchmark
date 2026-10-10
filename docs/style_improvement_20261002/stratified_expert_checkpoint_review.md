# 分层专家采样：真实训练检查点核验

核验时间：2026-10-02T06:05:39.573239+00:00。仅 CPU；未启动 Kit、GPU 或额外训练进程。

## 真实训练终态

真实训练 `status=passed`，1000 更新 / 8,192,000 物理 transitions / 32,000 控制步，耗时 1599.649 秒。训练记录 203 次跌倒重置、27,048 次时限重置；这不是零跌倒验收。

训练前封存 108 份源码；源码 archive SHA256：`e580ed9cd4114eb1cba038f3b61777564ed399c445924faa779376a837e2bb7c`。核验导入这份封存源码的 CPU 实现。

## 保存的精确检查点

| 更新 | transitions | bytes | SHA256 |
|---|---:|---:|---|
| 500 | 4,096,000 | 652,076,677 | `8b120930749ce6d5e124f7bc0fb4bfb84af3828ffa0fb0f4a43664cf89f2885a` |
| 1000 | 8,192,000 | 652,076,677 | `93c25e567d88cc4f4c639b1dea5e14494f0ef51dd304661fb23d36690e55328d` |

**第 200 次更新缺失**：初次读取时原子 `last.pt` 已变为 300；随后守候读取从 500 开始。没有将后续模型重命名为 200，也没有重建权重。watcher 的 `partial_missing_milestones` / exit 1 仅表示这个缺口。

500 与 1000 的完整格式、计数、物理状态和 AMP history、优化器 / normalizer / replay / RNG 键、采样器 config/index/probability/manifest/RNG 均通过校验。分别保存 500/1000 个采样批次和 2,048,000/4,096,000 个专家窗口。

## 实际 4+4 更新恢复试验

完整 GPU 生成的 step4 / step8 bundle 在 CPU 用封存 trainer 恢复；模型、两个优化器、四个 normalizer、完整 replay、runner payload、待恢复全局 RNG 逐项一致。私有与共享参考 RNG 的 CPU 后续抽样可确定复现。

前四更新：32,768 transitions、5 次跌倒；恢复后的四更新：累计 65,536 transitions、该段 1 次跌倒。分段相加为 6；最终 result 的 1 不是八更新累计。未验证 GPU 物理逐步等价，未在 CPU 恢复 CUDA RNG。

## 证据与限制

- [机器核验报告](stratified_expert_checkpoint_review.json)
- [完整快照 ledger](../../output/diffusion_amp/amp_style_amass2848_stratified_experts_v1/snapshots.json)
- [pilot 恢复核验](stratified_expert_pilot_resume_review.json)
- [watcher log](stratified_expert_checkpoint_watcher_remaining.log)
- [CPU evidence scripts Ruff](stratified_expert_checkpoint_ruff.log)

此报告核验保存和恢复机制，不据此声明动作风格改善、论文效果或 PhysX 接触 / solver cache 的逐位恢复。完整 22 条留出动作的风格与跟踪比较另行完成。
