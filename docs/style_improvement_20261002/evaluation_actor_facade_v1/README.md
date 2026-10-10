# 评估 actor 门面与显式旧 RNG 边界

状态：独立组件已准备，本文的 CPU 验证没有推进物理或启动 Kit。原训练、世界、主循环及 v2 推理模块没有由本次修改。

## 接口

```python
from CrowdSim.humanoid.runtime.evaluation_actor import (
    EvaluationActor, prepare_legacy_evaluation_rng,
)

actor = EvaluationActor.from_checkpoint(path, device="cpu", expected_sha256=sha)
# 仅在兼容旧 Trainer.load(restore_rng=False) 初始化边界时明确调用一次，
# 必须在无显式种子的 Diffusion planner 构造/采样之前；普通推理不调用它。
rng_report = prepare_legacy_evaluation_rng(actor.config)
outputs = actor.act(self_obs_357, task_obs_1044, deterministic=True)
applied = outputs["applied_actions"]  # normalized69，交给 AMPWorld.step 解码一次
```

`act` 只返回 `actions`、`mean_actions`、`applied_actions`。没有训练 log probability、value、AMP reward、优化器或 replay 接口；评估主循环应跳过训练 transition 收集和 D/critic 调用，不能补伪零值。

`config`、`provenance`、`describe()` 是独立副本。完整训练 provenance 经过有限 JSON 校验并限制为 2MiB；计数 `update_count`、`physical_transition_count` 和顶层 checkpoint SHA 来自选中的 baseline。评估阶段需要修改自己的普通报告字典，不应修改返回副本并期望其写回模型。

```python
actor = EvaluationActor.from_checkpoints(
    base_checkpoint, style_checkpoint, alpha=0., device="cpu",
    base_sha256=base_sha, style_sha256=style_sha,
)
```

复合载入复用现有 arm 模块的两个 NN，没有第三个 baseline NN。baseline 元数据保持原样；`describe()["experimental_arm_style"]` 记录双 SHA、24 个上肢轴、alpha、共享物理契约及实验限制。没有 style 时非零 alpha/非空 style SHA 被拒绝。非零上肢混合可能改变平衡和接触，不构成论文单 actor 或安全保证。

## RNG 合同

门面加载和 `act` 保持调用方 Python、NumPy、Torch CPU RNG。旧 Trainer 初始化会先用训练 seed 重置三种全局 RNG，再初始化 CPU PaperAMPNetwork；旧 loader 的 `restore_rng=False` 保留这一初始化后的状态。替换为隔离 RNG 的 v2 loader 本身不会复现旧全局采样序列。

`prepare_legacy_evaluation_rng(config)` 验证配置及默认 CPU/FP32 环境后，显式重置三个 seed，仅构造并删除临时 PaperAMPNetwork 消费相同初始化随机数。它不构造 Trainer、优化器或 replay，也不恢复 artifact 的训练 RNG。`torch.manual_seed` 保持正常 CUDA seed/lazy-seed 行为；本次 CPU 检查没有验证 CUDA 状态或 planner/物理轨迹等价，需单独真实运行验证。

## 实际验证

- [19 项 CPU 测试](tests.log) / [JUnit](tests.xml)：冷导入无 Torch/NumPy/SDK，有限 JSON、别名隔离、实际旧 loader RNG、错误配置前置拒绝、复合 NN 数量、baseline 元数据和输出合同。
- [真实工件 CPU 结果](actual_cpu_verification.json)：negative1000（1,000 / 8,192,000）与 mixed500（500 / 4,096,000）。两份 checkpoint 各三组 `(batch, input_seed)=(1,0),(8,1),(32,50)`，共 **6 输入批次 / 18 输出张量**，逐字节一致，最大误差 0。两工件 helper 后 Python/NumPy/Torch CPU 状态与实际 `PaperAMPTrainer.load(..., restore_rng=False)` 全一致。
- 实际双工件 alpha0 额外 1 组输入 / 3 输出逐字节等于 baseline；完整 baseline provenance 和计数一致。该额外检查没有混入上述 6/18 数量。
- [可复跑验证工具](verify_actual_rng.py) 仅 CPU，固定数学输入，没有真实物理输入或 rollout；TorchScript 弃用与既有转义警告记录在日志中，没有 SDK 初始化。

载入两次 restricted CPU/mmap：v2 验证完整模型/RMS/文件 SHA，再提取完整小 JSON provenance；两次 config/count 对齐且选中文件 stat 保持不变。门面不保留训练 optimizer、replay、runner state 或 artifact RNG 张量。v2 会保留网络结构中未被门面暴露的 critic/D 权重，不能把这种结构存储说成创建训练器。
