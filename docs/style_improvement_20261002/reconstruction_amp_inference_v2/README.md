# AMP 推理组件 FP32 边界修补 v2

2026-10-02。本轮只修改独立推理组件与专项测试；135 份训练 / benchmark 冻结业务源码没有修改。组件仍是 `prepared_not_runtime_integrated`，没有启动 CUDA / Kit 或新物理部署。

## 已确认问题与修补

v1 的输入 dtype 与可变 `action_mask.dtype` 比较。调用 `controller.double()` 会同时把模型与 mask 转为 FP64，从而允许 FP64 推理，违反已声明的 FP32 输入合同。[独立复现](../reconstruction_amp_inference_independent_review_v1/evidence.json) 和 [v1 冻结记录](../reconstruction_amp_inference_v1/source_freeze.json) 保留原貌。

v2 在每次 `act` 前检查：输入必须为 float32；全部模型参数 / buffers 与 mask 必须为 float32；self/task RMS buffers 必须保持 checkpoint 的 float64；全部存储必须在 mask 所在的同一个 device。手工 `double()` / `half()`、只转换 actor、`float()` 降低 RMS 精度以及部分 device 迁移会明确失败，不会静默重新转换。完整 device 迁移接口保留，但本轮没有验证 GPU。

正常加载 / 推理公式、357/1044 维观测、57/69 mask、归一化、clamp 和一次 PD 解码约定保持原行为。加载器不构造 optimizer、replay、专家采样器或训练 RNG。

## 验证

- [43 项 CPU 回归](tests.log) / [JUnit](tests.xml) / [Ruff](ruff.log) 全部通过。新增 7 项分别覆盖完整 / 部分 dtype 转换与 device 分离；错误调用不更新模型、RMS 或调用者 RNG。
- 两份不可变真实检查点：negative-backward1000，SHA `62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b`；mixed500，SHA `42939bf6a120e478541eb7ea7209d72158acb075fec30f86931341c8044d084b`。
- **每份检查点各三组输入** `(batch,scale)=(1,0),(8,1),(32,50)`，共 **6 组输入 / 18 项输出**。`actions`、`mean_actions`、`applied_actions` 与实际 `PaperAMPTrainer.load(restore_rng=False).act(deterministic=True)` 逐字节相同，最大绝对误差 0。[实际报告](actual_checkpoint_verification.json) / [日志](actual_checkpoint_verification.log) / [复跑工具](verify_actual_checkpoints.py)。输入为固定有限数学观测，未模拟新人体。
- 推理与重复调用保留 Python / NumPy / Torch CPU RNG 和全部 retained buffers。完整 trainer oracle 只在验证 harness 的显式 RNG 保存 / 恢复区域内调用。训练 bank SHA 仍只匹配 manifest 声明与大小，没有重新 hash 1.15 GB 数据。

本轮不验证 optimizer / replay / 物理恢复状态、真实 USD 地板 / 高度 / 接触、Diffusion 闭环或论文八场景效果；未读取封存34测试捕获。v1 归档与证据不覆盖。[v2 源冻结](source_freeze.json)。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python -m pytest tests/test_reconstruction_amp_inference.py -q

CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/reconstruction_amp_inference_v2/verify_actual_checkpoints.py
```
