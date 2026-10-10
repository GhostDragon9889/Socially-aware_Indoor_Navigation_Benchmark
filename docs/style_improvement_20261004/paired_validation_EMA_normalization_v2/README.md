# Paired validation：EMA 常量舍入兼容入口 v2（待模型执行）

本目录是原 `orca_all_temporal_paired_validation_v1/evaluate_paired.py` 的薄派生入口。历史源码、selection、registration、生产 185 文件和正式训练均不修改。

原门要求 OLD raw 与 NEW EMA 的 normalization buffers 逐 bit 相等。原 `WholeStateEMA` 对整个 state dict（包括常量）执行 FP32 blend，标量证据表明它可能引入确定的舍入差异；尚未读取实际正式 checkpoint 确认其值。

新门共享 [normalization_contract.py](../full_validation_loss_diagnostic_v1/normalization_contract.py)：OLD raw 必须等于 config FP32；NEW EMA 每个系数只能是 config FP32 或对应 checkpoint 更新数、原 `.995` 配方的模拟 FP32 值。其它值拒绝，不使用任意 `allclose`，不回写 buffer。报告保存真实值、delta、ULP 和门的来源。

派生只含四项精确字节替换：固定旧 HERE、替换 normalization 门、标记 v2 report、选择独立默认输出目录。其余 selection/loss/plan/RNG/metrics 及 77025 terminal audit＋真实 closed0 owner 门保留；逆替换必须还原全部原 22,889 B，并核对未改函数 AST 与源码。

执行前核对本入口、旧源码和共享 helper 的 SHA；同一来源在结束前再次核对。`source_registration.json` 仅登记当前源码身份，不提供未来 terminal 或效果证据。

仅检查来源（无 NumPy/Torch/权重或 selection 读取）：

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  docs/style_improvement_20261004/paired_validation_EMA_normalization_v2/evaluate_paired.py \
  --check-source-only
```

未来原正式训练真实退出、40000 terminal 审计完成之后，root 才运行 CPU paired evaluation：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/paired_validation_EMA_normalization_v2/evaluate_paired.py \
  --mode evaluate --terminal-audit-sha256 ACTUAL_SEALED_TERMINAL_AUDIT_SHA \
  --output-dir output/diffusion_amp/formal_EMA40000_vs_63e_validation_EMA_normalization_v2
```

此结果会明确记录派生入口身份；不能声称执行了原 reader 的逐 bit 相同版本，也不证明论文效果。

## 已完成的源码核验

作者 source-only 检查 `b1d239` 退出0，Ruff `a15da6` 通过；独立 peer 的只读 source-only `9cad23` 退出0，Ready／Required=[]。四项替换全文逆还原及11个未改函数源码／AST均通过。root 已核对入口、旧 reader、helper 的登记SHA并将新入口及登记封存（`c7945c`，退出0）；统一来源见[root 封存记录](../full_validation_loss_diagnostic_v1/root_source_freeze.json)。所有检查均未运行模型、读取权重或启动SDK／GPU。
