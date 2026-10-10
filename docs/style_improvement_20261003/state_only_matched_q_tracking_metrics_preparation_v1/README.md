# State-only full22：薄CPU指标入口

仅准备，未读未来terminal／tracking capture，未计算新指标。实际source-only入口`--prepare-only`
在CPU退出0（chunk d9e743，0.094秒），真实加载下列原函数对象并核对源函数SHA；没有Torch／SDK／GPU。
Ruff退出0（d22d16）。没有新框架、launcher、模型或数学实现。

## 直接复用的原函数

| 指标 | 原文件／对象 | 函数源码SHA |
| --- | --- | --- |
| 原90D风格 | `scripts/evaluate_pedestrian_style.py:main` | `51c23a1d…` |
| root／yaw／body／joint旋转分解 | `docs/style_improvement_20261002/decompose_motion_tracking.py:decompose` | `43e59cc1…` |
| 原汇总表示 | `docs/style_improvement_20261002/mixed_curriculum_style_v1/compare.py:summarize` | `ca4f9036…` |

完整路径、文件／函数身份与实际CPU命令模板见[commands_and_sources.json](commands_and_sources.json)。
新入口[run_tracking_metrics.py](run_tracking_metrics.py)直接调用这些对象，不重新compile、替换函数、复制kernel或改参数。

## 相对原q指标入口的I/O差异

- 原`state_prior_stratified_CPU_metrics_preparation_v1/run_CPU_metrics.py`绑定q935／D2019及其隔离evaluation bridge；不能用于state936／D1950。
- 新入口只登记full22的90D与decomposition，执行路径是当前生产185；6个benchmark来源必须匹配原185 epoch与实际result。
- 训练来源单独是state936 manifest5c2a；准入读取未来封存的完整terminal审计，不deserialize模型。
- root实际owner须封存、`actual_exit_code=0`、`child_exit_observed=True`且`command`逐项等于原完整22argv加实际checkpoint SHA。
- 实际result与terminal SHA由root在关闭、封存之后显式传入；当前身份为null，未伪造准入。
- 原c274 heldout reference、minimum-bin30／maximum-motion256／4threads，以及原90D／sigma1／速度bins保持。
- 所有22条完整clip、11982个post-step metric帧均保留；风格导数仅用同clip合法interval，按原规则为11960，不跨reset。
- 失败、跌倒和idle不被从inclusive结果剔除；原upright子集仍是另行标注的次指标。
- 不重新计算旧q、19f或默认结果，仅登记旧q两个已有输出路径供引用。

## 未来实际执行

root提供实际owner、terminal与result的封存SHA之后，按登记模板运行一次CPU：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
docs/style_improvement_20261003/state_only_matched_q_tracking_metrics_preparation_v1/run_tracking_metrics.py \
  --terminal-sha256 ACTUAL_SEALED_TERMINAL_AUDIT_SHA \
  --result-sha256 ACTUAL_SEALED_TRACKING_RESULT_SHA \
  --owner-receipt ACTUAL_ROOT_TRACKING_GPU_OWNER_JSON \
  --owner-sha256 ACTUAL_ROOT_TRACKING_GPU_OWNER_SHA
```

独立输出`docs/style_improvement_20261003/state_only_matched_q_tracking_metrics_v1`：
原`tracking90/evaluation.json`与heldout_reference、原decomposition.json、原汇总comparison.json、实际来源sidecar。
90D原入口会opaque hash实际controller来复核结果身份；没有载入其权重或启动训练。
来源epoch绑定不冒充完整runtime import ledger，CPU执行成功不等于动作风格或论文效果达标。
无sealed34读取，GPU pilot JSON与185／936／旧指标均未修改。
