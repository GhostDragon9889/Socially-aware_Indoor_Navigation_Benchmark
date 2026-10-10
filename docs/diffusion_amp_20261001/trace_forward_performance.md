# 五状态 DDPM：公式等价的闭式引导与可选 compiled forward

## 修改边界

`CrowdSim/humanoid/diffusion/paper.py` 已将 waypoint loss 的 autograd 梯度改为同一公式的闭式实现：

\[
L=\sum_{B,S}\operatorname{mean}_{K,XY}(\mu_{XY}-g)^2,
\qquad \nabla_{\mu_{XY}}L=(\mu_{XY}-g)/K.
\]

重复 waypoint 时间索引使用 `scatter_add` 累加，匹配原 indexing backward；不能用赋值覆盖重复索引。其余三个状态通道梯度为零。没有改变 loss、引导权重、posterior variance、100 步 DDPM、FP32 Tensor、随机噪声调用数量或重规划间隔。

新增 optional API：

```python
planner = PaperDiffusionPlanner(checkpoint_path=checkpoint, device="cuda:0",
                               compile_forward=False)  # 默认普通前向
planner.set_compiled_forward(True)  # backend=inductor
assert planner.compiled_forward_enabled
planner.set_compiled_forward(False)
```

只编译 bound `model.forward`：`torch.compile(dynamic=True, mode="default")`。**没有包裹／替换 `model`**，没有 `_orig_mod` checkpoint 前缀，没有修改 dataclass config 或权重、训练入口和 checkpoint restore。不启用 `reduce-overhead` CUDA graph 模式，不切换 autocast／matmul 精度。编译懒初始化，失败会明确传播；不在用户选择编译后静默退回普通前向。

没有修改 root 的 AMP runner。本轮可由 root 在新的 runtime flag 中显式集成，保持已有 checkpoint 参数与 1000 次训练目标。

## CPU 验证

`tests/test_diffusion_paper.py`：**13 passed，10.26s**。

- 六个新增真实 autograd 比较覆盖 FP32／FP64、单／多 batch、单／多 sample、重复 waypoint 索引；不是只对照同一闭式实现。
- 完整 100 步 CPU 采样在实际 Dynamo graph capture（backend=eager）下保持结果、51 个输出时间点、全局 RNG、全部 state_dict 和 config。
- 既有真实训练／原子 checkpoint／optimizer-RNG 恢复与条件 mask 测试仍通过。
- 三个修改／新增源码文件 Ruff PASS。

## CUDA 实测：同一固定权重和全部噪声

独立纯 Torch 进程，不启动 Kit、不训练、不修改任何权重。GPU allocator 硬限总显存 **15%**，编译线程 1，进程 420 秒上限。实际测试 batch8 后 batch64；每模式完成整套 100 步 FP32 DDPM，三次计时取中位数。

固定 AMASS checkpoint SHA 始终为 `4b6f41d0f55146bd12376db1fac0806a4f1b69ae4a12bdb1293e8baada576d7f`。每组 verification 对 **100 次真实 `torch.randn` 输出逐个计算 SHA256**，旧 autograd／闭式／编译噪声全部相同，输出时间也相同。

| Batch | 原 autograd 引导 | 闭式引导 | 闭式 + Inductor | 编译相对闭式速度 |
|---|---:|---:|---:|---:|
| 8 | 0.37980s | 0.35166s | 0.17584s | 2.000× |
| 64（root 实际尺寸） | 0.46657s | 0.41611s | 0.23015s | 1.808× |

首次冷编译（含第一整套采样）54.83 秒；dynamic graph 随后复用到 batch64，首次 batch64 完整采样 0.2174 秒。batch64 峰值 allocated **745,188,864B**、reserved **876,609,536B**，均在限制内。

原 autograd→闭式：两种 batch 的 XY／velocity／yaw／全状态最大差异**全部为零**（本 benchmark 使用单个 endpoint waypoint；重复索引公式在 CPU 单独验证）。

| 编译与普通闭式的最大差异 | Batch8 | Batch64 |
|---|---:|---:|
| XY | 0.00053120m | 0.00064564m |
| world velocity | 0.00033221m/s | 0.00039196m/s |
| wrapped yaw | 0.00075912rad | 0.00252414rad（约 0.145°） |
| 52帧完整五状态 | 0.00089538 | 0.00252414 |

Inductor 浮点融合**不是 bitwise identical**。benchmark 的保守统一 `1e-3` 输出门槛对 batch8 通过，对 batch64 因 yaw 未通过。当前 root runner 仅消费 `prediction.xy`，其差异小于 1mm；是否采用 compiled flag 应在运行报告中明确这一数值边界，不能将未达到统一门槛隐藏为成功。

默认设置原样记录：autocast=False、matmul precision=`highest`、CUDA matmul TF32=False、cuDNN TF32=True。没有因编译器提示而开启新的 TF32 优化。真实结果、每次时延／内存、compile cold latency、噪声一致性与 SHA 在 [trace_compiled_forward_benchmark.json](trace_compiled_forward_benchmark.json)。

## 集成与性能判断

建议 root 的新显式 flag 为 `--compile-diffusion-forward`，构造时传 `compile_forward=args.compile_diffusion_forward`，并保存 runtime flag、Torch 版本、编译模式、最大实测数值差异和 segment timers。checkpoint 原始模型 config／权重不必改动。

先记录 planner／physics+observations／AMP reward／PPO update 的分段耗时。batch64 规划本身仅约 0.416s；32步 rollout、15步重规划通常每 update 2–3 次计划。该编译改善预计每 update 约节省 0.4–0.6 秒，**不能单凭约 1.8× planner 倍速推断当前 12–14 秒整个 update 也会倍速**。实际旧运行的 physics／observation 环节仍需要定位。

保留 100DDPM／FP32／相同 noise draws／cadence。减少 DDPM 步数、换 DDIM、半精度或更改重规划间隔不属于本轮透明优化。当前正常运行旧进程已经加载的函数不受磁盘修改影响；安全 checkpoint 后由 root 决定显式新 runtime resume。

```bash
timeout --signal=TERM 420s env TORCHINDUCTOR_COMPILE_THREADS=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
../IsaacLab-Protomotions/.venv/bin/python scripts/benchmark_paper_diffusion.py \
  --batch-sizes 8 64 --repeats 3 --cuda-memory-fraction .15
```
