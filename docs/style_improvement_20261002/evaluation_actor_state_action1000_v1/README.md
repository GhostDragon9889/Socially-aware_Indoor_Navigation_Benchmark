# 2019D checkpoint 部署兼容与 actor-only 修正

## 实际兼容性

实际 sealed state-action step1000：653,027,293 B，SHA `66e03fadd0aabf02210d7427cb3f4d86d6ecadc639395c294a70072f0b1e00e2`，1000 updates / 8,192,000 transitions。旧 state-only step1000 SHA `62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b` 也实际通过同一推理入口。

| 证据 | 2019D | 1950D |
|---|---:|---:|
| 输入批次 / 比较输出张量 | 6 / 18 | 6 / 18 |
| 新 actor 与旧 facade 输出逐字节相等 | 是 | 是 |
| 新 actor 与真实 Trainer.load(False) 输出逐字节相等 | 是 | 是 |
| 最大动作绝对差 | 0 | 0 |
| 加载后保留的 tensor 字节（含 RMS/mask） | 16,393,672 | 16,393,672 |
| 原完整网络 tensor 字节（不含 RMS） | 40,173,104 | 39,890,480 |

每份 checkpoint 的三个输入批次由实际已保存的旧 full22 首个 clip 的 post-physics 状态与 `trajectory_xy` 在 CPU 重建；批量为 1/8/32 个时间帧，映射采用该 run 明确记录的 body/DOF 名称。地形采用该隔离平地协议的显式 Z=0 sampler。**原 runner 没有保存 self/task observation tensor，所以这不是原始观测逐字节恢复证明，也不是该新 actor 的真实 GPU 运行。**另外三个批次使用私有 CPU generator 生成数学观测，尺度为 0/1/50。

旧“27”证据为 3 数学输入批次 × 3 alpha × 3 输出 tensor，并非 27 组真实保存观测。本次既未复用其作为真实物理观测，也未读取封存的 34 test。

## 已修正的推理结构

原 facade 底层保留完整 `PaperAMPNetwork`，其 `policy()` 每次还计算随后丢弃的 critic 值。新 loader 先在 CPU 构造临时完整网络，严格验证并加载 checkpoint 全部 tensor、四套 normalizer 和 log_std 的有效尺度，然后把已加载的 task encoder 与 actor 交给 `FrozenAMPActor`。临时 critic、判别器和 log_std 被丢弃。

加载后的 forward 保持原来顺序：task encoder → 与 self observation 拼接 → actor。返回的 Gaussian mean 数值不变；不构造 Normal 分布、不执行 critic 或判别器。两套观测 RMS、dtype/device 门禁、clip 与 57 有效动作 mask 保留。公开 PACER checkpoint loader、Trainer/Adam/replay 构造器在实际证明中被设为禁止调用，加载及推理仍通过。

完整 D/critic 校验没有删除：坏判别器、坏 critic、无效 log_std 等 tensor 继续拒绝。相关 inference/facade/arm 三个测试文件 **93 项 CPU 通过**，Ruff 通过。训练网络、`evaluation_actor.py`、legacy RNG helper、settings 与训练 source bundle 均未修改。

## RNG 合同与边界

`prepare_legacy_evaluation_rng(config)` 仍构造临时完整训练网络，保持旧 loader 初始化的随机数消费。实际两份 checkpoint 上，它与 `PaperAMPTrainer.load(..., restore_rng=False)` 的 Python、NumPy、Torch CPU 状态完全相等；并未恢复 checkpoint 保存的 Torch CPU RNG。纯 facade 加载和 deterministic act 保持调用者的 RNG。

安装版 Torch 的 `torch/random.py:47` 调用 CUDA `manual_seed_all`；`torch/cuda/random.py:112` 通过 `_lazy_call(..., seed_all=True)` 对 CUDA generator 延迟设 seed。实际 CPU审查记录两条路径都调用一次 seed=7，且 CUDA 从未初始化。**这只证明源码与 lazy seed 调用一致；实际已初始化 CUDA RNG、CUDA 随机抽样、Diffusion planner 与 PhysX 轨迹等价仍待真实运行验证。**

## 证据与复现

修正前完整审查保存在 [audit_before.json](audit_before.json)；修正前两模型的输入与输出保存在 [before_actor_outputs.json](before_actor_outputs.json) 及同名 NPZ。新证明见 [actor_only_audit.json](actor_only_audit.json)，源码冻结、原 helper 不变检查和测试 SHA 见 [source_freeze.json](source_freeze.json)。原 before 记录保持原样，其完整 D/critic 保留字段仅描述旧实现。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/evaluation_actor_state_action1000_v1/verify_actor_only.py
```

本证明验证兼容性与纯推理输出合同，不产生训练、闭环导航、风格改善或论文等价成绩。
