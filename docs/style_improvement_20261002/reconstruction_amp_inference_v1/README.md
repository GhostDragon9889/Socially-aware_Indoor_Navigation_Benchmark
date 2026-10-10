# Reconstruction AMP 推理加载器

2026-10-02。新增组件已准备并完成 CPU 验证；**没有接入旧 MaskedMimic 入口，没有进行新物理部署测试**。

## 接口

```python
from CrowdSim.humanoid.runtime.amp_inference import ReconstructionAMPInference

controller = ReconstructionAMPInference.from_checkpoint(
    "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt",
    device="cpu",
    expected_sha256="62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b",
)
# Caller supplies finite float32 observations with shapes [B,357]/[B,1044].
outputs = controller.act(self_obs, task_obs, deterministic=True)
applied_actions = outputs["applied_actions"]
# Future runtime owner: world.step(applied_actions), with one PD decode.
```

`actions` / `mean_actions` 是原始 69 维归一化 Gaussian 均值；`applied_actions` 为 `clamp[-1,1] × canonical mask`，冻结四个手 / 脚趾部位的 12 个坐标，57 个坐标有效。输出不是物理 DOF target，不应先调用 `action_to_pd` 后再传入 `AMPWorld.step`。此组件只支持确定性推理，拒绝 `deterministic=False`。

调用者负责把实际世界状态构建为原 PACER 357 维 self 与 1044 维 task（十个未来 XY + 1024 高度采样）。加载器不伪造观测、不操作模拟器、不选择导航目标，也不更改场景 / 地板 / 时钟。

## 加载与隔离范围

- 纯标准库 [checkpoint_contract.py](../../../CrowdSim/humanoid/runtime/checkpoint_contract.py) 提供本地文件 / LFS / ZIP 预检和 TrainingConfig 接口校验，冷导入不会导入 Torch、NumPy、Kit 或 Isaac Lab。ZIP 头本身不能验证模型架构。
- [amp_inference.py](../../../CrowdSim/humanoid/runtime/amp_inference.py) 使用 `torch.load(...,weights_only=True,map_location="cpu",mmap=True)`，验证 reconstruction format/version、357/1044/195/history10、公开 PACER 尺寸的 MLP、30 Hz、5 s、clip5、冻结57/69掩码及全部模型键 / shape / dtype / finite。拒绝加载期间被替换的文件。
- 所有四份保存的 RMS 都检查 exact keys、float64 shape、finite、非负 variance / count。实例只保留同类 `RunningNormalizer` 的 self/task 两份，使用相同 epsilon `1e-5` 与 clip5；AMP/value RMS 验证后释放。
- 复用现有 `PaperAMPNetwork`；为保持完整权重契约，网络仍包含 critic / discriminator。所有参数关闭梯度并设为 eval，没有新建优化器、replay、专家采样器或 trainer，没有恢复训练 / 物理 / RNG 状态。
- 网络随机初始化放在 CPU `torch.random.fork_rng(devices=[])` 内，完成 / 失败都恢复调用者的 CPU Torch 状态。加载和重复推理不更新 RMS，也不调用 Python / NumPy seed。GPU device 的显式迁移尚未在本轮验证；默认 CPU 不初始化 CUDA。
- `config` / `describe()` 返回副本，避免外部改动报告或配置同时改动内部行为。

Loader 仅验证推理所需模型与 RMS 的完整性，不验证 optimizer、replay、物理恢复 payload 或 AMASS 全部帧。保持相同 actor 输出不等于恢复中途物理状态，更不等于论文八场景质量通过。

## 真实检查点验证

| 不可变检查点 | 实际更新 / transition | SHA256 | CPU 输出 |
| --- | --- | --- | --- |
| negative-backward 1000 | 1000 / 8,192,000 | `62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b` | 三项动作逐字节相同，最大误差 0 |
| mixed 500 | 500 / 4,096,000 | `42939bf6a120e478541eb7ea7209d72158acb075fec30f86931341c8044d084b` | 同上 |

每个模型使用私有 CPU generator seed `20261002`，测试 batch `1/8/32`、输入 scale `0/1/50`，共六组固定有限 float32 数学观测。与实际 `PaperAMPTrainer.load(...,restore_rng=False).act(...,deterministic=True)` 对照，包括 sign-bit 的逐字节比较。输入不代表新真实行人状态，不是风格或物理效果评估。

完整 trainer load 会初始化 RNG 与训练对象，因此**只在验证 harness 的显式 RNG 保存 / 恢复范围内作为 oracle**。推理 loader 本身没有该副作用；重复动作调用、全部 retained buffers、oracle 的模型与四份 RMS 都保持原值。每个推理实例实际 Tensor 存储为 `39,913,188` 字节，此数字是张量元素计数，未测量 RSS 或 GPU 显存。

两份模型的 AMASS v7 manifest 重新计算 SHA 为 `68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`，匹配 bundle 记录。训练 bank 的 `46e024...` SHA 仅与 manifest 声明匹配并复核大小，没有重复计算 1.15 GB bank hash。模型的原训练来源 JSON、当前共享类和新增模块 SHA 一并记录。

[实际完整结果](actual_checkpoint_verification.json)、[执行日志](actual_checkpoint_verification.log)、[复跑工具](verify_actual_checkpoints.py)。此轮没有读取封存34测试捕获。

## CPU 回归与冻结

36 项专项回归通过：正常 loader / actor parity、RNG 与 RMS 不变、禁止 trainer / Adam / replay 构造、57/69有效动作与单次PD解码、完整模型 / RMS 的坏键 / shape / dtype / NaN / 负统计、原子文件替换、错误 hash / schema / 物理接口配置及输入错误。小型 unit 数据由真实 trainer 保存，明确不是实际物理训练结果。

[测试源码](../../../tests/test_reconstruction_amp_inference.py)、[pytest 日志](tests.log)、[JUnit](tests.xml)、[Ruff](ruff.log)、[源文件冻结](source_freeze.json)。旧 135 份训练业务、benchmark、trainer、模型、RMS、风格比较工具均没有修改。已有 [单 App 桥接计划](../amp_diff_runtime_bridge_v1/README.md) 的真实 USD provider / floor / native height / contact 验收仍未完成。

复跑命令在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python -m pytest -q tests/test_reconstruction_amp_inference.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/reconstruction_amp_inference_v1/verify_actual_checkpoints.py
```
