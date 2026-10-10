# 独立 AMP 推理加载器只读复核

## 结论与范围

默认 FP32 推理链路未发现阻断当前已冻结训练或评估的错误。独立 CPU 检查实际加载 negative1000 固定权重，验证加载和连续推理不改变 Python / NumPy / Torch CPU RNG，全部 retained buffers 保持不变，动作有限且重复调用逐字节相等。没有启动 CUDA / Kit，也没有修改 owner 源码或读取 34 条测试结果。

发现一个低优先级公开 API 边界缺口：手动 `actor.double()` 后可以接受 FP64 输入，与 API 声明只接受 FP32 不一致。此操作没有出现在默认加载路径；它不阻断当前 FP32 评估。独立证据见 [evidence.json](evidence.json)，可复跑程序见 [verify_boundary.py](verify_boundary.py)。

## 已核对的合同

- **归一化一致。** [amp_inference.py](../../../CrowdSim/humanoid/runtime/amp_inference.py) 的 `act` 复用训练器 `RunningNormalizer.normalize`：保存的 float64 population moments 转换至输入 dtype，epsilon 为 `1e-5`，clip 为 5；仅调用 normalize，不更新统计。全部四份 RMS 的键、形状、dtype、finite、variance/count 非负均在加载时验证；只保留 self/task 两份。
- **动作与单次 PD 解码一致。** loader 返回 Gaussian 均值及 `clamp[-1,1] × canonical mask` 的归一化 `applied_actions`，57/69 活动轴与命名 hand/toe 冻结规则对应。`AMPWorld.step` 第 265–266 行执行唯一一次 `action_to_pd`；loader 没有把输出当作物理关节目标。源合同与 owner 的单次解码单测一致；本复核没有执行物理控制器。
- **实际旧 / 新 config 可加载。** 纯 stdlib 门禁只接受 reconstruction format/version，拒绝 public PACER / MaskedMimic 权重、未知配置字段、非 canonical observation / architecture / mask / dt / horizon。旧权重缺失的新可选配置字段使用 `TrainingConfig` 原默认。owner 的真实 negative1000 与 mixed500 均已加载，未通过改名冒充另一类模型。
- **RNG 与训练对象边界明确。** 网络构造在 CPU `fork_rng(devices=[])` 内；加载没有创建 trainer / Adam / replay / expert sampler，也没有恢复训练或物理 RNG。受限 CPU mmap 反序列化仍会读取完整 bundle 对象；“没有恢复或保留 optimizer/replay”不等于“从未反序列化这些保存字段”。随后只保留模型、两份 RMS、mask 及 JSON 小元数据。
- **元数据没有冒称部署通过。** `describe()` 明确 `prepared_not_runtime_integrated`、`physical_validation_performed=False`，没有把 tensor parity 当作真实 USD / controller / 风格 / 长时域评估。完整网络仍包含未用于动作的 critic/discriminator，与严格全模型键检查一致；39,913,188 字节为张量元素大小，不是实际 RSS 或 GPU 峰值。

## 可复现的 dtype 边界缺口

位置：`CrowdSim/humanoid/runtime/amp_inference.py:151–155`。输入检查使用 `observations.dtype == self.action_mask.dtype`，而继承的 `nn.Module.double()` 会把 mask 与模型同时转为 float64。实际 fixed negative1000 实例在 `.double()` 后接受有限 FP64 `[2,357] / [2,1044]`，返回有限 FP64 动作；同一实例随后反而拒绝原始 FP32 输入，但错误仍声称需要 float32。源码文档和 `describe()` 没有显式声明禁止 dtype 转换。

建议未来 owner 在集成前选择一个明确合同：硬性要求 FP32 输入及 FP32 模型/mask，或明确禁止 `.double()/.half()/to(dtype=...)` 并 fail closed。不要据此更改正在冻结运行的默认 FP32 权重或评估。CPU 复核只转换内存实例，没有覆盖 checkpoint。

## 真实 parity 数量

独立检查 owner 的 [实际 JSON](../reconstruction_amp_inference_v1/actual_checkpoint_verification.json) 和执行 harness：两个模型各 3 组输入，分别为 `(batch,scale)=(1,0),(8,1),(32,50)`；总共 **6 组输入 × 3 输出 = 18 次输出张量比较**，每模型 9 次。所有输出 byte-identical / max-abs-error 0。不是每模型 6 组或总 36 次输出比较。36 是 owner 专项 unit test 项目数量，与实际权重的比较数量不同。

本次没有重跑两模型 oracle。新独立检查只实际加载其中 negative1000 一份，并补足 dtype 边界、默认 FP32 连续推理、RNG、buffers、mask、训练对象不保留的针对性检查。

## 源与验证边界

固定 loader SHA256：`3c8ea7f7a89a23cbd0b96d9f18b496a6599937095403aa01953b82c188386312`。loader / checkpoint gate / runtime init / owner tests 四份源码的运行前后 SHA 与字节数完全相同；精确值见 evidence。实际 negative1000 checkpoint SHA256：`62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b`，651,850,375 字节。

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/reconstruction_amp_inference_independent_review_v1/verify_boundary.py
```

实际 exit 0。没有验证 optimizer / replay / physical resume payload 的完整性，未重新计算 AMASS 全部银行 SHA，未证明任意手动 dtype 变换、未来未知 TrainingConfig 或真实仿真后端兼容。当前 loader 与已有 frozen AMP world 尚未完成新的真实接线；CPU 同值结果不能替代该物理门禁。
