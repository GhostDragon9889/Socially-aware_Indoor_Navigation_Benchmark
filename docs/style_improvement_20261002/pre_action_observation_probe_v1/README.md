# 策略调用前观测探针（待合入，CPU 证据）

此补丁只增加三个文件，不编辑既有 deployment、driver、world、训练或权重。源码174的全部选定文件作为合并前保护条件；该集合不是完整运行导入账本。

## 观测问题与合同

1. 真实策略调用前的1044维任务输入，有多少归一化值被裁剪？记录原输入、实际 task normalizer 的未裁剪和裁剪后值，区分20维轨迹与1024维地形后缀。
2. 每条样本对应哪些实际 RMS moments？记录均值/方差/count 的 dtype、逐数组 SHA 和组合 SHA。
3. 诊断是否影响策略输出和随机数？原 actor.act 仅调用一次，原输入对象传入，原返回字典与张量对象直接返回。选定调用额外执行两次纯 normalizer forward，**没有额外模型 forward**。
4. SDK 快速退出是否丢失证据？每个选定成功调用独立原子发布 NPZ，随后 fsync JSONL；真实子进程 os._exit 的 CPU 测试仍能读取样本。

Raw 输入/moments 的 clone 和 z 计算发生在原 act 前；原 act 返回后发布样本，发布仍在既有 driver 的 world.step 前。call_index 是零起始策略调用计数，不冒充 native physics counter 或引擎时间。原 driver 的 AST 合同测试确认 act 在 step 前且两者各一次。

默认每30次调用保存一次（0,30,60,…）。只允许明确的 --inference-only、确定性单个 baseline EvaluationActor；拒绝 arm blend。NPZ 与 JSONL 不声明运行完成，必须另读真实 result/status。数据是仿真数值张量，不含账号或凭据。

## 验证

- 16 项 CPU 合同测试通过；输入/RMS/Python、NumPy、Torch RNG 不变，原输出对象身份保留，冷导入不启动 SDK/Torch，已有 SDK /arm experiment 入口拒绝。
- 一份真实 negative-step1000 权重、同一个已加载 CPU actor：基线调用一次、观测调用一次，三个输出逐字节相等、最大差0；无 optimizer/replay/App/Context/GPU。
- 真实权重测试使用合成 CPU 输入，**不是物理观测或闭环等价证明**。
- 新三个源文件 Ruff 通过；merge helper 只做 dry-run，不由作者执行 --apply。

合入后实际30秒基准待 root 单独运行。当前源码174与历史 source168 不同，不能提前宣称物理数组逐字节相同。额外 CPU IO/GPU→CPU copy 会影响墙钟时间，不能保证 PhysX/Warp bitwise。裁剪观测本身不证明闲置/失败的因果。

## 可审补丁及调用

`proposed.patch` / `merge_manifest.json` 绑定精确三个新增目标与每项字节/SHA；`merge_observer_proposal.py` 默认 dry-run。每文件硬上限50000B、总上限100000B，exclusive 发布不得覆盖已存在文件。仅 root 复核后合入。

```bash
python docs/style_improvement_20261002/pre_action_observation_probe_v1/merge_observer_proposal.py
```

合入后的显式实际运行参数（当前仅准备，未由作者执行）：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python scripts/probe_paper_pre_action.py \
  --amp-checkpoint output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt \
  --config configs/diffusion_amp/amp_style_amass_backward_penalty.yaml \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --scene-config configs/diffusion_amp/shared_obstacle_timed_validation.yaml \
  --actors 4 --duration-seconds 30 --seed 7 --inference-only \
  --observation-stride 30 --output output/diffusion_amp/pre_action_baseline_cuboid_30s_v1
```

新 launcher 仅委托既有 deployment，使用同一 driver/actor factory/控制 loop/world。实际加载的新增 helper 与 script 路径/字节/SHA 写入 pre_action_observer.json，不声称完整导入账本。CPU actual proof 及 JUnit 见 actual_CPU_final/parity.json、tests.xml；样本逐项落盘证据见各 samples.jsonl。
