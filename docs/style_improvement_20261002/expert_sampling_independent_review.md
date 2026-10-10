# 专家窗口采样训练接入：独立 CPU 复核

日期：2026-10-02。复核范围为新增 opt-in expert-window sampler 的配置、AMPWorld 专家入口、训练/评估分支和检查点。业务代码由资源审查 agent 修改；本次复核只读业务源码，新增本报告和 CPU 证据，未启动 CUDA 或 Isaac SDK。

## 结论

**在本次 CPU 和源码检查范围内未发现阻断性回归。** 66 项针对性测试通过；独立复现另有 139 项检查通过；五个集成 Python 文件 Ruff 通过。被测试源码与报告冻结时的 SHA256 一致，详见 [JSON](expert_sampling_independent_review.json)。这不表示新专家分布已改善真实物理步态。

| 契约 | 当前实现与验证 |
| --- | --- |
| 默认关闭 | `TrainingConfig.expert_sampling=None`；AMPWorld 默认 adapter 为 `None`。关闭分支的专家 features、actions 和共享生成器状态，与直接旧入口逐 tensor 相同。旧检查点缺新增字段仍按关闭模式加载。 |
| 共享 reset RNG | enabled 每个 expert batch **先执行且丢弃一次原 `corpus.sample_reference`**。替换窗口使用独立 CPU Generator。匹配 reset/expert 调用顺序下，下一次 reset 的 IDs、帧、所有返回状态和共享 RNG 精确一致；采样本身不推进全局 Torch CPU RNG。 |
| 真正替换专家内容 | 独立 fast-only 实验的专家速度实际来自 fast 桶，并与丢弃的旧批次不同。返回 history 按 `center,center-1,...,center-9`，next pose 按 `center+1`，与实际选中 bank 张量精确相同，不越 clip 边界。 |
| 完整恢复 | 配置/seed、manifest、实际 root-feature registry、中心概率指纹、私有生成器状态及计数均保存。缺 enabled 状态、配置/registry/概率变化或矛盾计数硬失败。真实 `PaperAMPTrainer.save/load` 后 `reset→expert→reset` 的内容和共享/私有/global RNG 连续性通过。 |
| 训练/评估隔离 | 只在训练分支 configure sampler；评估不构建训练 registry，不调用 expert batch。保存发生在专家采样与成功 update 之后，恢复 global RNG 仍位于模型/runner 设置完成之后。 |
| 概率透明 | 正权重但支持不足的整个动作类别硬拒绝；不可行 dataset cap 硬拒绝。类别→dataset→parent→合法中心的条件概率与抽样频率有 CPU 测试。明确声明改变 expert prior，未应用 p/q 重要性权重。 |
| 实验独立 | 新 profile 相对 backward-penalty baseline 仅改变 `training.expert_sampling`；初始化、manifest 和其余训练字段相同。resume 显式比较 profile 请求与 saved sampler 配置，不能用旧检查点静默开启或更改新分布。 |

## 关键代码位置

- [纯配置与默认值](../../CrowdSim/humanoid/application/settings.py) 实际仓库路径为 `CrowdSim/humanoid/application/settings.py:8,114,118`；验证不导入 Torch/Kit。
- `CrowdSim/humanoid/amp_world.py:90,170,197,209,219,223`：原 reset 入口保持直接 corpus 调用；只有 expert 入口使用可选 adapter。
- `CrowdSim/humanoid/training/expert_sampling.py:382,458,465,495,518,529,537,541`：稀有桶门禁、完整私有状态、一次原调用、独立替换与 CPU RNG 恢复。
- `scripts/train_paper_amp.py:332,336,492,539,540,710,714,742`：训练专属配置、resume 配置匹配、双 RNG snapshot/restore、成功 update 后保存及评估安全报告。
- `configs/diffusion_amp/amp_style_amass_stratified_experts.yaml:68`：明确的新 prior；`tests/test_expert_sampling_integration.py:241` 检查相对 baseline 仅这一训练字段变化。

## 验证证据与复跑

1. [针对性 pytest XML](expert_sampling_independent_cpu_tests.xml)：66 passed，1.75 秒。含真实小 CPU PPO/D update、Adam 状态保存/恢复、非法配置、旧检查点兼容和 Torch/SDK-free 配置子进程检查。
2. [独立逐项结果](expert_sampling_independent_reproduction.json)：139 checks，0.66 秒。使用真实 `AMASSCorpus` 和真实 `PaperAMPTrainer` 序列化；从当前源码 AST 提取 AMPWorld 的四个 expert 方法，避免导入 SDK。fixture 为有限、已知五类 root 运动的小型合成 bank，**不是物理仿真**。
3. [独立复现脚本](audit_expert_sampling_integration.py)。在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
../IsaacLab-Protomotions/.venv/bin/python -m pytest \
  tests/test_expert_window_sampling.py tests/test_expert_sampling_integration.py -q

CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/audit_expert_sampling_integration.py \
  /tmp/expert_sampling_independent_reproduction.json
```

## 尚未验证与明确限制

- RNG 配对仅保证**相同调用顺序**的随机序列；新策略可能改变真实跌倒/reset 时间，因此不保证两个完整物理训练轨迹相同。
- 新分布有意改变 discriminator 的 expert prior。没有 p/q 校正，也没有据此证明步态、导航、足底滑移或 heldout tracking 改善；需要独立的新训练与同协议物理评估。
- minimum parent/window 门槛针对整个正权重动作类别，**不是每一个类别/dataset 子组**。某些 dataset 的 rare 子组可能很小，支持表必须继续公开，不能把全类别门槛解释成每个 dataset 均足够丰富。
- sampler registry 实际哈希 root 分类特征、选中 IDs/中心和概率；完整 bank SHA 仅从 manifest 声明读取并检查文件大小，**没有在这里重算全部内容 SHA**。不可把 sampler 成功构建当成完整 bank 完整性验证。
- 本次没有测试 CUDA `map_location`、GPU 内存峰值、真实 AMPWorld reset、PhysX solver cache 或 bitwise 物理恢复。加载器的 GPU RNG tensor `.cpu()` 转换仅经源码检查；真实 CPU 保存/恢复已实测。
- 每个 enabled expert batch 保留一次被丢弃的旧 CPU/device gather，有额外时间和短期内存代价。这是配对实验控制要求，不能声称开销为零；runner 已保留专家采样计时。
- 早先 `expert_window_sampling.md` 的“旧 corpus RNG 不变”是**独立 sampler** 的验收范围；此处训练 adapter 会按原协议消耗 corpus RNG，两份结果并不矛盾。

原权重、历史物理实验和既有 ground-support 采集源码均未由本次复核改写。
