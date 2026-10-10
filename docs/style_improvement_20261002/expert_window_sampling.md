# 独立短窗口专家采样器：CPU 验收

日期：2026-10-02。新增 [expert_sampling.py](../../CrowdSim/humanoid/training/expert_sampling.py) 与 [tests](../../tests/test_expert_window_sampling.py)，**未接入现有 corpus、world、trainer 或 runner**。正在运行的方向项消融不导入本模块。v7 bank、manifest、生成器和配置均未修改。本模块通过资源/概率契约验收，不代表训练或物理姿态改善。

根因背景见 [分布与 full22 诊断](stratified_expert_sampling_review.md)。实际 v7 门禁见 [verification JSON](expert_window_sampling_v7_verification.json)、[CPU log](expert_window_sampling_v7_verification.log)；46 个 CPU 测试全部通过，见 [test log](expert_window_sampling_tests.log)、[JUnit](expert_window_sampling_tests.xml)。[Ruff](expert_window_sampling_ruff.log) 全部通过。

## API

```python
import torch
from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.training.expert_sampling import ExpertWindowSampler

corpus = AMASSCorpus(
    "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json",
    device="cpu", dof_velocity_frame="child_joint",
)
sampler = ExpertWindowSampler.from_corpus(
    corpus,
    target_weights={
        "low_speed": .10, "fast": .05, "sideways": .10,
        "turning": .25, "moving": .50,
    },
    generator=torch.Generator(device="cpu").manual_seed(20261002),
    dataset_weighting="sqrt_parent_support",
    dataset_cap=None,
    minimum_parents_per_bucket=5,
    minimum_windows_per_bucket=100,
)
sample = sampler.sample(4096)
history_state = corpus.state_at(sample.history_indices.flatten())
next_position = corpus.data["dps"][sample.next_indices]
provenance = sampler.describe()  # JSON safe; includes exact expected masses
resume_state = sampler.state_dict()
# Explicit caller integration would save resume_state in its full checkpoint.
sampler.load_state_dict(resume_state)
```

返回 CPU `motion_ids/frame_indices/absolute_indices/history_indices/next_indices/probabilities`，以及每条 draw 的 `strata/datasets/parent_sources`。history 始终10帧、newest-first；默认保留1个未来帧，next_indices 是实际中心的 next frame。调用者按既有模型 mapping 构建195维 AMP 观测；本模块不构建假的观测或动作。

数值分类按完整10帧 history 的速度中位数、横向速度比例中位数和 unwrapped 净 yaw 变化，优先级是低速→快速→横移→短转→其他移动。默认阈值分别为0.12m/s、1.8m/s、0.55比例、0.2rad/0.3s，全部明确写入配置指纹，**不是作者 semantic label 或最优超参**。整片 yaw-span 标签不参与采样分类。

层级概率为 `q(stratum) * q(dataset|stratum) / supported_parents / legal_centers_of_parent`。dataset 默认按该类别不同父源支持数的平方根加权；父源无论贡献1段或4段，在已选类别/dataset内有相同概率。每个 parent 的所有合法中心合并后均匀抽样。可选 dataset_cap 只有数学可行时接受，否则明确失败，不放宽或无限 rejection。

正权重短窗桶未达到 minimum_parents/windows 门槛时明确 unsupported；未知权重类别、非有限/负/布尔权重、非30Hz、坏 root quaternion、源与 bank 不一致、parent metadata冲突/重叠、跨 heldout parent 均硬拒绝。构建索引不消耗 RNG；调用者应像示例一样提供独立专家 Generator，才能保持旧 corpus/reset RNG 不变。显式复用同一个 Generator 会共享序列，不推荐。caller-supplied Generator 的完整状态、计数、manifest/config/registry 指纹必须由未来调用者显式 checkpoint；指纹不一致拒绝恢复。

## 实际 v7 支持

冻结 source manifest SHA：`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`。2,848段 /1,853父源，**494,792个合法窗口中心**。

| 短窗类别 | 实际合法中心 | 不同父源 | 本次显式目标 | 32,768 draws 实测 |
|---|---:|---:|---:|---:|
| low_speed | 110,378 | 1,674 | 10% | 10.147% |
| fast | 6,063 | 168 | 5% | 4.926% |
| sideways | 62,693 | 1,412 | 10% | 10.062% |
| turning | 114,472 | 1,441 | 25% | 24.973% |
| moving | 201,186 | 1,765 | 50% | 49.893% |

本次抽样覆盖 **1,847** 个真实父源；保存/恢复后32,768次抽样的 IDs、中心、history、next、概率和 metadata 逐值一致。全部边界合法，旧 corpus RNG 原样保持。构建、抽样和重复验收总 CPU 耗时约 **1.49s**，该时间是本机单次验证，不是训练性能保证。

本次 cap=None。理论 dataset 概率为 KIT34.839%、CMU20.392%、BioTroje13.014%、EyesJapan12.655%、ACCAD8.640%、EKUT6.219%、TotalCapture3.840%、TCD0.400%；KIT 当前原 sampler 为43.294%。这是一项可单独评估的 prior 变化，没有证明来源比例本身导致 full22 退步。

该层级仍有需要披露的局部偏重：fast 整体支持168父源，但一些 dataset 子组仅有2父源或21个短窗。默认门槛作用于**正权重类别总支持**，不承诺所有 dataset 子组都达到5父源/100窗。最大理论单父源概率 **0.4582%**，最大单中心概率 **0.00073791**，窗口分布 ESS约 **73,696**（合法中心共494,792）。因此不能称为全窗口均匀或消除所有稀疏性；未来训练消融应记录逐组 D 指标和重复度，再决定更严的子组准入，不能据抽样门禁直接宣称泛化收益。

registry SHA：`d83afe339bb0792bb17c19a258301496714eb26ec014dd29800edbc5b6b766ff`。完整根特征数组实际计算哈希；大 bank SHA 使用冻结 v7 完整性证明的声明，本模块没有重复读取整份1.15GB文件来重新计算。metadata、实际 motion_files/count/dt、bank大小、合法区间和所用根特征均直接检查。

## 集成与结论边界

建议 root 先完成方向项单变量训练与固定 full22 门槛，然后再运行**专家采样唯一变化**的独立实验；初次保留旧 reset sampler。新 prior q 会改变 AMP 的“真实样本”分布。如果要估计旧 prior p 而只利用平衡 batch 降方差，训练器还需显式 p/q 重要性权重，本模块没有默默修改 D 损失。

CPU 门禁覆盖短窗数值／yaw-wrap、完整 history+next、1段与4段父源概率、dataset层级与 cap、真实抽样频数、caller RNG、完整checkpoint恢复、metadata/heldout/源控制区间/原50Hz依赖区间/实际bank区间冲突、非有限和布尔类型。没有 Kit/GPU、新训练、数据改写、作者动作ID等价或论文效果声明。
