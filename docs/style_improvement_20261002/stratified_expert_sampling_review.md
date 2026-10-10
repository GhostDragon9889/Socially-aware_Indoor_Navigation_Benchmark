# v7 专家分布与 full22 退步诊断

日期：2026-10-02。仅 CPU 读取冻结数据、已结束训练日志及真实物理评估，不改 v7、配置、训练器或运行入口，不启动 Kit/CUDA。完整数值、源哈希和逐动作结果见 [JSON](stratified_expert_sampling_review.json)；可重复的只读计算见 [analyze_expert_sampling.py](analyze_expert_sampling.py)。

## 1. 实际退步及方向行为

| 指标 | 旧 319 集／64 环境 | 新 v7 2,848 分段／256 环境 |
|---|---:|---:|
| full22 完整动作成功 | 6/22 | 5/22 |
| 平均 24 body 位置误差 | 0.37729 m | 0.43099 m |
| 平均 body 旋转误差 | 0.85948 rad | 1.25407 rad |
| 出现跌倒的动作 | 1/22 | 2/22 |
| full22 物理反向速度帧 | 4.59% | 19.27% |
| full22 根 yaw 偏差 >90° 帧 | 11.58% | 20.83% |
| 去除根平移后的 body 位置误差 | 0.15753 m | 0.20169 m |
| 训练实际 transitions | 2,048,000 | 8,192,000 |
| 训练 falls /100,000 transitions | 4.2480 | 5.0537 |

同一 22 个动作、相同初始姿态和平移、相同确定性完整动作门槛下，18/22 的旋转误差升高；11/22 的位置误差降低，因此不是所有动作全面变差。新模型失去 `Jog_1`（motion23）、`Walking_1`（26）的成功，新增唯一成功是明确命名的 `BackwardsWalk001`（245）。训练原始 fall 数 87→414 不能直接比较，因为新训练物理样本是旧训练的四倍。这里同时报告按样本归一化的跌倒率，未把 time limit 算作成功。

新训练 751–1000 更新的平均反向帧比例 **56.95%**，ego forward velocity **−0.15992 m/s**，direction reward **+0.22606**，trajectory reward 仍 **0.98421**。旧训练同区间分别是 **3.39%**、**+1.41237 m/s**、**+0.00563**、**0.97798**。这说明位置跟踪高分可以与反向运动、姿态退化同时出现；增加 AMP 权重或仅看 D 分数不构成姿态收敛证据。

当前 `paper_equation` 忠实实现打印公式，对负 ego forward velocity 给正奖励；论文文字则声称抑制反走，两者存在明确矛盾。[NavIsaacLab IV-B](https://arxiv.org/html/2606.26265v1#S4.SS2) 本诊断将该激励视为优先可验证的行为原因，不能从一次多变量实验确定唯一原因。root 下一轮仅切换已有 `backwards_penalty`，保留其他数据与参数，作为单变量修正消融；本报告的采样建议不混入该实验。

## 2. 整段代理类别与 D 短窗不同

实际 `AMASSCorpus._indices` 先均匀选择 manifest 中的 clip，再在合法中心 `9..frames−2` 均匀采样。10 帧 history 按 newest-first 返回，并保留 next frame。**同一个 corpus sampler 和私有 RNG 同时供应物理 reset 与 expert batch**，尚无 dataset/style/parent 分层。D 每次对已有 expert batch 再随机抽样；并不会纠正原始分布。

v7 整段类别为 turning1568 / walking720 / standing427 / side116 / running17。`turning_like` 的判断是**整段根 yaw 范围 ≥0.8 rad**，并不是 D 的 0.3 秒短 history 都在转向。下面按真实合法窗口穷举，先求每 clip 窗口占比，再按当前均匀 clip 概率精确混合：

| 真实 10 帧窗口代理 | 旧 train319 | 新 train2848 | 固定 validation22 |
|---|---:|---:|---:|
| 低速：history speed median <0.12 m/s | 14.03% | **23.90%** | 1.64% |
| 快速：median ≥1.8 m/s | 8.95% | **1.58%** | 0.20% |
| 横移：median lateral/speed >0.55 | 12.42% | 11.27% | 9.96% |
| 转向：0.3 s 净 yaw 变化 ≥0.2 rad | 30.68% | **22.50%** | 57.83% |
| 其余移动窗口 | 33.93% | 40.76% | 30.37% |
| 实际根 speed 均值 | 0.892 m/s | **0.589 m/s** | 0.901 m/s |

类别按低速→快速→横移→转向→其余的优先级互斥；这些阈值只用于诊断，不是作者动作标签。新训练平均低速增多，实际短窗快速、转向反而减少。因此不能把 55.06% 的整段 turning 类当成 55.06% 的 D 急转输入。也不能把 17 条 running 类直接改为五类各 20%，否则会反复过采样极少父源。

专家实际 backward-moving 概率是旧 **10.49%**、新 **8.52%**；新数据没有把真实反走比例放大到约57%。这是反对“只由 AMASS 新集反向数据过多导致退化”的实测证据。

validation22 为 HumanEva10 + SFU12、8 个 subject-directory 命名空间；包含 backward、DanceTurns、RunningOnBench 等旧过滤结果，不是完全纯直行测试。这里只用固定验证集观察分布与结果，不建议按这 22 条动作拟合训练概率，更不加入 heldout 专家样本。

## 3. KIT、分段与暖启动

新集实际 KIT **1233/2848=43.29%**，CMU24.54%、EyesJapan14.82%、BioTroje10.67%；旧集没有 KIT/CMU，主要是 EyesJapan34.48%、ACCAD29.15%、BioTroje28.84%。这确实构成来源变化，但 dataset 本身不是姿态质量标签，不能由来源比例证明 KIT 是错误资源。

v7 的 1853 父源：1209 个贡献1段，403个2段，131个3段，110个4段。当前均匀 segment 会让4段父源获得1段父源的4倍概率。**只改为父源均匀会更糟地偏向 KIT**：KIT 概率变为 **57.42%**，低速窗口变为 **26.06%**。原因是 KIT 的实际父源占比更大。应在窗口类别、dataset 已确定后再处理 parent 概率。

v7 把单帧概率的最大/最小比例从旧集41.31缩小到4.05，严格非重叠分段、FK、child-joint 速度一致性都是已验证的改善；不能为追求旧指标把正确 native 速度契约改回 parent 系，或删除这些证据。

两轮共享同一公开 PACER actor/critic 暖启动，但旧 D 是随机 state+action，新 D 是公开 state-only 完整暖启动。新实验还同时改变了 child-joint 速度、RMS count cap、replay、D 更新频率、AMP 权重、mirror、effective likelihood、KL、命令速度/加速度和并行样本数。所有实际配置差异已写入 JSON；这个对照不隔离 dataset 因果。

公开 PACER 论文描述约200条手工选择的 locomotion 参考动作、随机 body/terrain、30 Hz procedural 训练，并与 TRACE 独立训练；它没有证明任意2,848个分段扩大后必然保持原有风格。[Trace and Pace 正文及补充材料](https://arxiv.org/html/2304.01893v1) 作者原始 PACER 训练动作 ID 与当前 bank 的具体重叠仍未知。NavIsaacLab 的2,848作者 ID 同样未公开，本地2,848是1853实际父源的派生分段。

## 4. 后续专家采样消融建议

**先完成方向项的单变量实验。**若反走和 yaw 改善，再单独改变 D 的 expert sampler；保持 v7 文件、物理 reset 分布、奖励、网络与其他超参不变。

拟议目标先明确声明为工程实验：低速10%、快速5%、横移10%、短转25%、其他移动50%。依据训练 procedural0–2m/s 而不是验证集 ID 选取；不是作者超参或已验证最优值。先统计每类的真实支持数，不能用现有整段类别代替短窗索引。

层级顺序：**合法短窗类别 → dataset → 父源 → 该父源所有合法中心**。

- dataset 初始权重可取 `sqrt(支持该类的不同父源数)`；可选单源上限只有数学可行时才允许。无足够来源时应报 unsupported，不静默放宽。
- 在已选类别/dataset 内均匀选父源，消除因分段数1–4造成的额外父源权重。然后在其所有合法窗口中心均匀采样。
- 快速、横移等正权重桶必须有足够不同父源和合法窗口；稀疏桶明确拒绝该 profile。五类各20%不作为默认。
- 低速、方向、速度、转向支持分布都应输出；未知 metadata、跨片 history、重复/相交 parent 区间、source/parent不一致、非有限输入硬拒绝。
- 初次只改 expert，不改 reset；两种用途使用独立、可 checkpoint 的 CPU `torch.Generator`，否则专家 batch 数量会改变 reset RNG 轨迹，使配对实验难解释。
- 如果新目标 `q` 是期望风格 prior，按 q 训练就是显式目标变化。若目标是估计旧 prior `p`、仅借分层减少方差，需要在 D 损失中显式使用 `p/q`，不能宣称未加权的平衡 batch 保持原目标。
- 记录每次更新的类别/dataset/parent 实际样本数、熵、ESS、期望概率、D 的逐类 positive/logit/clip-rate，而不是只记录全局 expert_probability。稀有样本改善必须同时看完整 heldout tracking、root yaw、contact、fall、foot sliding 和实际 body style。

独立接口拟定为 `ExpertWindowSampler.from_corpus(...)`、`sample(batch)`、`describe()`、`state_dict()/load_state_dict()`，返回 `motion_ids/frame_indices/absolute_indices/probabilities/strata/datasets/parent_sources`。默认不接入现有 corpus/trainer/runner；后续 root 可显式集成。

### 必要 CPU 回归设计

1. 合成多 source、多 segment、多 dataset 的不同速率／转向轨迹，短窗分类和 history10+next 边界有可验证的数值 oracle。
2. 父源贡献1段与4段时在已选 dataset/类别内概率相同；具体 center 概率可独立相加为1。源频数不应冒充父源频数。
3. 固定 caller generator 和保存/恢复后抽样逐值一致；registry、配置、manifest hash 不同硬拒绝恢复。
4. 稀疏正权重桶、全零/负/NaN/布尔权重、metadata冲突、非30Hz、坏 quaternion、跨片区间均应明确失败。
5. 完整 v7 实际索引只读验收，报告支持数及层级理论概率；不用真实训练伪造采样收益。

## 5. 边界

这份报告提供资源分布和已完成物理结果的诊断，**不宣称数据重采样已经改善动作**。完整 source/parent heldout gate 保留。一次种子、22动作对照不提供统计显著性结论；随机身体／作者8场景／完整论文指标也没有在本次 CPU 诊断中补齐。
