# 2848 专家采样：只读研究与下一轮选择

日期：2026-10-03。仅审阅源码、现有 train2848 的冻结 CPU 覆盖证明及作者公开资料；未改 bank/manifest、val22/test34、冻结源或默认值，未读取 sealed34 结果，未运行 GPU/Kit/模型。所有训练效果建议为未来实验。

## 本轮结论

**下一轮先做已由 root 选定的 AMP reward 系数单变量对照；`expert_sampling` 保持 `null`。** 采样分层可作为随后独立实验，现有代码已经支持，不需要新实现。不能把外部权重与采样同时改变，或仅因 MMD 下降而断言 mode collapse 已解除。

当前 `AMASSCorpus._indices` 使用均匀 clip ID，再在其合法中心中均匀选帧，忽略 bank 的 `motion_weights`；专家与 reset 都由同一 corpus CPU Generator 供样。十帧 newest-first history，另保留下一帧；clip N 帧有 N−10 个合法中心。理想概率为 `p(i,t)=1/(2848*(N_i-10))`（实际 float32 随机数离散化仍按源码执行）。本轮 state-prior profile 明确 `expert_sampling: null`。

## 覆盖事实与推断边界

沿用冻结只读 v7 证明，2848 段来自 1853 个父动作、200 个 subject namespace；644 个父源贡献多段，最多4段。KIT 占43.2935%，CMU24.5435%。uniform clip 的父源有效样本数 ESS≈1408.18；均匀父源并不自动去掉 dataset 偏置，已有计算会使 KIT 升至57.4204%。这些 namespace 不是已经核验的真人身份。

| 真实十帧运动学代理 | 当前 uniform-clip 概率 | 已有分层配置目标 | 合法窗口 / 支持父源 |
| --- | ---: | ---: | ---: |
| low_speed | 23.8952% | 10% | 110378 / 1674 |
| fast | 1.5801% | 5% | 6063 / 168 |
| sideways | 11.2666% | 10% | 62693 / 1412 |
| turning | 22.5018% | 25% | 114472 / 1441 |
| moving | 40.7564% | 50% | 201186 / 1765 |

表中当前概率是逐 clip 窗口比例再按 uniform clip 加权；不能用全部中心中 fast=1.2254% 替代。整段 `turning_like`=55.06% 也不是55.06%的短窗在转身。分类优先级为 low_speed > fast > sideways > turning > moving；阈值分别根速度中位数<.12、≥1.8m/s、侧向比例>.55、十帧净 yaw≥.2rad。它们是互斥运动学代理，不是作者语义类别、真 idle、跑步足部周期或风格ID。standing427中180个 push 名称混类已在旧覆盖审计登记；本次不删除它们。

已有分层先验将 KIT 理论概率降为34.8392%，但它同时重排类别、dataset、父源和中心概率。**其父源 ESS≈1179.53、中心 ESS≈73696.15，低于当前1408.18/425972.12；不能称这套 prior 自动增加统计覆盖。** 小子组在 uniform-parent 后可能获得高中心权重，最大单中心概率≈0.000737906。fast 的 TotalCapture 子组仅2父源/21窗口；minimum支持门槛针对整个类别，并非每个 dataset 子组。重复抽样不能制造新风格。

当前 native15 固定45D覆盖 .70/.61，对旧 .76/.61；90D和45D MMD下降、entropy上升，同时路线顶点211→160。这是混合证据，尚不足以诊断 GAN 的 mode collapse；速度任务/PD代理、公共初始化适配、源分布和优化都可能影响姿态占用。需在固定 basis 下一起看 occupied clusters、速度支持桶、轨迹成功/到达、跌倒与接触；保留失败、idle及未知速度桶。

## 已有配置和真实加载/恢复合同

`TrainingConfig.expert_sampling` 默认 `None`，唯一已实现 mode 是 `stratified_windows`。规范化配置必须显式给 mode、seed及全部五个 target_weights；dataset_weighting 可取 `sqrt_parent_support` 或 `uniform`，dataset_cap 是每个 stratum 内的 dataset cap，**不是全局 dataset cap**。`ExpertWindowSampler` 要求 train split、真实30Hz bank、十帧 history+一帧future、parent split隔离及合法边界；实际分类使用 gvs/grs，而非整段 style_class。构建检查 bank stat/声明SHA，另hash实际root分类特征；它不替代整个bank SHA重算。

现成配置 `configs/diffusion_amp/amp_style_amass_stratified_experts.yaml` 的 mapping 可用，但该完整 profile 是旧 backward-penalty baseline，**不能直接替代本轮 state-prior/protected profile**。若后续获准，只将其中 `training.expert_sampling` mapping 加到新 state-prior 对照 profile，其他字段逐项相同。其层级是 stratum → dataset sqrt(parent支持数) → uniform parent → uniform该父源的该类合法中心；无 p/q importance correction，D学习目标prior有意改变。

训练在首次 reset 后、训练分支中 configure adapter，评估不建registry、不抽专家。enabled 每个专家batch先真实调用并丢弃一次旧 `corpus.sample_reference`，保持匹配调用序列下的共享 corpus RNG消费，然后用独立 CPU Generator(seed7)抽替代窗口；不是估计同样draw数量。额外旧 CPU/device gather 开销计入 `expert_sampling_wall_s`。相同调用序列下下一次reset输入/RNG可精确匹配；不同策略的跌倒/reset时机可能分岔，不能保证完整物理rollout bitwise相同。

runner当前每update抽 `min(rollout_steps*num_envs,4096)` 专家，256×32→4096；2000update共8192000实际专家窗口。保存发生在成功 update 后，runner_state保存共享corpus_rng与独立expert_sampling，后者含完整配置/seed、registry/manifest/config/中心概率指纹、private RNG和batch/window计数。resume先比较请求profile与saved sampler配置，重建registry后逐项恢复；enabled缺状态、改变seed/probability/source/config硬失败，无静默重新播种。不能从本轮disabled checkpoint直接开启后续prior续训并声称fresh单变量对照。GPU map-location的RNG先转CPU。

## 作者资料能支持什么

1. AMP论文明确把 mode collapse 列为大而多样的动作集中的局限；并未保证匹配全部模式。支持“采样/覆盖值得独立消融”，不证明本项目低占用原因。[AMP论文，第9节](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)
2. 作者ASE代码以显式motion_weights多项式抽clip、再抽时间。其技能覆盖还依赖 latent-conditioned actor、encoder reward和可选动作diversity loss；不能把五桶采样称为ASE等价。[MotionLib](https://github.com/nv-tlabs/ASE/blob/main/ase/utils/motion_lib.py#L92)、[ASEAgent](https://github.com/nv-tlabs/ASE/blob/main/ase/learning/ase_agent.py#L358)
3. 作者现维护的MimicKit同样支持dataset YAML的逐clip显式权重并规范化，权重不是自动按时长决定。[MotionLib](https://github.com/xbpeng/MimicKit/blob/main/mimickit/anim/motion_lib.py#L27)
4. ADD作者代码把参考和实际观测的误差向量作为负例、零误差作为正例；这是tracking/MOO目标构造，不是多风格覆盖采样修复。若引入ADD，将是不同机制和实现实验。[作者项目](https://add-moo.github.io/)、[ADDAgent](https://github.com/xbpeng/MimicKit/blob/main/mimickit/learning/add_agent.py#L19)
5. ProtoMotions MotionManager支持motion weights；Mimic evaluator按训练追踪失败重加权并保存manager weights，当前实现保留min weight避免只追最近失败的少量clip。本项目 `train_paper_amp` 直接走AMASSCorpus/AMPWorld，未调用这条manager/evaluator训练采样路径，不能只改上游motion_weights并假定专家prior会改变。[MotionManager](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/envs/motion_manager/motion_manager.py)、[MimicEvaluator](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/agents/evaluators/mimic_evaluator.py#L90)

公开作者页面/代码本次于2026-10-03读取；main分支是当前观察，未把它假称论文历史commit。没有找到NavIsaacLab作者2848 clip IDs、类别质量分布或权重，现有v7仍是重建库。

## 推荐顺序

1. 当前下一轮：仅AMP外部权重2→.5，expert_sampling仍null；与公共state-priorfresh2000同预算、同固定评估比较。
2. 若该轮后仍需要专家先验消融：fresh公共coupled初始化，仅开启既有 mapping（.10/.05/.10/.25/.50、sqrt_parent_support、无cap、seed7），全程同2000/16.384M预算。研究假设是减少低速专家质量、提高moving/fast曝光并限制clip切片重复影响；不是作者比例复原。它是“专家prior机制”单变量，但内部含多个层级，改善不能单独归因于fast权重或parent均衡。
3. 在已enabled的同一prior基线形成后，若要更严格的单标量研究，可只改某一target weight或dataset_weighting，其他mapping完全不变；不要同时改阈值、bank和奖励。当前不实现新的speed-conditioned D、p/q校正、online失败重采样或ASE/ADD架构。

复用既有104/66 focused测试和真实v7 draw/RNG证明；本次只做静态读取、概率摘要重算与源hash，不扩大测试或建立新的训练准入框架。证据记录见 `evidence.json`，所有future training_effect=false。
