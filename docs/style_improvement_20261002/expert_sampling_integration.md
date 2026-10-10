# 分层专家窗口的可选训练接入

2026-10-02，代码及 CPU 门禁已冻结；本次未启动 CUDA/Kit 或训练，没有修改既有模型、v7 manifest/bank、评估器及主指南。该实验尚未证明动作质量提升。

## 唯一训练变化

新配置 [amp_style_amass_stratified_experts.yaml](../../configs/diffusion_amp/amp_style_amass_stratified_experts.yaml) 以 negative-backward 配置为基准，`TrainingConfig` 仅新增启用 `expert_sampling`。奖励、权重初始化、数据、网络、PPO/判别器优化参数、动作解码和 curriculum 均保持原值。

原有 `expert_sampling=None` 是默认值，旧 checkpoint 缺少该字段仍解释为关闭。默认路径继续调用原 `corpus.sample_reference`；没有新增默认 runner-state 键。评估不构建训练索引、不加载训练专家库、不执行额外 sampler 调用。训练保存的 expert 配置不成为部署运行必需资源。

## 随机数与实际样本

reset 与 reference 路径未改。启用后的每个专家批次先**实际执行一次原 `corpus.sample_reference(batch, history_frames)`，丢弃其返回值**，再使用独立 CPU `torch.Generator(seed=7)` 采样新专家窗口。没有用相同 draw 数量或估计来替代原调用。

这保持了相同调用顺序、batch 大小下原共享 corpus RNG 的状态和后续 reset 输入。额外旧 CPU/device 状态 gather 是明确的配对实验成本；会与原采样时间一起记录在 `expert_sampling_wall_s`。策略变化引起的跌倒/reset 时机、batch 大小和实际随机数消费可能分岔，不能声称全过程真实训练 rollouts 相同。

新专家 payload 直接从已选窗口的 `history_indices`、`next_indices` 读取原 corpus，history 为 newest-first 十帧，另保留一帧 future；所有状态、动作和 child-joint 速度保持原加载/解码合同。只有专家先验改变，不作 p/q importance correction。

层级为：真实短窗类别 → dataset（sqrt(parent support)）→ uniform parent → uniform legal window。不是把五个全 clip 标签强制各设 20%。本轮明确先验为：

| 短窗类别 | 目标概率 | v7 合法窗口 | 支持父源 | 实际 32,768 次抽样 |
| --- | ---: | ---: | ---: | ---: |
| low_speed | 0.10 | 110,378 | 1,674 | 3,224 |
| fast | 0.05 | 6,063 | 168 | 1,667 |
| sideways | 0.10 | 62,693 | 1,412 | 3,311 |
| turning | 0.25 | 114,472 | 1,441 | 8,156 |
| moving | 0.50 | 201,186 | 1,765 | 16,410 |

共 494,792 个真实合法中心；CPU 建索引 1.270 s。探针覆盖 1,848 个父源。KIT 的理论专家概率约 34.8392%，实抽 34.8358%；没有硬性全局 dataset cap。阳性类别需至少 5 父源、100 窗口；缺支持时硬失败，不自动重分配概率。此门槛针对完整类别，不能误称每个 dataset 子组也满足门槛。

## 保存与恢复

训练 opt-in 状态保存在 `runner_state['expert_sampling']`，与原 `corpus_rng` 分开。保存：完整规范化配置及独立 seed、实际 index/manifest/config 指纹、float64 每中心概率指纹、私有 CPU RNG、窗口/batch 计数。恢复重建实际索引并逐项比较；从 GPU map-location 载入的 RNG 先转 CPU，再交给严格状态校验。缺失启用状态、修改 profile/seed/索引/概率/源或破坏计数均报错，不重新播种继续训练。

新内层状态版本为 `navisaaclab.expert_window_sampler_state.v2`；原 isolated v1 报告/源码快照保持历史身份，不能把其“构建/独立抽样不消耗 corpus RNG”当成新桥接器行为。既有 disabled 训练 checkpoint 的 runner-state 合同保持兼容。

## 门禁证据

- [104 项 focused CPU 测试日志](expert_sampling_integration_tests.log)、[JUnit](expert_sampling_integration_tests.xml)：真实小 AMASSCorpus、启用/关闭的实际 World 专家方法、一次真实 CPU PPO/D 更新与 optimizer checkpoint、配对 reset/expert、完整 seed/save-resume、错误配置与损坏状态。
- [Ruff](expert_sampling_integration_ruff.log)：四个业务文件和新增测试 PASS。
- [真实 v7 验证](expert_sampling_integration_v7_verification.json)、[日志](expert_sampling_integration_v7_verification.log)、[可复跑程序](verify_expert_sampling_integration_v7.py)：交错原调用后的 reset 全 payload/shared RNG 精确相同；恢复后的专家 history/next、后续 reset 和两个私有 RNG 精确相同；没有改动 global Torch CPU RNG。
- [实际 pure preflight](expert_sampling_integration_preflight.json)：当前本地资源及新配置 PASS，无 Torch/SDK 导入。全库 SHA 使用既有冻结声明；此探针没有重复 hash 1.1 GB bank，实际分类所用 root 特征另作 SHA256。
- [独立复核](expert_sampling_independent_review.md)：另 66 项 pytest、139 项实际 corpus/World/trainer-checkpoint 检查 PASS，未改源码。
- [当前变更与源 SHA](expert_sampling_integration_review.json)、[before 快照](expert_sampling_integration_before/source_manifest.json)。

概率 SHA256：`427ea2d078181fdf811fb2c3701fec48955dba2f2c10a297964ce888e7917172`；registry SHA256：`a42c2455c17507ba164aae63ae0cc9d68be83c79073cf46f5b4a39e685de7ae4`。v7 保持 `68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`。

## 后续训练方式与判定

应开启 **fresh 公共 PACER warm start** 的新实验目录。不能把 negative-backward 已训练 checkpoint 直接续训并声称仅改变专家样本分布；该 checkpoint 没有相应 sampler 状态，严格恢复会拒绝。

以下四次更新是建议 GPU smoke 命令，本次未执行。参数取实际 negative-backward 基线；正式对照将 `--updates` 改为 1000 并用新输出目录，再做同预算、相同固定 reference 的完整 22 条跟踪、inclusive/upright 风格、速度桶、跌倒及真实足部接触评估。

```bash
python scripts/train_paper_amp.py \
  --config configs/diffusion_amp/amp_style_amass_stratified_experts.yaml \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --output output/diffusion_amp/amp_style_amass2848_stratified_experts_pilot_v1 \
  --num-envs 256 --updates 4 --rollout-steps 32 --episode-steps 300 \
  --terrain-extent-m 192 --spawn-margin-m 48 --physics-only --seed 7 \
  --trajectory-source pacer_curriculum --curriculum-mode smooth \
  --curriculum-speed-max-mps 2 --curriculum-accel-max-mps2 1.5 \
  --checkpoint-every 4
```

CPU sampling 门禁通过不等于 GPU 训练或物理效果通过。当前 negative-backward 的 11/22 成功、0 跌倒仍是既有基线，不属于新采样配置的成绩。新先验是公开的部署/训练实践选择，不能称作者 2,848 原始 clip 或论文采样概率相同。
