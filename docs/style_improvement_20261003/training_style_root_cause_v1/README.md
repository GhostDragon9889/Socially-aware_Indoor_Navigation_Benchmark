# AMP 风格与导航退化：独立 CPU 根因审查

## 范围与结论

只读审查当前 state-prior `step_2000.pt`（SHA `19f3341b…`）、已保存训练日志、1,024 条保存的训练回放及已存在的完整 22 条 validation 结果。没有启动 GPU、更新模型、读取 sealed 34 test 输出、改动历史 934 源码或历史 183 manifest。

当前 trial 的状态动作时序、镜像变换和 history RMS 未发现确定的实现错误。root 另外找到一个实际恢复风险：将新配置的 AMP 权重改为 `.5` 后，旧 runner 的 `--resume` 仍加载 checkpoint 中的 `2.0`，没有比较这两个值；这样新配方可以被静默忽略。该问题已作最小修复，并通过独立审查，见下节。

## 当前训练的实际证据

[CPU 量化报告](training_root_cause_CPU.json) 绑定实际权重、日志和源码 SHA；[执行脚本](audit_training_CPU.py) 只读保存的 train replay，不创建新的物理轨迹。

| 最后 100 个训练 update 的均值 | 当前 state-prior |
|---|---:|
| AMP 原始奖励（已经含内部 scale 2） | 0.36882016 |
| 当前 outer weight 2 的 AMP 贡献 | 0.73764032 |
| outer task weight .5 的任务贡献 | 0.42636432 |
| AMP / task 平均贡献比 | 1.73007 |
| 同一批原始奖励乘 outer .5 的数值 | 0.18441008 |
| 同一批数值的 AMP / task 比 | 0.432518 |
| `50 × symmetry_loss` | 0.01600110 |

最后两行奖励权重比较是固定旧数据的代数计算，不能预测新训练后的质量或替代梯度归因。AMP 贡献大于 task，说明下一个单变量权重实验有明确检查对象；它没有证明权重是所有退化的唯一原因。

### 状态与动作

`state_action` 在物理步前取 `s_t..s_t-9` 历史与实际执行的 `a_t`。专家在同一帧 `t` 取历史，动作使用 `q_(t+1)/PD_scale` 并做相同 clamp / 冻结坐标处理。没有发现一帧错位。`state` 分支明确使用物理步后的 history，保持原 PACER 的 state-only 奖励时序。

AMASS 不提供实际控制命令。上述专家动作是运动学 PD target proxy：不会补偿惯量、重力、阻尼、当前跟踪误差和接触力，因此不能称为 inverse dynamics。这个近似需要保留在结果解释中。

实际新判别器的 12 条冻结动作列仍全部为零；57 条有效动作列已经学习。1,024 条固定 train replay 上，将动作置零的平均 logit 差为 **0.001104**，将动作循环错配的平均差为 **0.000350**。在这一批保存状态上，动作条件贡献很小，证据不足以把腕部退化直接归因于动作列。该固定状态干预不是物理重运行。

### RMS 与镜像

1950 维 history RMS 按完整 `10×195` history 更新；新增 69 维动作没有被错误拼进 history RMS。保存回放归一化后的 clip 坐标占比为 **0.1292%**，不支持将此批样本的退化解释为广泛 RMS 饱和。

固定 PACER `_sym_loss` 的公开实现确实使用 `50`。当前 policy loss 平均有效 57 维，公开 loss 平均 69 维：对于相同有效坐标残差，当前正则强度是公开式的 `69/57 ≈ 1.210526`。这是已有明确工程适配，不是已证实的因果错误。此次单变量实验保持它不变。

### 腕部误差与动作风格

已有完整 22 条结果显示，新模型的腕部误差比上一 protected 模型大。当前策略的任务输入只包含 XY 未来与地形，没有 AMASS 目标上身姿态或风格 latent。同一 XY 路径可以对应多个合理上身动作，因此 unconditional AMP 没有信息去追踪唯一的原动作腕部轨迹。动作分布质量和逐帧 mocap 腕部误差需要分别检查。

添加目标姿态、条件判别器或 ASE latent 将成为额外方法变更，不能将它悄悄写成对 NavIsaacLab 原方法的严格复现。当前不添加这些改动。

## 恢复奖励身份修复

[修改前后身份及精确增量](resume_reward_guard/staged_change_manifest.json) / [必要回归](resume_reward_guard/test_resume_reward_guard.py)。

新增 `validate_resume_reward_config(requested_training, saved_config)`，用 canonical `TrainingConfig` 补全请求中的省略字段，仅比较实际奖励的五个字段：`amp_reward_scale`、`amp_reward_weight`、`task_reward_weight`、`direction_mode`、`amp_probability_epsilon`。请求和 checkpoint 不一致时明确拒绝 training resume，并要求 fresh experiment；不覆盖 checkpoint 配置。

runner 只在已有 `args.resume` 内的 `not args.evaluate_steps` 分支调用。fresh training、纯 evaluation、显式 power stage 均保留原路径。移除唯一新增 helper 与两条 guard 语句后，生产代码 AST 与修改前完全相同；这是结构检查，没有宣称新的 GPU 数值回放。

实际必要回归 **10 passed**，Ruff 通过。将比较条件 `!=` 在内存中反转为 `==` 后，实际 **8 failed / 2 passed**，证明回归可捕获错误；生产文件未被此语义 mutation 修改。cold import configuration 未导入 Torch / Isaac Sim / Isaac Lab。

[独立代码审查](../resume_reward_guard_peer_review_v1/review_v2.json) 实际通过 12 项来源/语义检查，无发现。独立审查也实际调用既有 `power_stage`，确认显式 power 阶段迁移仍可接受、隐式 power 变更仍拒绝；只保持 `scale×weight` 乘积相同而分别改变两个字段，仍会拒绝不一致的恢复配置。

生产 runner 和旧 isolated 934 runner 原本不完全相同。因此另提供 [旧934适配后的 runner](resume_reward_guard/isolated934_after/scripts/train_paper_amp.py)，只添加同样的两条校验语句，不将生产 183 的其他 runtime 变化带入新训练。旧 934 文件保持不变。

## 最小下一轮

仅将 `amp_reward_weight: 2.0 → 0.5`，同步 `reward.amp_weight` 声明。内部 `amp_reward_scale: 2.0` 保持不变，使 effective softplus 系数由 4 变为 1，与固定公开 PACER 的 `scale2 × disc_reward_w.5` 相同。NavIsaacLab 作者的私有配置仍未知，不能据此声称论文超参数完全一致。

采用 fresh 训练，保留 2,848 条 AMASS、固定一半真实 XY / 一半 procedural 的命令、PD、mirror50、replay、history RMS、terrain statistics、PPO 和训练预算。需要完成同一完整 22 validation、四人 / 15 人导航和固定参考风格评估之后再决定是否改默认模型。训练 reward 变化、低 MMD 或某一场景的零跌倒，都不能单独证明达到论文的八场景效果。

最终 v2 的 [独立源码、配置与实际冷 loader 审查](pacer_reward_candidate_peer/final_v2_source_profile_actual_loader_peer.json) 已通过：935 个实际文件逐一匹配声明，932 个原文件未改、两个已批准 guard、新增唯一绝对配置；实际加载 coefficient 为 1，旧 weight2 resume 被拒绝。继承的两条旧实验描述已在独立 v2 中更正，初始准备版保留。

[真实 cold 模型/RNG 证据的独立审查](pacer_reward_candidate_peer/final_v2_actual_cold_owner_peer.json) 通过 39 项来源/语义检查，20 个实际 owned imports 来自最终 935 源码。实际 owner CPU 断言检查了两个同初始化模型的全部参数、normalizers、Torch RNG、synthetic actions、raw AMP reward 完全相同，以及新 weighted AMP reward 为原值的四分之一；没有复载大模型或重复大型测试集。Synthetic 输入和相同 synthetic joint limits 的检查界限保留，不能代替物理训练或导航效果。

另外已完成 [production V12 的 185 文件源码冻结](../native_runtime_epoch_v12/README.md)，绑定 public 配方、独立 guard 审查和 canonical 测试，保留历史来源与默认模型。

## 实际训练与评估证据接口

[完整训练 payload reader 的独立审查](pacer_reward_candidate_peer/complete_payload_reader_source_peer.json) 通过 967 项来源及结构检查。将路径、schema、935 来源和奖励身份的适配逆转后，主 reader 全文字节及 AST 与原版相同；六个训练、优化器、Replay、RMS、RNG 和实际 result 检查函数的 AST 保留。新 archive 实际包含 935 个训练源码及四份启动来源文件，没有将 production 185 混作训练来源。

该 reader 随后实际审查了封存的 pilot4：[完整 pilot 审计](../state_prior_pacer_reward_training_audit_v1/actual_pilot4_v1/audit.json) 通过 2,081 项检查，确认 4 个 update / 32,768 transitions、五个奖励字段和完整 payload。它明确记录 `terminal=false`、`runtime_finalization=false`，仅证明短训练链路已经运行。

[最终 evaluator 独立审查](pacer_reward_candidate_peer/final_evaluator_source_peer.json) 通过 40 项检查，实际 20 个必要测试全部通过。实际 dependency validator 与 frozen reader 的 `files` 字段、935 manifest 和固定旧评估协议匹配；首轮字段名接线失败已保留并修正。在内存中反转奖励比较条件后，测试出现 1 failed / 19 passed；未修改 evaluator 源码。完整 22 条 tracking、共享导航、45 维固定参考风格和 acceptance 数值均保持原协议。

此处 evaluator 仍为准备状态，没有接收未来 terminal 权重、运行 GPU evaluation 或证明动作质量提升。训练完成后的权重身份、完整 payload 和固定评估结果仍必须分别通过实际审查。
