# S1 终态 D 的 CPU 离线输入诊断

本目录仅诊断已封存、实际退出 0 的 S1 终态判别器。结果是终态 D 的离线重新打分和输入敏感性；不是采集时奖励复现，不是训练消融、留出集准确率或动作质量的因果证据。

入口为 `scripts/diagnose_shared_social_discriminator.py`。它复用现有 Pop-Art metadata、来源、inode/SHA、CPU tensor 准入和原始 D/RMS builder，只构造 D2019 与 AMP RMS；不构造 actor/critic/CNN，不恢复优化器、replay 或保存的 RNG。完整其它模型数据与原 actual0 CPU audit 的同一个 checkpoint SHA 绑定。

固定输入为 S1 `step_002000.pt`、`rollout_update_002000.pt` 与完整训练／CPU 审计退出 receipt。读入使用 `weights_only=True`、`map_location=cpu`、`mmap=True`，在保留描述符上校验前后文件和来源一致。使用终态 AMP RMS，而非采集时 frozen RMS；真实专家 4096 行与 policy 8192 行全部保留，包括失败、截断及各自归属。

按相同行的 history 分别计算原动作、动作归零、同种群动作置乱及跨 expert/policy 动作交换；policy 另做同一控制时刻、同一物理四人组动作轮换。输入梯度针对单行 logit 对标准化／clip 后 D 输入，分别报告 history1950 与 action69 的 L2 和逐坐标 RMS，不能把维数不同的 L2 直接解释成因果重要性。

policy 原始 Gaussian 动作与实际 clamp/mask 动作分别核验；专家实际边界及 57 维 mask 核验。terminal expert 包没有 next-frame mocap position，因此仅核验 inverse-PD/clip 的原源码来源，不伪称从动作库重新推导了专家动作，也不证明代理动作实际驱动出参考运动。

`per_sample.pt` 保留每行 logit、梯度幅度、确定性 donor 索引、policy 控制／actor／episode／失败归属及 expert motion/frame ID。`diagnostic.json` 绑定原输入、代码和逐样本输出 SHA。fixture 测试不涉及 SDK、数据采样或训练。

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/diagnose_shared_social_discriminator.py \
  --execution-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_execution.json \
  --output-directory docs/style_improvement_20261005/shared_social_D_input_diagnostic_v1/actual_S1_v1 \
  --seed 7 --batch-size 256
```

输出目录必须是新路径；既有证据不能覆盖。实际执行已完成，CPU 工具 chunk `20f1f6` 退出 0；六项 fixture chunk `4fdb51` 退出 0；Ruff chunk `e25539` 退出 0。完整退出记录见 `execution_receipt.json`，逐样本输出及 SHA 见 `actual_S1_v1/`。

## 真实 S1 结果：负向诊断

| 动作输入条件 | 排序 AUC | expert − policy 平均 logit | policy 平均绝对 Δlogit | expert 平均绝对 Δlogit |
|---|---:|---:|---:|---:|
| 原输入 | 0.99189869 | 3.744614 | — | — |
| 同行动作归零 | 0.99188221 | 3.740479 | 0.002118 | 0.002369 |
| 同种群动作置乱 | 0.99189311 | 3.743791 | 0.001102 | 0.000717 |
| 跨种群动作交换 | 0.99187890 | 3.743509 | 0.000993 | 0.001171 |
| policy 同时刻四人组动作轮换 | 0.99189822 | 3.744397 | 0.001016 | — |

原输入的 balanced accuracy 为 0.978149；上述动作干预后该数值保持一致。逐坐标梯度 RMS（history / action）为 policy 0.004051 / 0.000736，expert 0.003182 / 0.000579。维数分别为 1950 / 69，梯度是标准化／clip 后坐标的局部敏感性。

这批 S1 终态 D 的动作通道影响很小，不支持“动作代理主导终态分类分离”的假设，应降低该假设的下一轮训练优先级。此负向结果不证明训练早期、其它状态或重新训练后动作通道无影响，也不能解释 S1 的动作质量失败；未进行训练消融或因果质量评价。完整 8192 policy 与 4096 expert 行及失败归属均保留，未缩小评价协议。

policy 原动作在本包中无越界坐标，但全部 12 个冻结维度仍有非零 raw Gaussian 值，实际输入按 57 维 mask 归零；专家冻结列为零、范围符合 clip，边界坐标为 0。专家动作是 next-frame mocap position / PD scale 的代理；保存包缺失 next-position，未重采样动作库或伪称验证真实专家控制。终态 AMP RMS count 为 245760000，采集时为 245637120，故所有分数仅为终态离线重新打分。

初次 pytest 在收集前被环境中 ROS 插件缺少 `lark` 阻断（chunk `9cd780`、退出 1）；禁用无关插件自动加载后六项测试实际执行并通过。
