# 固定真实 AMASS2848：原窗口先验与 S1 实际专家末批

新增 [`diagnose_shared_social_expert_coverage.py`](../../../scripts/diagnose_shared_social_expert_coverage.py)，只读固定 v7 **train** bank、原 BABEL train overlay、S1 已正常退出的训练／D 诊断和实际终态 packet。它没有采样新窗口、构造 policy／D、加载模型 checkpoint、启动 SDK／CUDA 或修改 S2。用于决定后续风格实验的支持范围，不构成训练效果或失败原因证明。

## 已实际执行

- 第一版 4 个合成 CPU 测试 `a669a7`／0、Ruff `e1b575`／0；实际分析 `24a38b`→会话38422→`cbf248`／0。
- 最终增加真实退出 int0／marker 拒绝测试及完整训练计数记录：5 tests `0f341c`／0、Ruff `457767`／0；实际分析 `aa1028`→会话60638→`0fc555`／0。
- 两版全量 `measurement` 完全相等（`c9848d`／0）。各精确源／artifact／exit 见 `execution.json`，第一版保留。

全部 **2848** 段、**494792** 个 history10／future1 合法中心重新通过既有未改动的 `classify_windows`。实际 root velocity／XYZW feature SHA `2b80a026…` 与原真实覆盖登记一致；manifest `68d41be2…`、1,155,817,682B train bank `46e02411…` 全 SHA 前后相同。未打开 validation22／Test34 bank 或 BABEL val／test 标注。

下面三个分母不同：均匀每个合法窗口；实际原 **uniform clip→该 clip 内 uniform legal center** 先验；S1 实际最后一个专家 batch。不是把全 bank 的窗口频率直接当作训练概率。

| 短窗运动 proxy | 所有中心数量 | window-uniform 比例 | 实际原采样先验 | S1 末批数量／4096 |
| --- | ---: | ---: | ---: | ---: |
| low_speed | 110378 | 22.3080% | 23.8952% | 980／23.9258% |
| fast | 6063 | 1.2254% | 1.5801% | 46／1.1230% |
| sideways | 62693 | 12.6706% | 11.2666% | 452／11.0352% |
| turning | 114472 | 23.1354% | 22.5018% | 972／23.7305% |
| moving | 201186 | 40.6607% | 40.7564% | 1646／40.1855% |

先验是按原采样规则 `p(center)=1/2848/(clip_frames-10)` 计算的名义概率；未模拟 `Torch.rand` 的有限 FP32 网格。它不是全程实测 exposure 直方图，也不代表确定 seed 的每次 draw 概率检验。

分类继续按原 10 帧中位 root 速度／侧向速度比例和 net yaw：low <.12m/s、fast >=1.8m/s、lateral >.55、net yaw >=.2rad，保留原覆盖优先级。它们不是评价 speed bins、BABEL 动作真值或作者类别；同一整段中可存在多个短窗类别。

原 fast 先验的整段 proxy 来源中，running_like 仅占 **27.59%**，walking_like **43.90%**、turning_like **22.10%**、其他约 **6.41%**。因此整段 running_like 的17段不能代表全部 fast 支持，也不能直接以五类各20%提高它们。原先验中 fast 总质量只有1.58%，是待验证的支持约束；末批期望64.72、实得46，**单批波动不能证明全程系统性欠采样，更不能证明低速失败的原因**。

## 实际 exposure 和语义边界

末批 `motion_ids/frame_indices` 与真正封存的 `rollout_update_002000.pt` 专家 packet **逐项相等**；所有历史和 next-frame 边界合法。4096行包含2149个clip、4061个不同中心、1523父动作、187受试者。相同父动作的多段仍有相关性，不能据此称均匀人物／父源。

完整实际训练计数为2000更新、16,384,000物理转换、8,192,000专家窗口；这里只保留的末批4096行不证明每个ID的完整训练曝光。当前结果缺少全程 per-ID ledger；后续新实验应记录它，不能把末批未出现的clip称为整个训练没使用。

| 原 BABEL train overlay 的状态 | 末批行数 |
| --- | ---: |
| 完整 clip 的显式 timed interval union 覆盖 | 1218 |
| matched，但没有完整 clip 的显式 timed union | 1165 |
| 没有 train annotation exact match | 1713 |

保留原 `train.json`＋`extra_train.json`、所有未知和多标注者 evidence；不投票、不替换 style5、不以数据集名称推断动作。Union 覆盖不是单一动作／标注者一致或原 pose clock 测量。使用 extra train 的事实单独声明，符合[BABEL 官方 data 文档](https://babel.is.tue.mpg.de/data.html)。

该官方文档对 dense `train.json` 仅有 sequence label 的序列采用整段单动作跨度约定；当前原 overlay **没有应用该约定**，所以“没有显式 timed union”不等于“没有语义标注”。后续可独立补充这一明确的 dataset assumption，核验 dense member／单动作条件和实际时长，保留其与显式 frame 时间的区别；不能无条件推广到 extra 多标注者或把约定当成新时间测量。

## 后续优化依据

先完成正在运行的 S2 与独立全动作／固定风格／200s评价，再决定下一变量。此次 evidence 支持使用**短窗实际 kinematics 和任务请求 context**描述训练支持，不支持只按 whole-clip 标签或已停住的身体速度重采样。固定2848池、未知标签、heldout边界和现有评估参考继续保持。

本轮重新查阅 [CMP 原论文 §3](https://arxiv.org/html/2608.03234v1#S3)：它学习任务／动作相关性，以在线高 advantage 和 demonstration 分支训练 context adapter。这是可研究的方向；当前 reader 没有 relevance network／adapter，也没有把简单 soft weighting 宣称为 CMP。后续相关性和训练效果须由实际独立消融验证。

S2 与其自动评价 owner 仍独占 GPU，本轮没有新 GPU work，默认权重保持。**尚无达到或超越 NavIsaacLab 论文完整效果的证据。**

复跑须使用新的输出目录：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/diagnose_shared_social_expert_coverage.py \
  --diagnostic-exit-receipt docs/style_improvement_20261005/shared_social_D_input_diagnostic_v1/execution_receipt.json \
  --output-directory /tmp/S1_expert_coverage_new
```
