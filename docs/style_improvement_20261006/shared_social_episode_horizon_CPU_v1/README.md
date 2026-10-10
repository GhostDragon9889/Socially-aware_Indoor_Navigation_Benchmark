# S1/S2 实际训练 episode 时长与 AMASS 重置先验

本批只读取已完成 S1/S2 的真实记录，实际工具会话14028退出0（`cbbb57`）。
没有启动 NN、CUDA 或 SDK；固定2848数据、现有指标和正在运行的S3配方保持。
结果见 [actual_result.json](actual_result.json)，退出依据见 [tool_execution.json](tool_execution.json)。

## 实际测量

| 项目 | S1 | S2 |
| --- | ---: | ---: |
| 已完成整组 episode 数 | 80,304 | 70,574 |
| 末尾未完成的组控制步，按组求和 | 2,559 | 2,277 |
| 已完成整组 episode 平均时长，秒 | 1.699144 | 1.933537 |
| AMASS 生成计划数 | 696,322 | 677,932 |
| 真实来源之外的未来规划节点比例 | 48.6179% | 49.6418% |
| 独立完整 validation22 平均控制时长，秒 | 18.154545 | 18.154545 |

每组固定两名 AMASS、两名 procedural 行人；任一成员终止或来源用尽均整组重置。
累计真实16,384,000 actor-controls、实际终态32×256采集和64组POST年龄逐项验证。
平均时长公式为：

`(累计 actor-controls /4 − 末尾每组未完成年龄之和) /已完成组重置数 /30`。

最终控制步已重置的组未完成年龄为0，不能使用其重置前POST年龄。
统计是跨64组汇总的完整 episode 平均时长，不是整个World每隔1.93秒重置一次。
来源结束标志可能与跌倒/接触终止重叠；目前没有全程逐次重置的动作ID、帧和持续时间档案，
不能精确给出仅由来源结束造成的组重置数或完整 episode 时长直方图。

外推分母是每份生成计划的50个canonical未来节点（0.1–5秒），
不是actor观测的10个未来目标点，也不是独立跟踪评价的末尾保持比例。
两种时长统计属于不同任务分布，不能当作严格相同条件下的性能对照或失败因果证据。

## 固定2848真实bank验证与单变量候选

实际mmap读取原始bank并验证全部motion ID、来源、帧数与30Hz时间间隔对应清单。
共523,272帧，最短91、最长338、均值183.733146帧；没有过滤短片段。
原始采样实现先等概率选动作，再对合法帧区间使用`floor(torch.rand * room)`。
past9意味着必须保留10帧历史；当前要求31个真实未来控制步。

| 理想离散均匀采样分析 | future31（当前） | future81（候选） |
| --- | ---: | ---: |
| 保留训练动作数 | 2,848 | 2,848 |
| 最少真实未来时间，秒 | 1.033333 | 2.700000 |
| 合法重置中心数 | 409,352 | 266,952 |
| 单动作剩余时间均值，秒 | 3.412219 | 4.245552 |
| 两个独立AMASS剩余时间最小值均值，秒 | 2.395560 | 3.429636 |
| 重置时10目标点在来源之外的理想比例 | 41.3422% | 26.4056% |
| 重置时完整5秒目标均为真实来源的概率 | 19.8665% | 26.3131% |

上述是clip-uniform/frame-uniform的数学先验，抽象了实际torch.rand的有限精度。
没有实测future81训练曝光；两个AMASS最小值未包括接触、跌倒或任务上限。
81是让全部2848仍可采样的最大固定future值：最短91帧减去10帧历史。
改为82会排除部分真实动作，应明确拒绝而不能静默筛选。
future81只改变AMASS参考重置帧的最低剩余时长；是否改善长时路线和动作风格需实际训练后独立验证。

## 来源及论文协议

[NavIsaacLab原论文IV-B/IV-C](https://arxiv.org/html/2606.26265v1#S4)说明目标条件AMP、
真实AMASS风格训练和完整闭环动作评价，并报告8场景、3–15人、200秒的同AMP高层对照。
公开论文没有给出与本地相同的逐次参考重置帧规则；future81是本地实验参数，不能声称作者原配方。
[PyTorch2.11的torch.load文档](https://docs.pytorch.org/docs/2.11/generated/torch.load.html)
支持显式CPU映射、`weights_only=True`与`mmap=True`，本次按该API读取真实记录。
16项实现来源与83项实际输入均在报告中绑定。

## 验证与复现

新时长内核24项和原相位检查8项，共32项通过，Ruff通过。
将“每条片段支持足够未来”条件反转后实际3项失败、29项通过；随后源码逐字恢复，32项再次通过。
审阅覆盖帧端点、clip而非window均匀采样、右删失年龄、最终重置、整组成员与计数约束。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/diagnose_shared_social_episode_horizons.py \
  --baseline-post-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_execution.json \
  --candidate-post-receipt docs/style_improvement_20261005/shared_social_full_epochs_v1/post_training_execution.json \
  --output /tmp/shared_social_episode_horizon_fresh.json
```

必须选择尚不存在的输出文件。当前诊断没有证明论文效果、失败因果或future81性能提升。
