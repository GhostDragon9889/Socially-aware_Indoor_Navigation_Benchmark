# 完成导航中的原始预测与执行参考速度

[`reader.py`](reader.py)仅读取已封存的S0/S1四人200秒记录：复用已审查的locomotion准入及全体control/substep/force/reset核对，检查实际nav与CPU审计退出、child与owner绑定、所有原始数组SHA。没有载入Diffusion/AMP模型、重采样路径、改变参考或运行SDK。结果是已发生闭环的关联，不是因果消融。

## 计算合同

每个已执行control选择最新提交的plan，重建全部目标与phase。raw51帧是10Hz线性预测；control区间结束age1..15的raw速度为 `(raw[lower+1]−raw[lower])*10`，其中`lower=(age−1)//3`。在age3/6/...恰好到达knot时取已执行区间左侧，不取未来下一段。参考速度取同control已保存的因果reference_velocity。全体失败、接触及reset均保留。

plan数组没有保存episode ID；从真实physical在plan提交control对应actor的episode关联，并核对每个reset都有新plan及全部已执行plan属于当前episode。没有声称恢复未保存的reset前plan状态。raw target最大重建差约6.36e−7m，reference target/refV重建差0。

独立报告每条注册预测前0.5秒和全5秒的两种速度：净位移范数／时长；各XY段弧长之和／时长。往返预测的净位移小，不代表没有预测运动。末100秒的控制定义为control≥3000；末100秒plan行只取提交control≥3000，两类分母分开。

## 实际结果

| 全部24,000 actor-controls | S0 | S1 |
| --- | ---: | ---: |
| raw已执行区间速度均值，m/s | 1.311654 | .328018 |
| 因果执行参考速度均值，m/s | .972276 | .253320 |
| raw速度<.1m/s的controls | 151 | 6,265 |
| reference速度<.1m/s的controls | 186 | 11,648 |

S1末100秒每人注册200条预测，并各有3000个执行control：

| 行人 | raw前0.5s弧长速度均值，m/s | raw前0.5s净位移速度均值，m/s | reference速度均值，m/s | raw弧长速度<.5的plan行 |
| --- | ---: | ---: | ---: | ---: |
| 0 | .186987 | .122953 | .094107 | 200/200 |
| 1 | .169528 | .135915 | .067004 | 200/200 |
| 2 | .242124 | .033200 | .115178 | 191/200 |
| 3 | .182856 | .129176 | .101658 | 200/200 |

S1末100秒全部12,000 actor-controls的执行参考速度都<.5m/s。原始DP预测已经以低速为主，因果适配后的执行速度进一步降低。行人2的净位移速度.0332远低于弧长速度.2421，存在回摆；不能将净位移小直接称为静止。结合[原始支撑诊断](../shared_social_locomotion_CPU_v1/README.md)的长双支撑，只能提出反馈预测与参考响应待进一步检查，不能归因于某一模型、适配参数或奖励。

S0末100秒每人注册629条plan，含多次跌倒后的重规划；S1只有200条。所有行保留，plan行统计不当作两侧等样本独立试验，不与论文数值直接排名。

## 实际执行

真实CPU工具chunk `25b586`，exit0；三项边界／往返／非有限测试 `2f41f1`，exit0；Ruff `1dfcdc`，exit0。`actual_diagnostic.json`保存全部分组、输入和source身份，`execution.json`保存真实退出和日志SHA。S2的22项guarded来源没有改变。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261005/shared_social_planner_response_CPU_v1/reader.py
```

reader拒绝覆盖既有`actual_diagnostic.json`；新执行需在新证据目录保留对应reader／来源和输出。当前目录是已经完成并封存的证据，不应删除结果后重跑。
