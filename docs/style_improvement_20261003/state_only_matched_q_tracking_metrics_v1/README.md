# State-only matched q：实际full22风格与旋转诊断

原CPU指标单次执行已实际闭合：session91146，acf983→05cfb7，exit0。
使用已封存state1950终态97b2…582与生产185 tracking结果d23d5，原函数对象未改。
22条完整clip／11982个post-step metric帧／11960个同clip风格interval；所有失败与idle保留。
未重算旧q／19f／默认指标，未deserialize权重、启动SDK／GPU或读sealed34。

| 指标 | 实际值 |
| --- | ---: |
| 完整clip tracking成功 | 18 / 22（81.82%） |
| 出现跌倒的clip | 0 / 22 |
| world body位置误差 | 0.218970 m |
| world body旋转误差 | 0.751018 rad |
| inclusive 90D MMD² | 0.05209593 |
| 速度Wasserstein | 0.079735 m/s |
| root yaw误差 | 0.476513 rad |
| heading-relative body旋转误差 | 0.521384 rad |
| parent-relative joint旋转误差 | 0.274913 rad |

后三项是另外定义的同帧诊断，不能替代world旋转误差。0跌倒不等于全动作跟踪通过；4条tracking失败全部保留。
这不是闭环多人导航结果，也不证明达到作者未公开的准确评估协议或论文完整效果。

[原90D风格结果](tracking90/evaluation.json) · [原旋转分解](decomposition.json) · [汇总](comparison.json)
· [实际来源](actual_source_admission.json) · [真实CPU闭合](actual_CPU_closedreceipt.json)。
