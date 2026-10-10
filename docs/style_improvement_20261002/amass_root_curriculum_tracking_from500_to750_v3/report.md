# AMASS root 课程：完整22段验证误差分解

本报告读取既有真实物理捕获；CPU处理，没有启动SDK/GPU或重算官方style/MMD。
所有22段和跌倒帧均保留在主指标中，未来XY tracking不等于learned Diffusion导航。

## 主指标（完整片段、同一相位）

| 指标 | negative-only1000 | AMASS-root750 |
| --- | --- | --- |
| 成功片段 /22 | 11 | 16 |
| 有跌倒片段 /22 | 0 | 2 |
| 真实poststep总帧 | 11982 | 11982 |
| 全24body位置误差 m | 0.300708 | 0.401208 |
| 全24body旋转误差 rad | 0.861454 | 0.805335 |

## 同帧诊断（不替换世界坐标主指标）

| 平均值 | negative-only1000 | AMASS-root750 |
| --- | --- | --- |
| root XY m | 0.228346 | 0.325017 |
| root yaw rad | 0.600987 | 0.501213 |
| 各自root/heading局部body位置 m | 0.119189 | 0.135398 |
| heading局部body旋转 rad | 0.573434 | 0.590142 |
| parent局部关节旋转 rad | 0.301823 | 0.288845 |

局部诊断没有时间best-match，不能把heading移除后的下降当作独立因果比例。
速度区间保持reference_root_vel XY范数的原合同；各区间帧数和误差详见comparison.json。

## 描述性success子集（明确两种分母）

| 子集 / run | clip分母 | frame分母 | clip均值位置/旋转 | frame加权位置/旋转 |
| --- | --- | --- | --- | --- |
| each_run_success_only /baseline | 11 | 7191 | 0.238868m /0.603017rad | 0.243243m /0.597196rad |
| each_run_success_only /amass_root | 16 | 9228 | 0.22842m /0.694024rad | 0.224361m /0.639865rad |
| common_success_intersection /baseline | 11 | 7191 | 0.238868m /0.603017rad | 0.243243m /0.597196rad |
| common_success_intersection /amass_root | 11 | 7191 | 0.211295m /0.580238rad | 0.209807m /0.566239rad |

这些按结果选择的子集仅解释分母敏感性，不替代ALL22，不证明论文.218m/.343rad采用相同分母。

## 配对、跌倒和来源

全部22初始状态/AMP history与所有参考目标/时钟数组通过dtype及bitwise配对。
每段跌倒连续时段、真实source帧和相对reset时间及首fall前后1秒分解在comparison.json；不会重置或移除跌倒帧。
源码tar逐member验证，真实step1000检查点SHA及捕获来源绑定均记录。

- baseline controller SHA256: `62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b`
- amass_root controller SHA256: `1e87b82b643a1d94bf2117de5a57a27ebb1e0e9811a4cabf66ab1588fcaf9592`

候选预算750updates，baseline1000；半/较少预算的探索诊断不是同预算对照。
Lineage模式: continued_from500；continued_from500是新独立续训，不是原轨迹750，不能承诺PhysX cache bitwise恢复。
该工程课程同时改变固定source命令、合法reset/clip-end和明确clip外线性延拓；不是tail单变量试验。
单seed、反复使用的validation不能证明统计显著性、作者复现或未查看的34条test性能。
训练源码归档与验证捕获source分别保留，不假定今天整个工作树等于历史运行。
