# AMASS root 课程：完整22段验证误差分解

本报告读取既有真实物理捕获；CPU处理，没有启动SDK/GPU或重算官方style/MMD。
所有22段和跌倒帧均保留在主指标中，未来XY tracking不等于learned Diffusion导航。

## 主指标（完整片段、同一相位）

| 指标 | negative-only1000 | AMASS-root1000 |
| --- | --- | --- |
| 成功片段 /22 | 11 | 17 |
| 有跌倒片段 /22 | 0 | 0 |
| 真实poststep总帧 | 11982 | 11982 |
| 全24body位置误差 m | 0.300708 | 0.229298 |
| 全24body旋转误差 rad | 0.861454 | 0.831882 |

## 同帧诊断（不替换世界坐标主指标）

| 平均值 | negative-only1000 | AMASS-root1000 |
| --- | --- | --- |
| root XY m | 0.228346 | 0.142768 |
| root yaw rad | 0.600987 | 0.551314 |
| 各自root/heading局部body位置 m | 0.119189 | 0.125401 |
| heading局部body旋转 rad | 0.573434 | 0.561986 |
| parent局部关节旋转 rad | 0.301823 | 0.287741 |

局部诊断没有时间best-match，不能把heading移除后的下降当作独立因果比例。
速度区间保持reference_root_vel XY范数的原合同；各区间帧数和误差详见comparison.json。

## 描述性success子集（明确两种分母）

| 子集 / run | clip分母 | frame分母 | clip均值位置/旋转 | frame加权位置/旋转 |
| --- | --- | --- | --- | --- |
| each_run_success_only /baseline | 11 | 7191 | 0.238868m /0.603017rad | 0.243243m /0.597196rad |
| each_run_success_only /amass_root | 17 | 9902 | 0.199851m /0.704901rad | 0.194611m /0.646962rad |
| common_success_intersection /baseline | 11 | 7191 | 0.238868m /0.603017rad | 0.243243m /0.597196rad |
| common_success_intersection /amass_root | 11 | 7191 | 0.168965m /0.560491rad | 0.168929m /0.550762rad |

这些按结果选择的子集仅解释分母敏感性，不替代ALL22，不证明论文.218m/.343rad采用相同分母。

## 配对、跌倒和来源

全部22初始状态/AMP history与所有参考目标/时钟数组通过dtype及bitwise配对。
每段跌倒连续时段、真实source帧和相对reset时间及首fall前后1秒分解在comparison.json；不会重置或移除跌倒帧。
源码tar逐member验证，真实step1000检查点SHA及捕获来源绑定均记录。

- baseline controller SHA256: `62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b`
- amass_root controller SHA256: `dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e`

该工程课程同时改变固定source命令、合法reset/clip-end和明确clip外线性延拓；不是tail单变量试验。
单seed、反复使用的validation不能证明统计显著性、作者复现或未查看的34条test性能。
训练源码归档与验证捕获source分别保留，不假定今天整个工作树等于历史运行。
