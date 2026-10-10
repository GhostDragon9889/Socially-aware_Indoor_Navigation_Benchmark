# 下一轮 AMP 训练的效果验收与奖励诊断

## 已测 60 秒记录

输入为 `shared_obstacle_clean_sdf_probe_v1`：旧 corrected1000 actor、旧 Diff1000、显式 clean guidance20／peer5／geometry1、endpoint 模式。CPU [评估报告](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_obstacle_clean_sdf_probe_v1/pedestrian_metrics/report.md) 与 [逐时段图](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_obstacle_clean_sdf_probe_v1/pedestrian_metrics/phase_quality.png) 已生成；输入 SHA、逐接触帧与 terminal 前窗见同目录 JSON。

- static normal contact：4/7200 actor-frames，0.05556%，2 个连续事件，均为 actor0；最大单 body-pair 法向力3570.22N。peer：0/10800 无序 pair-frames。
- terminal：actor0 6.5s、actor3 21.6s、actor2 22.0667s；当帧均无 static/peer 接触。actor0 最近 static 接触早1.8s，2s 窗含4帧；另外两人同 episode 无已记录 static/peer 接触。时间先后不能证明原因，地面支撑和自身接触不在本合同内。
- 三个20s段：terminal1/2/0，tracking均值.1713/.1682/.1862m；有效相邻帧速度1.0385/.5294/.4966m/s。末段不跌倒同时速度下降，不能仅以跌倒数判断行走效果。
- 每人累计有效路程33.30/33.76/56.63/41.33m，但首完整 route vertex 到达计数全0。完整顶点最小距离4.352/3.802/6.454/2.142m；动态5m lookahead 目标与路线顶点是两个合同。
- 2048姿态、固定train cluster：K50/K100 coverage=.84/.67；Gaussian MMD²=.113998，train/test参考=.035754。节点、尺度和核是本地显式选择，不能与作者数字直接比较。

## 500／1000 checkpoint 比较门禁

冻结新 Diff5000、clean guidance、训练源、physics、初始状态、场景和 seed；在同一组独立验证命令上比较 transfer 起点、update500、update1000。验证不能只复用训练生成轨迹；每次恢复从相同已保存起点运行，记录至少200s共享弯曲路线与足够大地面的解析直线／转弯控制。不同训练 checkpoint 之间不改 reward、RMS、接触阈值或 planner。

| 门禁 | 需要记录／比较 | 判读 |
| --- | --- | --- |
| 数据与物理有效性 | 非有限值、reset/episode disagreement、越界、SDK测量fallback、来源／合同差异 | 这些错误必须为0；有错误先修数据或采样，不能用训练补偿 |
| 跌倒暴露 | fall终止/actor-seconds、首次跌倒时间、生存曲线、各actor/seed结果；同时报告censored episodes与timeout | 以匹配暴露和独立seed比较，给episode/actor分块区间；短记录0次不等于长期稳定 |
| 控制质量 | 同刻tracking mean/RMS/p95、速度与目标速度误差、停滞暴露、完整顶点到达／路线进度、signed power分布 | 排除reset跃迁；低tracking误差必须伴随运动和路线进度，不能奖励原地跟踪弱计划 |
| 障碍与社会接触 | static actor-frame、peer unordered pair-frame、连续事件、法向力；几何代理另外报告 | 按同阈值／同子步合同比较，不将根位置未入障碍当无全身接触 |
| 姿态质量 | 同抽样量、同train-only K50/100、同held-out MMD核与采样seed；再检查视频／姿态 | coverage、entropy不能单独决定自然度；MMD下降不能以停滞或跌倒换取 |
| 学习过程 | 连续验证窗口的上述质量；训练 reward分量、D两类概率、gradient penalty/norm、PPO KL/clip、value残差 | update数只是预算；训练均值平台不是任务收敛证明 |

如果500→1000在多seed下仍改善完整路线进度，且跌倒／接触、tracking与姿态指标没有出现明确退化，可继续同一stage训练。若训练loss降低但验证路线不进展、速度下降或跌倒上升，先定位 planner命令、动作代理与控制／reward 的冲突；不要只延长更新。这里没有把任意人工阈值伪装成论文验收线。

## Style 压力与 critic：当前证据

旧 corrected1000 的前／末100更新均值：

| 指标 | 前100 | 末100 |
| --- | --- | --- |
| style reward | .534400 | .292841 |
| 已乘.5的 task reward | .392764 | .435126 |
| style/task均值比 | 1.36061 | .67300 |
| D expert / policy | .74797 / .20987 | .85772 / .12357 |
| value loss（normalized） | .096802 | .013248 |
| PPO KL / clip fraction | .017802 / .22927 | .015487 / .19629 |
| policy / D gradient norm | 56.6521 / 1.7109 | 19.0904 / .9495 |

末段 style 并没有在均值上压过 task。现有 `2*softplus(logit)`、epsilon1e-4 的数学上限是18.42068，远高于单帧position贡献上限.5；最后一更新 style max5.2665，但没有保存per-sample高分位、advantage相关或梯度分解，不能据此断言稀有style样本已经主导策略。D expert/policy 分离扩大值得做状态输入消融，也不能直接证明动作shortcut。

建议先记录style/task分量的p50/p95/p99、触cap比例、分量与advantage相关，以及held-out专家／策略 D 分布。只有当验证控制质量退化、style尾部／优势集中同时出现时，才另建明确对照stage降低style权重或采用有界style reward；不在本轮 NavEq2 profile 中偷偷改公式。原始AMP的有界二次奖励与当前PACER BCE／softplus并非同一映射，见 [AMP §5](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)；PACER原配置也使用不同task/style组合，见 [官方源码](https://github.com/nv-tlabs/pacer)。

公开 value RMS count=6,881,280,001、mean50.0823、variance153.0001，新2,048,000样本只给旧统计约0.0298%的附加权重。这是迁移先验很强的证据；现有normalized value loss下降且finite，尚无critic数值发散证据。下一轮先记录denormalized value/return残差与explained variance，分清网络拟合与统计适应。若固定任务改变后critic持续有偏且PPO行为退化，可显式actor-only／新critic或重新校准value统计的独立stage；不能静默清零旧RMS、优化器或用新增metadata改写旧checkpoint含义。

新状态D＋完整1950RMS对照已实现并通过CPU数值门禁，root另有真实2更新pilot；它仍是公开PACER实现对照，不是作者NavEq2状态加动作的实验。大地面50s无跌倒的低层诊断也不能替代200s共享障碍、全局路线与姿态验收。
