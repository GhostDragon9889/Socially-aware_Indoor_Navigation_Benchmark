# State-action AMP：完整 200 秒导航对照

本报告只审核已完成的真实四人 200 秒记录，不执行仿真或修改评估公式。原四模型评分、失败帧、分母和历史报告保持不变。**保留 selected negative-guide20 为默认；新模型不满足导航验收。**

| 模型 | 摔倒 / 到达路线顶点 | 停留 actor-s | 同 episode 有效间隔 | Inclusive MMD² | 速度 W1 m/s |
|---|---:|---:|---:|---:|---:|
| selected guidance20 | 0 / 21 | 194.7667 | 23996 | 0.07238060 | 0.353368 |
| mixed state-only 2000 | 11 / 19 | 71.7667 | 23985 | 0.11648595 | 0.306563 |
| mixed state-action 2000 | 2 / 2 | 591.7333 | 23994 | 0.07372768 | 0.738402 |

新模型停留占有效间隔 **73.985%**。6000 个控制帧、24000 个 actor 帧全部保留；仅跨 episode 的差分边界排除，23988 个加速度三帧组有效。主 MMD 使用冻结 c274 参考、原特征、核和速度分箱；新模型仅 4 个速度箱有足够支持，原两比较模型为 5 个，不能将条件 MMD 视为同速分布或整体行走改善。

## 实际原生测量

[独立 raw 审核](../state_action_navigation_ground2000_review_v1/audit.json) **1148 检查通过**：6000 控制步 / 24000 物理子步 / 96000 actor 子步，真实 COM 重建最大误差 9.90e-7 m。双脚支持比例 81.625%，无脚支持 0.205208%；有支持时脚部 body-origin 切向速度代理均值 0.0422783 m/s。这是运动代理，不能等同接触点滑动、摩擦利用率或参考动作真实 COM。

静态碰撞报告 0 actor 帧；同伴接触 30 actor 帧、15 无序对帧，峰值 794.129 N。原始同伴力矩阵未保存，因此保留已发布通道及阈值一致性的审核范围。`status=passed` 表示完成并通过测量一致性检查，不表示动作或导航质量达标。

[原 idle/phase 报告](../state_action_navigation_ground2000_review_v1/stagnation.json)中 actor0 从约 50.83 s 到 200 s 长时间停留，目标误差均值 0.0831 m；执行参考平均速度 0.1059 m/s、原 5 秒计划位移均值 0.2138 m。这提示近时域指令与闭环响应均需要诊断，不能仅凭停留认定归一化或判别器为唯一成因。

## 来源与比较边界

新模型 checkpoint SHA256 `eeac037798d2244e0cd4f0a81b40f51020c2aa3320106c7f10620e8c2ddde526`。实际运行使用隔离 v3 / old146：92 个受管理模块执行、95 个已提交事件，两个 registered admission/entry 事件和 926 个隔离代码文件已核验。最终 import-ledger finalization 为 false 的事实保留。基线历史 source127、mixed source138 与本次 source146 分别绑定，没有声称整个源码相同。

混合样本 state-action 模型使用独立随机 D2019，未传入公开 state-only D/AMP RMS；专家 action 为 next-frame 归一化 PD 位置代理，不是测量的专家控制。对 mixed2000 的 D 输入、初始化和 AMP 统计转移均有变化；selected 基线还具有不同课程及一半训练预算。此对照不能归因为单一 D 输入因素，也不是作者实验的精确复现。

完整字典：[comparison.json](comparison.json)；所有 inclusive 评分：[evaluation.json](evaluation.json)；原生对象原样组合：[native_ground_comparison.json](native_ground_comparison.json)；根朝向与腕部描述：[descriptive_yaw_wrist.json](descriptive_yaw_wrist.json)。
