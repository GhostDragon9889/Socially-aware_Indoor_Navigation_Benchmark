# NEW planner：仅将候选数1改为10

2026-10-03。本准备使用同一 production185、NEW Diffusion `b56346fb…`、默认 AMP `62c4edc1…`。唯一方法变量是 `diffusion_samples: 1 → 10`；输出目录另设。保持四人、200秒／6000控制步、seed7、guide20、15步／0.5秒重规划、alpha0、30Hz控制／120Hz物理、同一场景和路线。未修改生产、默认、旧封存文件或当前 q 训练；没有启动 GPU、SDK 或反序列化模型。

## 选择依据

已完整封存的 [OLD／NEW 实际比较](../default_DP_pair_actual_comparison_v1/current185_OLD_NEW.json)：OLD 为0跌倒、0静态接触、0行人pair-controls、21路线顶点；NEW 为1跌倒、0静态接触、70行人pair-controls、9路线顶点。开放环 ADE/FDE 改善没有转化成本场景物理质量改善。

[原事件诊断](../default_newDP_failed_event_diagnostics_v1/README.md)显示：唯一跌倒行人3在141.833秒无足部支持，且当帧及同episode±2秒没有peer/static接触；三段peer事件涉及另外的行人。行人1 idle比例66.3277%、最长28.7667秒，其同期58份raw五秒计划朝lookahead投影均值为−.411601米。这些相关观测没有证明单一原因。本轮检验额外候选与既有筛选是否改善表现。

## 主来源对照

截至本次查询，Nav arXiv 页面仅列2026-06-24的v1。论文明确五状态、历史／邻居／地图条件、未来5秒轨迹转局部target，没有披露候选数、guide20或重规划15步。十候选属于TRACE公开方法扩展，不能称Nav作者精确超参数。[Nav论文](https://arxiv.org/html/2606.26265v1#S4)、[版本记录](https://arxiv.org/abs/2606.26265)

| 项目 | TRACE/PACE公开材料 | 当前本地实现 |
|---|---|---|
| 多候选 | controllable animation抽10份后筛选 | 现状1份，本轮10份 |
| 重规划 | 物理动画跟随2秒再重规划；预测实验的1Hz是另一协议 | 15控制步＝0.5秒，本轮保持 |
| 条件历史 | 3秒、10Hz，使用PACER实际运动 | 91个30Hz真实状态取`::3`，31帧含当前；reset前不可用历史有mask |
| 低层target输入 | PACER接收未来5秒的10个XY点，间隔0.5秒 | timed-path目标也为10个`.5..5s`点；仅有输入时间形状一致，未证明控制策略相同 |
| 目标和速度 | TRACE预测5秒；物理动画实验采用9秒特定时刻全局waypoint | 本地10个`.5..5s` timed-path目标；causal_accel限速1.5m/s、限加速2m/s²，本轮保持 |
| 引导 | clean trajectory guidance，并研究PACER价值函数引导 | guide20是本地有量纲目标损失；不能与官方目标权重或classifier-free strength直接比较；本轮无critic引导 |

出处为[作者论文§3.3、Implementation Details、supplement实验设置](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf)及[官方10候选示例](https://github.com/nv-tlabs/trace/blob/main/configs/eval/nusc/target_pos_perturb.json)。预测示例参数不冒充物理动画或Nav参数。论文PDF缓存 SHA 为`dc539e42766e573a6ccdeb526857278f3e716255a99e251523d63aa3243869ec`。

## 现成代码的真实行为

- [生产部署参数](../../../CrowdSim/humanoid/application/deployment.py:111)固定samples1；外部薄入口只替换这一forwarded值，并调用原actor-only工厂及原driver。
- [driver](../../../scripts/train_paper_amp.py:557)给整场4人scene_id0，planner batch8容下全部4人。[筛选](../../../CrowdSim/humanoid/diffusion/paper.py:681)将场内goal＋scene损失和unordered peer损失相加，向全部4人赋同一个argmin候选编号；不会拼接各人的独立最优轨迹。
- [clean引导](../../../CrowdSim/humanoid/diffusion/paper.py:267)对候选损失求和；S增加不会隐式把guide20除以10。[上下文](../../../CrowdSim/humanoid/diffusion/paper.py:591)每4人编码一次，再repeat至40候选batch；权重只有原来的一份。
- [真实历史](../../../CrowdSim/humanoid/amp_world.py:314)在物理步后更新，[planner输入](../../../CrowdSim/humanoid/amp_world.py:349)取实际历史和可用mask，没有把先前计划当成已发生历史。

[源形状估算](memory_source_estimate.json)：重复map grid由1.53MiB增至15.31MiB；S10的一份noisy tensor为40.625KiB，pairwise XY差为65KiB。网络输入梯度／临时激活随候选batch增大；100去噪步串行。估算采用现有默认52×5、conditioning256、grid32×56×56形状，未读取checkpoint配置或测GPU峰值，不能证明显存足够或总时间恰好10倍。

## 可执行参数与剩余工作

[run_parameters.json](run_parameters.json)保存完整argv；`root_GPU_command_after_q_training_and_evaluation_only`是待root独占执行的精确数组。[入口](run_samples10.py)复用原preflight、actor-only factory、native scene request及driver。实际SDK-free子进程已观察exit0／0.514秒，见[闭合记录](actual_CPU_preflight_closed.json)和[真实preflight](actual_SDK_free_preflight.json)：actor request完全相同，parsed/forwarded仅samples和output变化，未建立未来输出。

[CPU reader草案](CPU_reader_metadata_adaptation_draft.json)尚未登记未来actual owner／CPU reader，不具备执行准入结论。root须在q训练／评价结束后建立独立进程owner，绑定同185及外部入口／参数，再运行完整6000步和原raw CPU数学。所有跌倒、接触、idle、reset、错误、OOM及非零退出必须保留；不能静默减样本。candidate数变化会改变随机数消耗，不能声称S1原候选与S10首候选位级配对。

可证伪比较：固定场景报告F／static／peer、每人路线顶点和原idle事件；若失败仍存在或进展未改善，该单变量假设不获支持。原native准入仍需完整6000步 **0跌倒／0静态／0行人接触**，MMD不参与滤除失败。9与21路线顶点分别作为NEW和OLD观测参照，不新增质量门槛、不承诺改善、不自动推广默认。
