# 相同候选与时域：原始／可执行参考的联合选择

此目录把新增候选评分接入独立的真实物理评估入口。新入口与隔离runner已经落盘；当前仅有源码与CPU接口证据，尚未运行这两个设置的SDK或物理仿真，不能声称接触、跌倒或动作风格改善。

## 依据与本地证据

[NavIsaacLab IV-A/B](https://arxiv.org/html/2606.26265v1)使用五秒轨迹和AMP整身控制；[TRACE/PACER §3.3/4](https://xbpeng.github.io/projects/Trace_Pace/Trace_Pace_2023.pdf)采用闭环重规划、候选筛选，并研究控制器value guidance。当前实现的是本地约束后几何评分实验，不是作者value-function引导，也不是未发布作者配置的等价实现。

[三个封存物理对照的真实CPU诊断](../causal_reference_clearance_CPU_v1/README.md)发现raw到reference有局部新增净空不足，执行目标前缀peer总计也有改善。尤其S10的reference peer proxy比S1少，但真实接触更多。因此不能将此代理等同于碰撞或断言它造成跌倒，更不能凭此直接删除适配器。

## 唯一对照变量

- `matched_raw50`：对同一原始候选的五秒几何评分，再执行选中的causal reference。
- `reachable50`：先为同一候选银行逐个构造causal reference，对这份实际将交给AMP的参考几何评分。

两臂保留一次原S10、100步DDPM生成与原52点clean guidance；共享同一50个未来评分点（.1..5.0s）、10个waypoint、goal20／peer5／static10及半径.6／.8。原适配器速度1.5m/s、加速度2m/s²、overspeed brake及普通重规划参考连续性均保持。共同场景只选择一个sample index。局部原实现筛选覆盖.1..5.2s，历史S10结果单独保留，不能把它充作本次matched_raw50对照。

两臂未来物理状态发生分化后，Diffusion历史和候选也会分化；相同seed及零新增RNG不保证全程生成逐位相同银行。同候选的隔离比较仅指同一次输入／采样银行，不冒充闭环反事实配对。

## 接线与结构

- [可复用几何模块](../../../CrowdSim/humanoid/training/reachable_selection.py)：原adapter和原cost复用，保留全部候选及成本；[8项真实CPU检查](../reachable_reference_selection_CPU_v1/actual_CPU_tool_receipt.json)。
- [独立公开CLI](evaluate_candidates.py)与[延迟runtime callback](selection_runtime.py)：资源预检阶段只导入stdlib及既有SDK-free接口；实际callback才导入Tensor几何模块，不构造或调用NN。
- [隔离物理runner](evaluate_runner.py)仅有[七个必要替换片段](runner_exact.patch)，逆变换恢复整份73954字节原driver和整份AST，来源见[派生记录](runner_source_derivation.json)。它复用原唯一App/world、planner调用、AMP actor、actions、reward、reference状态初始化、routes、reset及control loop。

[实际未来命令](commands.json)固定AMP62c4、DPb563、v7 AMASS、相同四人fixture、seed7和200s。命令中的预检与GPU分开记录；必须待当前Diffusion40k训练及其原定固定AMP评价完成后串行调度新GPU实验。

## 本轮已实际验证

- 几何模块8项CPU检查及实际FP32四人、十候选callback的2项CPU检查通过；[callback真实记录](actual_callback_CPU_validation.json)。所有原adapter字段完全相等，world字段、九字段形状、成本守恒及随机状态检查通过。首次pytest外部ROS插件缺lark的失败保留，禁用自动插件后运行这两个case；实现没有因此改动。
- 两模式的真实SDK-free资源预检分别实际退出0，工具块`e508e7`／`54974e`，用时.464／.463s；[matched原始输出](matched_raw50_actual_SDK_free_preflight.json)／[receipt](matched_raw50_actual_preflight_receipt.json)，[reachable原始输出](reachable50_actual_SDK_free_preflight.json)／[receipt](reachable50_actual_preflight_receipt.json)。验证actual S10、AMP62c4、guide20／peer5／static10、四人6000步及新source身份；未创建物理输出目录。
- [独立完整源码审查](peer_review.json)Required0；[root来源封存](root_source_freeze.json)绑定实际检查、预检及四份新源码，原185源码没有变化。Ruff真实块`1bde97`退出0。

[物理／候选审计入口](audit_candidate_plans.py)现已补齐，并经[独立源码审阅](candidate_audit_peer_v1.json)Required0及root身份／AST／Ruff检查。它复用原物理计算，重建全部候选和6000控制步，分别显示数据一致性与实际跌倒／接触零门槛；场景YAML读取前后核对真实预检SHA。[补充计划v2](commands_and_audit_plan_v2.json)绑定已封存的reader、注册、待实际运行的owner模板和原两臂命令，旧commands与freeze保留历史状态。**reader尚未执行，两个物理输出目录和实际owner均不存在**；真实RGB-D渲染也继续排在GPU串行队列。

## 原始记录与结果要求

原reference archive保存control_step、actor_ids、selected_sample、raw51、reference151、velocity及原全部flags。新增九字段保存所有十个候选的world raw51、reference151/velocity、goal/static/pair/joint cost、waypoints及scene_ids；unsafe和未选中的候选也保存。pair cost按原unordered pair的半份分给两端，只是记账；joint cost重复到每个actor行，不可再次对actor求和。

没有保存完整52×5候选状态银行到新增archive；保存的是common50 XY及真实t0，因此不能用新archive复算原5.1/5.2或未保存yaw/velocity guidance。物理身体位置、足部支撑、真实contacts及全部6000 controls继续由原记录器保存。

需要实际root工具闭合、来源和完整捕获封存后，才运行原物理计算与新增candidate一致性核对，随后比较完整跌倒、静态/行人接触、到达、idle、轨迹误差及90D/45D风格。所有失败保留；本目录不提供自动晋升默认或零失败保证。封存34条测试动作仍不用于本轮优化。
