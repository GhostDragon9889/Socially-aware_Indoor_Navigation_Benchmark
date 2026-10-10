# 5D Diffusion → 30 Hz AMP：接口源代码审查

2026-10-04。只读 current185 源码、固定 TRACE/PACER 作者源码和既有轻量失败 JSON；没有加载权重、模型、SDK、CUDA、NPZ 或 sealed34，没有重复 D6 诊断、修改源码或启动新实验。

## 结论

**未发现可证明的 Hz、索引、根坐标或 S10 joint selection bug。** NEW b563 的 offline ADE/FDE 改善没有自动转化为闭环安全：既有完整记录为 OLD S1 F0/peer0/routes21，NEW S1 F1/peer70/routes9，NEW S10 F3/peer279/routes18。三组已保存计划的 target 相位检查均通过；这是来源有效和时间接线一致，不是动作质量通过。[实际完整摘要](../default_NEW_samples10_actual_runs_v1/actual_complete_comparison.json)。

最值得检查的是 **raw guided plan 与实际执行 reference 的安全合同不同**。另有明确的低层动态行人观测缺口和训练 heading 代理差异；后二者不能直接称私有 Nav 作者代码的 bug。

## 已核对的边界

| 接口 | 当前实际行为 / 判定 |
| --- | --- |
| 历史 | AMPWorld 存 91 个 30 Hz 帧，`[:, ::3]` 得 31 帧，含当前时刻，覆盖 −3..0 s；reset 前历史被 mask，不跨 episode 伪造可用历史。[amp_world.py](../../../CrowdSim/humanoid/amp_world.py:251) / [planner_inputs](../../../CrowdSim/humanoid/amp_world.py:346) |
| Future → task 时间 | 训练切片为 `frame+1 : frame+53`，即 .1..5.2 s；planner 只取前 50 帧并 prepend 实际 t0，输出 51 点 0..5 s。guide 索引为 `round(t/.1)−1`；AMP 查询 +.5,+1,..,+5 s 十点，dense reference 为 151 点 30 Hz。没有 .1 s 偏移。[paper.py](../../../CrowdSim/humanoid/diffusion/paper.py:616) / [返回值](../../../CrowdSim/humanoid/diffusion/paper.py:689) / [task_trajectory](../../../scripts/train_paper_amp.py:614) |
| 作者时间依据 | TRACE 固定 config `history_num_frames=30`、future52、dt.1；PACER补充 B.1 说明十个 .5 s 间隔的未来 5 s 目标。这解释 52 是模型窗口，不能把每次实际 AMP 任务也称 5.2 s。[作者 config](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json#L123) / [正文和补充](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf) |
| 根状态 / 坐标 | root position 与 `root_link_lin_vel_w` 同源；XY、速度均以当前 yaw 旋转一次，预测再旋转一次回 world XY。不是 root COM 速度与 link 位置混用。[simulator.py](../../../protomotions/simulator/isaaclab/simulator.py:936) / [world_to_agent_state](../../../CrowdSim/humanoid/diffusion/paper.py:62) |
| Map / neighbor | raster crop 与 feature query 同为 12 px/m、ego pixel(56,112)；地图与邻居都按同一个 ego yaw。邻居当前距离 ≤15 m 决定 mask；全空邻居有合法 zero pooling。本文未发现坐标重复旋转。[raster](../../../CrowdSim/humanoid/reconstruction_scene.py:182) / [prepare_context](../../../CrowdSim/humanoid/diffusion/paper.py:438) |
| Guidance / S10 | 当前部署 guide20、agent collision5/radius.6 m、scene collision10/radius.8 m。碰撞成本作用 world XY，S10 在 agent weight>0 时按全 scene 共享 sample index；不是每个人独立选不同未来。所有碰撞项仍是 soft cost。[deployment.py](../../../CrowdSim/humanoid/application/deployment.py:113) / [joint selection](../../../CrowdSim/humanoid/diffusion/paper.py:681) |

## 三个有价值、可证伪的缺口

### 1. 优先：raw 的碰撞成本不会约束 causal_accel 后的路径

`paper.py:271–286,665–688` 对 denoised raw XY 计算碰撞 guidance 和 candidate cost。随后 runner `567–587` 才调用 `CausalReferenceAdapter.build`；该函数逐人根据**上次执行的 reference**、raw interval velocity、位置误差和速度/加速度限制重建 151 点。它没有地图、同场景行人或 collider 输入，也不重新筛选适配后的候选。代码自身明确注明可能切过障碍、没有 peer/contact 保证。[reference.py](../../../CrowdSim/humanoid/training/reference.py:92)。所以 raw joint clearance 即使通过，也不能推出执行 reference clearance；这是一处可证明的合同缺口，**不是已证明本次70/279接触的成因**。

NEW S1 141.833 s 跌倒时，执行 target 误差 1.15539 m、raw target 误差 .609633 m；reference speed 1.16589 m/s、真实 root speed 1.93207 m/s，四子步无足支持、无当步行人或静态接触。这个例子支持“执行/物理跟踪并不等于 raw ADE”的边界，不能归因于碰撞或平滑器。[已有诊断](../default_newDP_failed_event_diagnostics_v1/README.md)。

**最小下一步：**先用已保存同一次 replan 的 raw/ref 检查同一未来时刻的最小 pair separation、地图 clearance 与 reference deviation，尤其首次 peer 接触前的同 episode 窗口；不运行模型、不改变指标。如果出现 raw-safe/ref-unsafe，可登记一个独立 `reference-adapter=none` 对照，仅移除后处理，保留同模型、seed、路线、guide、S1 和完整失败；清除被 parser 禁止的 inactive adapter 参数。raw 直接执行没有速度/加速度保证，结果可能更差，不能预期晋升默认。若差异不存在，则这项接触解释被否证，不继续扩展平滑器。

### 2. AMP 低层没有动态同行人环境输入

当前 `build_task_observation` 是 20 XY + 1024 terrain heights；`AMPWorld.height_at` 只有 ground、静态 cuboids 或 native provider，完全不查询其他行人的实时 body/root/velocity。故两个完全相同自状态和目标、只有旁人不同的输入，当前 AMP task observation 相同。[observation](../../../CrowdSim/humanoid/amp/observations.py:131) / [height_at](../../../CrowdSim/humanoid/amp_world.py:205)。高层 neighbor conditioning 不等于低层 reactive social awareness。

公开 TRACE/PACER §3.2 明确把附近 humanoid cuboid 写入 height/velocity map，并用 obstacle/interpersonal collision termination 训练；作者 terrain 源码存在 `root_points` → group heightfield 的分支。[TRACE/PACER 正文 §3.2](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf) / [PACER 固定源码](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/tasks/humanoid_pedestrain_terrain.py#L711)。Nav 的私有 obstacle encoder / crowd training 未披露，不能据此断言 Nav 必须用相同 3-channel 网络。

**最小验证：**未来 SDK-free 两演员 synthetic observation 对照，固定 ego/trajectory/static ground，只移动 peer，确认当前 1044D不变；再在独立 staging 的 `height_at` adapter 中使原1024采样包含动态 peer footprint，检查只有相应格值改变。现有默认 AMP 没受过该输入分布训练，不能直接加上部署便称有效修复；需独立 crowd-training 方案，优先级低于第1项，不在当前185/默认上试。

### 3. Heading 数据是速度代理，而部署输入是真实 pelvis heading

本地 ORCA 与作者 ORCA reader 都用 `atan2(vy,vx)` 作 heading（零速度时0），没有人体朝向测量；AMPWorld 则用实际 root quaternion yaw。这在转身、后退、侧移和近静止时是不同语义，且它同时决定 raster/local trajectory frame。[本地数据](../../../CrowdSim/humanoid/orca_corpus.py:178) / [作者 reader](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/dataset_specific/orca_peds/orca_dataset.py#L199)。Nav §IV-A 五状态中的 θ 为 heading；把物理 yaw 随意替换成速度角并不能称论文修正。[Nav §IV-A](https://arxiv.org/html/2606.26265v1#S4.SS1)。

**最小验证：**未来只从已有 capture 统计 `wrap(root_yaw−atan2(root_vy,root_vx))`，明确低速 heading 不可识别；按接触前同 episode 与非接触区间比较，不改 yaw、不重做D6。若无显著偏移/低速集中，则该 domain-gap 假设不获支持；若存在，再讨论独立条件表示消融，避免未经验证改变物理姿势或所有地图坐标。

## 范围与来源

上述检查不能把权重质量差异排除掉：b563 原训练是本地 ORCA-Maps 子集且训练预算有限；目前正式 full Maps+Interact、EMA、all-temporal40k 尚在运行，未来效果未知。当前 offline 指标衡量 raw 位置，不测 AMP physical tracking、contacts 或 support。建议先完成已登记正式模型评价，再按第1项做 bounded 已保存数据诊断；本报告不新增运行准入条件、不选择好帧、不申请并行 GPU。

缓存主来源为 `paper_fidelity_research_v1/trace_and_pace_text.txt`（§3.2、§3.3、补充B.1）及固定 `third_party/{trace,pacer}`；没有重复外网抓取。小身份记录见 [source_bindings.json](source_bindings.json)，不代表重新证明完整185 epoch。
