# 状态判别器消融：实际训练状态

2026-10-03 12:27:29 UTC 启动 fresh 2,000 轮训练，使用 256 环境、每轮 32 控制步；训练与监视器均在 13:28 UTC 实际退出 0，完成 16,384,000 条物理转换。完整动作、跟踪风格及四人 200 秒物理／风格评价均已完成。

## 已实际完成

- 四轮试训：root GPU session 50235，completion `e84e31`，实际退出 0；[GPU 记录](pilot4_actual_launch.json)。
- 完整 CPU 检查：session 3985，completion `5d7f39`，实际退出 0，2,087 项通过；[完整检查](../state_only_matched_q_training_audit_v1/actual_pilot4_v1/audit.json)与[退出记录](pilot4_CPU_actual_completion.json)。
- 试训检查点为独立只读文件 `step_4.pt`，SHA256 `0c038b28138ed64fab3453032b4b20b1e7127b1a7e35e7666a407fd465980191`；模型、两个 Adam、回放、RMS、随机状态与 q 采样状态均核对。[封存记录](pilot4_seal.json)。
- 完整训练：root GPU session 90291，completion `df0725`；CPU 监视器 session 77981，completion `cad301`，两者实际退出 0。终态 checkpoint SHA256 `97b2d2783ae73c968e2650e9ed189fd7254c3a0ec1a1b627ed2cd9bdd074a582`；三个独立快照与终态产物只读封存。[根封存记录](terminal2000_root_seal.json)。
- 完整终态 CPU 审计：session 17728，completion `64152a`，实际退出 0，2,076 项通过。[完整审计](../state_only_matched_q_training_audit_v1/actual_terminal2000_v1/audit.json)与[实际退出记录](terminal2000_CPU_actual_completion.json)。检查包含模型、两个 Adam、RMS、回放、随机状态、936 个源文件与专家采样状态；不声称物理轨迹逐位恢复。
- 全部 22 条验证动作：root GPU session 19040，completion `f9ef20`，实际退出 0；[实际 owner](tracking2000_actual_root_owner.json)封存全部 23 文件、11,982 帧。18 条成功、0 跌倒；world body 位置误差 .218969918m，旋转误差 .751017820rad。
- 原 CPU 风格及旋转分解：session 91146，completion `05cfb7`，实际退出 0；[结果](../state_only_matched_q_tracking_metrics_v1/README.md)。90D MMD² .05209593，未超过上一轮 q 的 .04156963；4 条跟踪失败与全部 idle 均保留。parent-relative joint 误差 .274913rad 是另定义诊断，不能替代 world body 旋转指标。
- 四人 200 秒闭环：root GPU session 96014，completion `b20884`，实际退出 0；[实际退出与完整捕获封存](../state_only_matched_q_physical_actual_runs_v1/fixture200s_actual_launch.json)。6,000 控制步全部保留；3 次跌倒、0 time-limit、0 静态接触、0 行人接触，路线顶点到达数为 `[7,8,8,8]`，合计 31。
- 原 CPU raw 审计：session 73760，completion `2f8258`，实际退出 0，203 项数据检查通过；[原始审计](../state_only_matched_q_physical_metrics_v1/raw/audit.json)与[退出记录](../state_only_matched_q_physical_actual_runs_v1/fixture200s_raw_CPU_actual_launch.json)。数据完整性通过，物理零失败要求因 3 次跌倒未通过。

## 输出与待完成

- 四人 200 秒的原 90D／45D 风格评价均实际退出 0：[汇总与收据](../state_only_matched_q_fixture_metrics_actual_v1/README.md)。MMD² 分别为 .06529463／.03235954；45D 占用 32/50 与 61/100 个聚类。保留全部失败、idle 和跌倒区间，仅按原规则排除跨 reset 的导数。因物理要求失败，不执行此设置的 native 扩展，不晋升默认模型。
- 原始训练输出：`output/diffusion_amp/amp_style_state_only_matched_q_isolated_v1`；独立快照为 `amp_style_state_only_matched_q_isolated_snapshots_v1/step_{500,1000,2000}.pt`。完整训练独立初始化，没有续训四轮试训权重。

## 方法和解释范围

复用 q 专家分层采样、固定 2,848 AMASS 数据、轨迹任务、奖励、物理、PPO 及相同 pre-step 历史时刻；仅采用 1,950 维状态历史判别器并直接迁移公开 PACER 的状态判别器与配套 RMS。未引入新的 Diffusion 候选。

这是用于诊断动作风格的 PACER 状态判别器消融。NavIsaacLab 论文的判别器包含状态与动作，故本实验不标为作者完整方法。不同判别器构造会消耗不同的初始化随机数，也不声称两次训练逐位配对。

训练和仿真 GPU 任务串行运行。首次原 owner 将 ToDesk 视频进程当作并行训练，试训在创建输出前被拒绝。新 owner 保留真实 NVIDIA／进程证据，仅允许明确识别的该视频进程；本次实际试训与完整训练启动时查询均为空，没有关闭远程桌面。完整模型与采样检查的数学不变；[薄适配身份](background_v2_source_binding.json)和实际检查明确记录新的执行入口。

训练中的跌倒 reset 属于探索过程；最终效果需独立动作跟踪、完整 200 秒物理和风格评价。默认模型保持不变，34 条封存测试输出不用于调参。
