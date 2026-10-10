# 当前最佳模型的 30 秒动作回放

视频：[pedestrian_motion.mp4](../../output/diffusion_amp/recorded_backward_penalty_overview_30s_v1/pedestrian_motion.mp4)。

- 来源：`shared_amass2848_backward_penalty_ground_support_200s_v1/physical_states.npz` 的前 30 秒实测物理状态，四名行人。原始完整物理实验为 200 秒、零跌倒；本视频只展示其中一段。
- 渲染产物：H.264，1280×960，15 fps，450 帧，ffprobe 时长 30.000000 秒。
- 文件：668,916 字节；SHA256 `03799ba8ac61ecd19d33eb8f8a0497301a2fa9e1045f9561d0121684c86ba4ab`。
- 回放记录：`physics_steps=0`、`policy_forward_calls=0`、`simulation_time_advanced_s=0`。它用于观察已发生的动作，不是一次新物理评估。
- RGB 帧非黑；抽查第 225 帧，四个带纹理的 SMPL 行人、地面、墙体和遮挡物正常显示。画面始终带有来源时间标记。

权重是已通过完整 22 条验证动作的负方向惩罚版本（成功 11/22），不是仍在开发的 AMASS 原始 XY 课程版本。视频不能证明作者八场景协议的论文效果，亦不替代完整动作误差、风格和接触报告。

完整渲染配置、输入及代码哈希见 [result.json](../../output/diffusion_amp/recorded_backward_penalty_overview_30s_v1/result.json)。
