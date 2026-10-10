# 200 秒行人验证掉落：分析与修复

完整机器证据：[pedestrian_fall_analysis.json](pedestrian_fall_analysis.json)；真实动作选择：[pedestrian_walking_reset_manifest.json](pedestrian_walking_reset_manifest.json)。

## 失败确实存在

`output/resource_validation/pedestrians_200s/result.json` 为 failed。861 个已记录物理步骤、28.7 秒、6 次运行中 reset 后，下一步因 pelvis 小于 0.35m 终止；已保存帧最低为 0.350046m。不能把最后一帧仍略高于门槛解释为 PASS。

- env2 在 25.5667s 重置，骨盆约 0.9465m；26.9s 为 0.8914m，27.2333s 为 0.7225m，28.7s 为 0.350046m。这是重置后逐渐失稳，未观察到重置瞬间穿地。
- 最后 45 帧与其他行人的最近 XY 距离为 7.6047m，证据不支持近身行人碰撞是这次掉落的触发因素。
- 其余三人的最低骨盆高度分别为 0.71094、0.87209、0.85737m。
- 原 npz 只有 time、root_pos、dof_pos、reset_mask，缺少 root quaternion、接触、扭矩、motion IDs/time、目标高度与 reset reason，因此无法独立确认唯一物理原因。

## 已确认并修复的控制与 reset 问题

1. **目标骨盆高度反馈错误。** 原导航 override 每帧把未来目标 Z 直接设成当前实测 Z。`build_sparse_target_poses` 保留位置误差的 Z 分量，因此下降时该条件的垂直误差仍恒为零，目标没有提供恢复站立的方向。新增显式配置 `humanoid.pelvis_height_above_ground`；独立行人配置设为 0.93m，目标为各未来 XY 的 terrain 高度加 0.93m。其它未配置的运行入口保留原行为；filler holding slots 保留自己的 holding pose。
2. **一般 AMASS 库被当成步行重置分布。** 原每次 goal reset 从 345 条一般验证动作中随机选首帧，包含跳跃、踢腿和 jiggles。最后一次 env2 重置后的 DOF 与 clip315 `MPI_mosh/00058/jiggles_2` 首帧最接近，RMSE0.0568rad（第二候选0.131rad）；该首帧 vz 为 −0.386m/s。这个 ID 是通过已推进一步的 DOF 推断，原运行没有保存实际 ID，不能声称精确恢复了随机状态。
   独立行人 profile 现在通过实际 MotionLib 源路径 patterns 选择 9 条前向 Walking 动作：`[0,10,19,26,234,235,244,253,254]`。接入前检查有限姿态/速度、首帧 pelvis0.8–1.1m、绝对垂直速度≤0.15m/s、root tilt≤15°；若不满足或没有匹配则明确报错。使用现有 MotionManager 持久排除与 reset 时间零契约，不改原始 `.pt`，不创建或冒充论文的2,848动作清单。
3. **reset 返回的目标观测过时。** 原 `BaseEnv.reset` 已计算返回 obs，导航 wrapper 才刷新新路径的 SFM 速度。现已在刷新导航状态后重建 reset IDs 的观测；非 reset 环境和原 info 保持一致。

## 测试与修改范围

- 28 项 CPU 回归通过，耗时3.37s；真实 AMASS 通过 `weights_only=True,mmap=True` 读取，并使用真实 MotionLib 查询与 MotionManager 采样。seed7 共10,000次抽样只得到上述9个 ID，验证原库 stat 未改、排除能在权重更新后继续生效。
- 新测试覆盖下降状态下地面相对目标、不连续 env IDs、filler保留、非有限地面报错、重置后新目标观测、既有排除/子集重置、失败时不改变 sampler、缺失/低身/倾倒/下落首帧明确拒绝。
- 四个改动 Python 文件 Ruff `--target-version py312` PASS。既有 builder 的 unused import 和一行两个 imports 同时做了最小清理。
- [JUnit](pedestrian_fall_repair_tests.xml)；原文件精确快照在 `before_fall_repair/`，源码 SHA256/before SHA256 见 JSON。

修改：`CrowdSim/nav_manager.py`、`CrowdSim/world/builder.py`、`CrowdSim/config/pedestrian.yaml`。新增：`CrowdSim/humanoid/motion_reset.py`、`CrowdSim/tests/test_pedestrian_reset_targets.py`。

## 尚需真实验证

CPU PASS 不证明 200 秒稳定性。必须用相同 seed7、4人、30Hz、6000步骤重新执行实际 PhysX 验证，并保持 pelvis0.35m失败门槛；任何再次失败都要保留失败状态及原始轨迹。建议同时保存 motion IDs/time、root quaternion、目标高度、reset reason 与接触信息以区分控制域、急转弯或接触动力学风险。

当前控制仍是 SFM 的 pelvis 条件 + 公开预训练 SMPL MaskedMimic。本次修复既不训练 AMP，也不代表论文定制 gait diffusion/AMP 资源已补齐。导航抑制 clip_done 后 core motion time 继续增长、参考查询会 clamp，旧日志“motion loops forever”不准确；本次未在缺乏训练依据的情况下新增时间 wrap。`simulate.py`、SDK环境、原AMASS、checkpoint与失败检测均未改。
