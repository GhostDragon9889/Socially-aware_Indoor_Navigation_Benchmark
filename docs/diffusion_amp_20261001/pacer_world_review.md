# 独立 AMP world / runner 合同复核

日期：2026-10-01。本报告是 root 正在实现的新训练链的只读复核快照，
不是最终源码哈希或收敛验收。没有启动 Kit 或修改 root 所有的 world/runner。

## 确认问题

1. **冻结坐标向 action-conditioned D 泄漏专家标签。**
   `AMPWorld.expert_batch` 将 hand/toe expert action 的 12 个坐标设零，而最初
   `PaperAMPTrainer.act` 的 applied action 只 clip，不清零。runner 在物理 PD 解码之前
   计算 `amp_reward(history, applied_actions)`；PD 解码才把这些坐标冻结。
   因此 D 可以只用这些实际不驱动物理人体的坐标区分 policy 与 expert。
   已交由 training owner 修复：policy sample/log-prob 保留完整 Gaussian，送 D 的
   effective applied action 按相同冻结合同投影，并验证/序列化该约定。

2. **训练 resume 未验证当前 dataset / physics 合同。**
   runner 构建当前 `--manifest` corpus 与 world 后，只核对旧 checkpoint 的 diffusion hash。
   改用另一训练 manifest / 源 bank 后仍可恢复，但 trainer 的 provenance 保留旧 dataset；
   world 的时钟/limits 等也未与原合同比较。建议非 evaluation 的 resume 验证
   manifest/source hash、split、rig/names/limits/control Hz，或显式声明 finetune
   并记录新的来源。已通知 root；当前默认训练命令没有更换数据，未受此分支影响。

3. **逐 body mass 的名称合同需要重排。** `physics_manifest` 直接将
   `root_view.get_masses()` 的 backend 序列与 common `body_names` 并列记录。
   当 `data_conversion.body_convert_to_common` 不是 identity 时会误标各 body 的质量，
   总质量不变。应按 common 转换重排或同时保存 native body 名称。本审查没有启动 Kit
   读取 native 名单，因此这里只确认代码缺少该转换，未将条件错误宣称为实测错序。

## 核对通过的逻辑

- 物理 120 Hz、decimation 4，单次 `Simulator.step` 的控制间隔是 1/30 秒。
  训练 runner 的下一状态轨迹奖励先将 plan_age 加一，再取下一物理时刻的插值目标。
- corpus 取 `[t,t-1,…,t-9]`，expert PD action 使用 `t+1` 的参考 joint position；
  past/future 边界计算不会跨 clip。原 bank 的 30 Hz 由 manifest 筛选门槛限定。
- reset 不额外推进物理。Lab 写入 joint/root 后使缓存失效，`body_link_pose_w` 通过
  `_ensure_fk_fresh()` 调 `update_articulations_kinematic()`；重读状态刷新 FK。
- reset 平移 root 世界 XY；AMP 使用 root-relative body/key positions 且不含 root height，
  因而这个平移不会错误改变参考历史特征。最当前帧被实际物理状态替换。
- diffusion 的 91 帧 30 Hz planar ring 经 `::3` 成 31 帧 10 Hz，最后一帧是当前状态。
  reset 将旧 episode 历史标为 unavailable，不会当作真实历史送 planner。
- current semantic DOF 顺序与 PACER 完全一致；expert 通过实际 PD scale 逆映射，
  scale 为零的冻结坐标用零值，避免除零。`action_to_pd` 输出物理顺序，
  `Simulator.step` 无第二次策略 action scale。
- rollout 在 reset 前 clone transition，保留 terminal/truncated next-value；
  episode counter 与 AMP history 在 reset 分支更新。

## 已有外部物理证据

root 的 `output/diffusion_amp/physics_probe/result.json` 记录 4 env、90 步、3 秒的
纯 PACER/SMPL/PhysX 接口运行；最低 pelvis 高度 0.4411 m，有 4 次 fall reset。
这验证了接口能执行、状态有限及 native mass/limits/gains 可读取，**不证明稳定行走**。
本报告不将 `status=passed` 扩大成收敛或论文 benchmark 复现。
