# Social AMP：完整 22 动作跟踪评估入口（仅源码准备）

`evaluate_social_tracking.py` 为原 `scripts/benchmark_paper_motion_tracking.py` 的受限源码派生入口。原完整 28,945 B / `a89f05f94869695c9f908f8e9681f130a12b7e1f76085b6a5a8b91598efc616e` 固定；生产185、历史评估、指标实现均不修改。未运行模型、SDK、checkpoint、AMASS package 或评估；没有新效果结果。

## 保留的评估协议

使用原 N1、非共享、平坦无障碍 `AMPWorld`，每条动作仅在真实源 frame9 重置一次。原十个未来根 XY 位于 +.5…5s，尾部只保持命令端点；后物理步与同一源帧比较。序列中不因 fall / time limit 重置，不增加其它行人。原世界的120Hz四子步、30Hz控制、动作到 PD 的实际具名解码不变。

固定 v7 validation 全部22条、11,982个后物理步；拒绝 motion selection / clip cap / step cap。原初始重置帧留在捕获但不计主误差，真实失败和 idle 全留在原 SO3／身体位置指标及完整序列分母。入口检查完整性，不要求跟踪成功，不把论文报告值作为晋升门。

## 唯一接口改动

旧1044任务观测与 `PaperAMPTrainer.load` 替换为已封严格 [Social checkpoint loader](../social_checkpoint_inference_staging_v1/checkpoint_inference.py)（`cd227f5d…`）及共享 raw Social producer。显式不可变 checkpoint SHA 必填；加载器验证完整 Social 模型、四 RMS 和保存的元数据，最终仅持冻结 encoder／actor／输入 RMS／57维动作投影。训练计数来自 checkpoint 声明，并非该加载器独立观察的物理训练证据。B=1不改变训练 actor_count；不恢复 Adam／replay／训练 RNG。

直接读取 checkpoint `provenance.AMASS`，保留原训练／整个验证 split 的 parent-source 不重叠检查，不伪造 legacy `dataset` 字段。原 reset 的参考速度／十帧 AMP history 由 actor 声明的 DOF velocity frame 决定。

producer 在原 `reset_reference_clip` 后收到 `begin_episode()`；每次 actor 输入来自同一当前 COMMON 真实状态，额外验证 producer 快照与原 `actual` 十项姿态／速度字段逐值一致。N1 stable actor ID=0、scene ID=0；显式 root grid／root heading／`static_center_ground_z`／exclude ego，须与冻结 actor 合同相符。地形回调为原平坦世界 `world.height_at` 的绝对静态高度，不补 +1.8 或另做 clip／缩放。

任务 heading 奇异时仅允许真实物理 fall 且本 episode 曾存在真实 defined heading 才延续旧任务坐标；不改物理 quaternion，也不改原 self357 heading 数学。缺失上下文或健康奇异姿态拒绝。该坐标延续不等于实际身体 heading，也不改善旋转指标。

## 捕获与来源

保留原全部 root／world24 body／DOF、同帧 reference、动作、fall、time-limit、episode、AMP history、查询相位与尾部保持字段。新增 `observation_raw_self`、`observation_raw_task`、owner、真实观测 pelvis quaternion／ground／up／fall／heading defined／fallback／used heading、actor／scene／episode／age／physics counter。

`observation_*` **是 actor 使用的控制前状态**；同一行原 pose／reference 是该动作后的物理末态。初始行 observation 是 frame9 实际重置状态，没有有效动作。两种时序明确分开，不能把 observation quaternion 当后物理步 quaternion。

报告标记新的入口、完整原 source 与依赖 SHA、精确派生 SHA、冻结 NN snapshot 与 device／动作57投影、producer 几何及时序。原来源不是逐 bit 原 runner 执行声明。19项一次性精确字节替换，加1项显式异常 guard，必须逆还原全原源码；六个未改顶层函数逐源码／AST一致，嵌套真参考 trajectory 保持源码一致。异常 guard 内正常控制／metric／NPZ／return 的 AST 与未包裹版本完全一致。原 metric／聚合／reset／corpus／布局完全复用，没有重写物理或误差算法。

输出要求新空目录，保留原异常报告与一次 cleanup。控制／正常保存发生意外 `BaseException` 时，另存 `motion_ID_partial_failure.npz` 后 bare rethrow 原异常，不补帧或重试物理。各 channel 独立长度保留；若中断发生在半帧追加中，报告逐 channel 数量、形状及任何无法 stack 的字段，不伪造对齐。另存最后已返回的 owned preaction packet 为 `latest_owned_observation_*`，它可能已出现于完成帧，不能认作失败动作或后物理步。部分捕获不计算成功指标，也不加入完整22条聚合。保存本身失败会记录 archival error，不能替换原控制异常。若实际执行中断，报告是失败与已完成序列的部分聚合，不能冒称22条完成；源码准备不证明 native仿真或论文复现。

## 未来执行

仅在 root 安排实际评估后，使用已封 Social checkpoint 的真实路径及 SHA：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/social_tracking_evaluation_staging_v1/evaluate_social_tracking.py \
  --checkpoint /ABS/READONLY_SOCIAL_STEP.ckpt --checkpoint-sha256 ACTUAL_SHA256 \
  --deterministic-evaluation --verify-package-hashes \
  --output /ABS/NEW_EMPTY_SOCIAL_TRACKING_OUTPUT
```

本入口仅提供完整22低层跟踪接线。固定四人200s的独立源码入口已另写，正在审阅；Nav native场景接线及全部实际物理验证仍待完成；不调用新的 grouped training World，不复制本轮共享训练协议。

## 已完成源码审阅

Root完整源码读取a14f14／d29b15／601111及独立Paper peer均已完成，Required=[]／Optional=[]。Author stdlib精确派生／compile e4b697与Ruff0a5bf0退出0；独立peer的唯一静态核验411d58退出0，未导入或运行候选程序。当前源码79fd24b2与README准备只读封存；这些检查不证明模型或物理效果。
