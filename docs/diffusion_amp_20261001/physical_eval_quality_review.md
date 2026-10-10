# Diffusion + AMP 物理评估独立质量复核

日期：2026-10-01。仅审阅源码、论文与已有 root 实际产物，运行两个窄 CPU 回归；未启动 Kit／GPU，未修改业务源码、模型或运行中的评估。源码与输入哈希、完整指标见 [证据 JSON](physical_eval_quality_review.json)。

## 已完成的四人 200 秒评估

实际输入为 AMP `amp_training/step_1000.pt`，搭配单独训练的 ORCA social diffusion `social_diffusion_training/step_1000.ckpt`；它不同于 AMP 训练绑定的 AMASS diffusion，runner 通过显式 `--allow-evaluation-planner-change` 记录了两份 SHA。场景为新建 cuboid fixture，不是作者八场景。

| 真实记录 | 结果 |
| --- | --- |
| 控制采样 | 6,000 帧 × 4 人，30 Hz；状态均有限 |
| 跌倒／时限重置 | 16／0；各人跌倒 3、5、5、3 次 |
| 跌倒分母 | 16/24,000 frame-agent = 0.0667%；16/20 observed episodes = 80%；4/4 人至少跌倒一次 |
| 路线顶点到达 | 0、0、1、0，总计 1 |
| A* 查询失败 | 271 次；失败时条件目标保持当前物理位置 |
| 同时刻计划跟踪 | 平均 0.20594 m，RMS 0.25332 m |
| 计划终点到条件目标 | 平均 4.68303 m，95% 分位 8.07621 m |
| 未换目标、未重置段的目标距离减少量 | 合计 −19.98174 m，即总体没有持续向这些目标推进 |
| 地图 root-point 代理 | 0 帧 root 进入 cuboid／越界 |
| 30 cm 行人 XY-disc 代理 | 1,494/24,000 = 6.225%，22 个事件 |
| 实际 PhysX 接触 mask | 未记录；真实地图／行人接触指标仍未知 |

复核状态见 [runner result](../../output/diffusion_amp/shared_obstacle_evaluation_1000/result.json) 与 [CPU metrics](../../output/diffusion_amp/shared_obstacle_evaluation_1000/pedestrian_metrics/evaluation.json)。reset mask 与 episode ID 边界均为 16、差异为 0；排除重置跳跃后的各人行走距离为 173.45、84.46、191.85、157.21 m。

**程序完成不代表稳定导航。** 立即 reset 将跌倒状态只保留一帧，不能把很小的逐帧跌倒率与作者未公开分母的 Table II 跌倒率直接比较。低局部目标跟踪误差也不能覆盖全局路线未到达的问题；这轮实测还不能证明收敛或论文复现完成。

## PACER 温启动与 AMP 目标

- `training/trainer.py:215` 从 SHA 校验的公开 no_shape 模型迁移完整 actor、critic、task encoder、sigma、obs/value moments。CPU 实际权重测试确认新 actor 的确定性输出与公开参考网络一致。
- 当前 self357、task1044、任务 10×0.5s 至 5s、69 维 PD action 与 PACER 的接口一致。按 body/DOF 名称映射、hand/toe 冻结和生效 action mask 都保持显式合同。
- 新判别器接受 history+action（10×195+69），公开 PACER 的 state-only D 未迁入。这符合本项目按论文 Eq2 建立新 D 的选择，D reward 的 softplus×2 变换和专家 action 的下一帧 mocap PD 代理仍是公开论文未确定的重建假设。
- 当前物理 rig、质量、limits、PD gains、120 Hz 子步与原 PACER IsaacGym 训练分布并非已证一致。因此数学接口相容、温启动可执行，不等于已经获得稳定导航 controller；该初始化也不是作者 NavIsaacLab AMP 权重。

## 有符号功率与奖励上涨

独立从本地论文第 7 页核对 Eq5：`−0.0005 Σ Γᵢωᵢ` 没有绝对值／clamp。负机械功会给正奖励。Eq6 的 `1−exp(min(0,u)/1.8)` 会奖励负向速度，与论文“防止倒走”文字相矛盾；当前 `paper_equation` 保留印出符号。这是公式本身的歧义，不能静默改符号后继续称为同一实验。[论文](https://arxiv.org/abs/2606.26265)

首轮 update1000 的 reward=3.919805，其中 AMP=0.209416、trajectory=0.951552、energy=6.458343、direction=0.010883；能耗经外层 0.5 权重贡献 3.229171，占总分 **82.38%**。AMP reward 比首轮下降；总分上升不能据此说明动作风格或稳定性改善。

另有独立信号错误：旧 implicit PD `data.applied_torque` 是最后子步**之前**的裁剪 PD 估算，配的是最后子步**之后**的速度，错位 1/120s；不是实测 drive 输出。细节见 [扭矩语义审查](torque_semantics_audit.md)。本轮已运行 eval 导入的是旧模块，因此不能用后来修改的 source retroactively 解释它。

root 新建的 `AMPWorld.PaperPowerSimulator` 每个真实120Hz step/update 后读取 `get_dof_projected_joint_forces` 与同子步 joint velocity，四个功率取平均 W，再经 `signed_power_w` 进入相同系数公式。代码保持原控制器／物理频率，并用 `power_stage` 阻止训练 resume 静默换信号。这修正了错时刻和估算冒充实测的问题；projected force 仍包括自由轴传递力、限位等，并非单独纯电机 torque。真实 getter 形状、顺序、有限值、符号与性能由 root 的物理 probe 验证，本审阅者没有运行 SDK。

原 metrics.jsonl 有 1,009 行，含 checkpoint resume 后的重复 update；fall/time counters 在新的运行段重新开始。分析应按 checkpoint／阶段区分，不能直接将累计列逐行相加。

## 下一轮命令与风险

下面命令使用**已有冻结首轮 AMP**，测量新 projected power，仅用于独立评估。它不会把新 reward 信号追溯写入旧训练模型。若先完成新 power-stage 训练，应把 `--resume` 换成该阶段冻结模型并使用新的 output 目录。

```bash
PYTHONPATH=.:src ../IsaacLab-Protomotions/.venv/bin/python scripts/train_paper_amp.py \
  --resume output/diffusion_amp/amp_training/step_1000.pt \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training/step_1000.ckpt \
  --allow-evaluation-planner-change \
  --num-envs 4 --evaluate-steps 6000 --episode-steps 6001 \
  --replan-steps 15 --planner-batch-size 64 \
  --scene-config configs/diffusion_amp/shared_obstacle_validation.yaml \
  --synchronized-replanning --compile-diffusion-forward \
  --power-source projected_joint_effort_substep_mean --capture --seed 7 \
  --output output/diffusion_amp/shared_obstacle_evaluation_projected_review
```

评估后仅 CPU 分析：

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/evaluate_paper_pedestrians.py \
  --states output/diffusion_amp/shared_obstacle_evaluation_projected_review/physical_states.npz \
  --manifest data/diffusion_amp/amass_manifest.json
```

风险与验收边界：

1. 默认 episode300 会每10秒时限 reset；6001 避免6000步末帧人为截断。旧本轮 episode6000 的实际时限 reset 为0，因为每个人都提前跌倒过。
2. 单个 A* 5m arclength lookahead 端点通过 soft diffusion guidance 进入采样；guidance 不是到达／全轨迹碰撞约束，最终 t0 也不硬投影端点。当前实测终点误差大，不应只延长运行就预期修复导航。
3. A* 的0.35m clearance 与当前全身实际形状并非同一几何合同；靠墙 root 能仍在free域，却无法作为合法clearance start继续规划。actual body contact mask缺失。
4. SMPL overlay只作显示，控制仍由真实69DOF PhysX state；capture每15控制步记录一帧，MP4为2fps，共400帧／200s。它不保存每个物理帧的视觉接触细节，完整状态以30Hz NPZ为准。
5. `_capture_frame` 使用有限Kit render updates并关闭 playSimulations；未额外步进物理。视频创建、RGB有效性与渲染耗时仍需实际产物验收。
6. compiled forward 是显式数值实现选择：原验证噪声序列一致，XY最大差异亚毫米、yaw差约0.0025rad；不能称 bit-identical。synchronized planning保留最长0.5s间隔，但改变抽样batch与重规划时刻。
7. 本轮发现NPZ／MP4序列化异常可能保留statuspassed；root已在审查中将except强制设failed。执行当前真实AST exception handler的CPU注入得到failed／exit1。此回归仅验证状态处理，未模拟真实视频编码失败。

## 窄 CPU 验证

已执行真实公开 PACER checkpoint actor迁移一致性，以及有符号制动／反向奖励的可微回归：**2 passed，2.03s**。CUDA不可见，未导入Kit。命令与边界见JSON；未重复全套CPU／GPU或物理评估。剩余作者精确2,848清单、八场景、定制Diffusion/AMP权重、完整评价分母仍未具备，不能将这些新建实验标为作者严格复现。
