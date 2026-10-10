# 独立低层轨迹课程

## 方法来源与边界

[Trace and Pace §3.3](https://arxiv.org/html/2304.01893v1#S3.SS3) 明确 TRACE 与 PACER 独立训练，运行时再闭环结合；其实现细节说明 PACER 训练使用程序化2D轨迹。官方 [PACER 源码](https://github.com/nv-tlabs/pacer/tree/baeb6241f03fc9d78370a64e3c2dcef9d830d97f) 中，`pacer_no_shape.yaml:53–57` 设置10个0.5s未来点、速度0–3m/s、标量speed变化上限2m/s²、尖转概率.02；`humanoid_traj.py:106–114` 使用101个轨迹顶点和2rad/s普通heading增量。

固定官方 revision：`baeb6241f03fc9d78370a64e3c2dcef9d830d97f`。上游作者为 NVIDIA CORPORATION & AFFILIATES；其仓库 [LICENSE](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/LICENSE) 为 [CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。本模块的 `pacer_knots` 数学分支根据该公开生成器的程序化 reset 逻辑适配，保留来源和许可归属；不能将上游来源代码重新解释成主仓库许可证。

**默认 `smooth` 是显式本地课程改进，不是逐项复刻官方 PACER 训练或 NavIsaacLab 作者实验。** 官方尖转直接替换一个knot的heading增量为±π，标量speed限制不等于向量加速度限制；默认课程把目标角slew限制为2rad/s，并额外限制命令速度向量每.1s变化≤.2m/s。每次计划独立抽取 `speed_goal ~ U[0,3]`，5s内保持；从实速向目标以≤2m/s²变化。初始速度仅作为平滑起点，目标分布不随物理跟踪速度变化，避免静止初态通过random-walk强化停滞。

普通与罕见sharp目标角仍被采样，实际heading命令slew以及向量加速度限制随后应用。以当前world yaw/运动方向为起点；零速时使用root yaw。公开生成器原来第一段heading与speed均重新随机，且使用整episode的101顶点；本模块默认5s／10Hz／51顶点，并在每次调用实际root重新锚定。这些差别均不隐含声称source完整parity。

## API 与恢复

实现：[trajectories.py](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/training/trajectories.py)。

```python
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryCurriculum

curriculum = PacerTrajectoryCurriculum(seed=7)
prediction = curriculum.build_plan(world.state(), actor_ids=ids)
plans[ids] = prediction.xy       # [B,51,2]，绝对world XY米，t=0:.1:5s
goals[ids] = prediction.xy[:, -1]
snapshot["curriculum_state"] = curriculum.state_dict()
```

输入为common RobotState的`root_pos[B,3]`、`root_vel[B,3]`、`root_rot[B,4]`，world米、XYZW单位四元数。可用唯一非空actor IDs选择子集。输出还含`velocities_xy[B,51,2]`、`times_s[51]`、`speed_goals_mps[B]`、sharp目标mask、反射mask、起点速度clip声明及method metadata。首个XY恰为当前root；函数不写任何physical state，不移动或teleport角色。

使用私有CPU `torch.Generator`，不消费Torch全局RNG。`state_dict`含严格schema/config、CPUuint8随机状态与累计生成计数；恢复必须先保持完整runner plans／plan ages／physical history，然后在下一次采样前恢复课程状态。若`trainer.load(map_location="cuda")`把嵌套RNG tensor搬到CUDA，runner须仅把`generator_state`转回CPU再调用`load_state_dict`，不能丢弃或重seed这个状态。缺失状态／配置改变拒绝续训。

默认不处理地面边界，需要caller为整个episode和未来horizon留足物理地面；速度上限约束命令，不能宣称实际物理root必≤3m/s。显式可选`reflection`只镜像命令，声明反射会破坏加速度界且实际起点不得在域外；本轮root训练选无反射和扩大地面。当前保守CLI clearance为`3*(episode_seconds+5)`，300步=10s需要45m；64env网格跨度28m时extent128／margin45或48满足该几何检查，margin40不满足。真实防跌倒效果仍需SDK验证。

部署保留独立的 Diffusion5D planner；`state_action`与现有论文式reward仍为主训练。课程不改变AMP专家动作含义、判别器结构、奖励、权重或历史checkpoint。

## CPU 验证

[test_pacer_trajectory_curriculum.py](/project/EXP/SangforAndSaut/ProtoMotions/tests/test_pacer_trajectory_curriculum.py) **25项通过，Ruff通过**，结果见 [trajectory_curriculum_tests.xml](trajectory_curriculum_tests.xml)。

- 真正生成128人的5s平滑计划，检查speed≤3、velocity差分accel≤2、非空sharp分支、位置差分一致与physical原始Tensor未改。
- 从实际官方`TrajGenerator.reset`抽取AST，禁用real/fixed/slow路径，仅替换随机源为同一CPU Generator；相同knot设置下，`pacer_knots`位置数值与完整RNG消耗对照通过。此测试证明正常程序化公式／draw order，未运行旧SDK、官方flags的其他分支或其duration/查询API。
- 私有RNG完整续采样逐Tensor等值；全局Torch随机状态未变；actor子集与invalid输入／恢复合同拒绝。
- 零速128人、seed7：独立目标均速1.46977m/s，.5s命令均速.81882m/s，5s均速1.41887m/s；maxspeed2.98086、max向量accel约2.000001（float32舍入），未来root距离最大12.57845m。该样本为真实CPU命令生成，不是物理行走结果。

## 集成审查

只读核对root runner：课程不构建／调用Diffusion planner，replan写absoluteXY／goal／age0；snapshot保存独立课程RNG，恢复plans与物理history后恢复课程，再恢复global trainer RNG。初始plan_age触发首次replan，所以`new_goals`在课程分支不采样不会导致首步消费全零plan。课程配置由saved state严格校验；source改变要求fresh实验。

已向root指出CUDA map_location随机状态与margin40不足45m两个具体问题，由root整合修正。本说明只记录CPU/source门禁；后续真实PPO课程训练与部署Diffusion效果使用独立输出和冻结来源验收。
