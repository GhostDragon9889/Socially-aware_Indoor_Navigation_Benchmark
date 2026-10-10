# Learned XY 参考执行适配器与重规划跳变诊断

## 诊断依据

只读分析 `output/diffusion_amp/shared_obstacle_clean_sdf_5000_v2/physical_states.npz`；原run完成6000步、4人、200秒，status passed。只表示有限运行完成，不是稳定导航判定。

已排除reset/episode跨界的相邻样本。每15个control frame的全局重规划tick，`30*norm(target_xy[i]-target_xy[i-1])` p95为 **8.93017m/s**；其他tick p95为1.40373m/s。其他tick还包括11次reset触发的off-grid共享重规划。把这些重新划入后，精确重建411/411 reported forward calls，重规划p95=8.85120m/s，真正within-plan p95=1.39945m/s。

身体root XY有限差分speed p95为 **1.30385m/s**，不是8.93。原NPZ没有native root_vel，完整51-knot raw plan也未保存；因此无法事后重造原模型全部轨迹、原始实速初始化或执行一次虚构的物理改善。实际来源hash、分母与统计见[诊断JSON](target_jump_diagnostic.json)。

## 新API

```python
from CrowdSim.humanoid.training.reference import (
    CausalReferenceAdapter, ReferenceAdapterConfig,
)

adapter = CausalReferenceAdapter(ReferenceAdapterConfig(
    max_speed_mps=3.0,
    max_acceleration_mps2=2.0,
    initial_speed_policy="reject",  # Or explicit "brake" with violation reporting.
))
plan = adapter.build(raw_xy, reference_xy, reference_velocity)
# raw_xy: [B,51,2], world meters, 10Hz, t=0..5s
# plan.xy / velocity: [B,151,2], world meters / m/s, 30Hz
# plan.raw_xy remains an exact clone of the original learned plan.
# plan.raw_sampled_xy, deviation_m, initial_overspeed, speed_violation available.
executed = plan.state_at_step(age)  # integer or per-actor integer Tensor
next_plan = adapter.build(new_raw_xy, executed.xy, executed.velocity)
metadata = adapter.describe()  # JSON-safe deployment contract
```

配置还显式记录`control_dt_s=1/30`、raw knot0.1s、horizon5s、tracking time0.25s及初始speed判断数值容差1e-6m/s。可选times必须匹配相应均匀grid，禁止把51个稀疏点当151个实际control点。1.5m/s消融只改变显式speed bound，原始模型点保留。

该组件不持有RNG、world或隐式reference state，不能调用A*、SFM或physical teleport。episode初帧/reset用**真实当前**rootXY/velocity初始化；普通replan必须续接上次已经执行的reference XY/velocity。caller负责保存计划/state及原始点供评估。

## 限幅含义

将raw51个点线性采样到实际control grid。第k步模型区间速度为`(raw[k+1]-raw[k])/dt`；desired velocity为这个速度加上`(raw[k]-reference[k])/tracking_time`，先按向量norm限speed。再按向量norm限制`desired-v[k]`到`amax*dt`，得到`v[k+1]`，以`xy[k+1]=xy[k]+v[k+1]*dt`积分。

当前位置误差使用同刻raw[k]，避免额外的feed-forward一个control步领先。当初始速度已在speed球内，下一速度是球内两向量的凸组合；向量加速度限制直接成立，含浮点误差。不能把151点再压成51点并linear-interpolate后宣称所有30Hz加速度仍≤2。

**初始实速>max时speed与acceleration无法瞬时同时满足。** 默认reject明确列出actor ID；显式brake保留原v0，不夹physical velocity，在acceleration限制内逐步减速，并保存initial_overspeed和各个未来sample的speed_violation。不把期间超速伪装成全部满足max。

这只是**deployment adaptation**，不是论文定义。速度/加速度约束适用于reference，不能保证身体运动满足同一限幅。它可能lag、切角或穿过障碍，不能标collision-free、navigation-ready或论文复现。没有边界、contact、routearrival或稳定性保证；实际身体状态、raw/执行参考偏离、contact/fall/全局路线指标需独立完整评估。

## Root runner只读集成审查

本agent未编辑runner/paper.py；已检查root的实际调用：

- `describe()`/constructor/build返回字段匹配；plans用151@30Hz，raw_plans用51@10Hz。
- 普通replan取`plans[actor,plan_age]`和对应velocity，reset时只替换未初始化actor为真实root状态。
- route强制replan使用另一个mask，不通过伪改plan_age选取尚未执行的future state。
- 物理步后age+1，reward target、task observations和captured executed velocity使用相同dense时钟；raw_target单独使用10Hz同刻插值。
- 每replan完整raw/executed/velocity/violation/deviation存入`reference_plans.npz`，physical capture另存raw_target；没有只覆盖raw而丢失比较依据。
- 当前CLI仅允许该适配器用于evaluation，所以未改已有training resume参考状态合同。将来若开放训练，必须保存dense/reference_velocity/raw/初始化mask及合同，不能沿用51点旧snapshot静默恢复。

未发现当前API/时序阻塞。实际evaluation采用哪个speed/initial policy须由root显式记录；本审查没有执行任何Kit/CUDA或改现有checkpoint。

## 验证

**28 CPU tests passed**，4threads、无CUDA/Kit。涵盖恒速同相位、向量turn限幅、猛烈raw计划及exact保存、普通replan连续性、真实reset初始化、overspeed reject/brake、1.5m/s消融、per-actor state与无隐藏状态、稀疏/坏grid、NaN/dtype/shape拒绝、有效gradient及float32米坐标精度。Ruff Python3.12通过。[测试日志](tests.log)、[JUnit](tests.xml)、[格式gate](ruff.log)、[文件与caller快照](implementation.json)。
