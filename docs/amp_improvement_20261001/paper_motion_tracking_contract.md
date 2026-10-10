# 全身体运动跟踪基准：协议与有限验证

日期：2026-10-02。实现文件为 `CrowdSim/humanoid/motion_tracking.py`、`scripts/benchmark_paper_motion_tracking.py` 和对应 focused tests。当前源已冻结；完整22个旧 validation clip 的物理运行由 root 执行，本报告不提前声明其结果。

## 方法与公开资料范围

[NavIsaacLab 论文 IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3) 使用 AMASS 行走类动作评估运动姿态，报道平均刚体位置误差0.218 m、旋转误差0.343 rad；任一时步刚体平均位置误差超过0.5 m判序列失败，成功率64.6%。作者具体2848动作ID、测试clip清单、旋转误差公式和指标聚合总体尚未取得。

本基准明确使用世界坐标下24个body **原点**的欧氏距离，以及 XYZW 四元数的符号不变 SO(3) geodesic angle。逐步先平均24个body，位置均值严格 `>0.5 m` 时锁存一次失败。mean指标包括失败后的实际帧，不猜作者是否只统计成功序列；`maximum_step_mean_body_position_error_m` 也不能直接当作者未说明聚合方式的最大刚体误差。公开参考权重/自己训练controller及当前neutral SMPL/PhysX设置不被标为作者同一实验。

## 闭环控制与时间协议

- 固定checkpoint actor deterministic mean，原生 PD action decoder；actor只接收实际物理 self state 与未来10个**reference root XY**位置（+.5..5秒），不接收参考3D关节姿势或身体朝向目标。
- 每clip只在 source frame9 初始化一次真实 root pose、root velocity、DOF pose/velocity；10帧历史由当前实际初态及前9帧mocap组成。原始参考应用一个固定XYZ平移，与物理初态对齐。
- 每个30Hz控制步先查询当前相位的未来XY，执行真实物理，再与**相同post-step reference frame**比较；不做逐帧pelvis/yaw对齐或最佳相位搜索。
- 不在跌倒、达到tracking failure或time limit后teleport/reset；继续测量已请求序列，分别报告fall事件和位置失败。
- 第一个reset frame不计误差均值/成功帧，action_valid=False。reference clip尾部之外的5秒命令只hold最后XY；指标在真实clip结束时停止，不延长成功分母。
- 默认要求至少1秒实际可测量帧，且reference明确30Hz；50Hz未resample的motion拒绝。本协议initial9帧不属于评估区间。
- `--max-steps` 是诊断截断；partial序列 tracking_success=None，不计完整成功率分母。完整eligible clip才进入成功分母；fall是否发生另列，不能用两个口径拼成一个成功率。

## 资源、坐标与泄漏门禁

preflight在Torch/SDK导入前校验container/LFS pointer、实际checkpoint SHA、manifest schema/IDs/时间窗口/输出目录，调用 `assert_parent_split_disjoint` 识别 `#frames=` segment alias。实际actor加载后再绑定其**training manifest SHA与training package SHA**，检查整个被评估heldout split的parent source与训练parents disjoint；支持新训练curated manifest对照旧heldout manifest，不能只检查文件路径字符串或被选中的一个clip。

当前processed reference的24-body名字由项目canonical PACER/SMPL顺序假设；package没有逐帧body-name标签，此假设写入protocol。physical capture则记录实际common与native名字并按双射重排，四元数XYZW，body angular velocity世界坐标。

最新脚本显式使用 loaded actor `TrainingConfig.expert_dof_velocity_frame` 创建 AMASSCorpus，再核对 corpus实际provenance。

| actor配置 | reference / initial-reset DOF速度 | 实际native速度 |
|---|---|---|
| `legacy_stored`（旧checkpoint默认） | 现存文件声明或明确legacy假设的stored frame；旧bank实际parent_joint | child_joint |
| `child_joint`（新style profile） | 当前帧local rotation将parent stored转child；已声明child的不再转换 | child_joint |

这是有记录的坐标协议变化，旧结果不被悄然改写。body position/rotation指标不依赖速度frame命名；初始化velocity和训练reference history会受其影响。实际旧validation frame9的CPU adapter比较：pose/rotation/DOF位置完全相同，DOF速度 RMS差0.253246 rad/s；这只是单样本两坐标表示差，不是物理velocity误差。[原生D6速度坐标审计](../style_improvement_20261002/dof_velocity_semantics_audit.md) 用同瞬时32步物理angular velocities验证child frame，保留其实际证据与范围。

## CPU 与真实pilot证据

23 focused CPU tests PASS（1.55s），Ruff三文件PASS；含真实小checkpoint加载冻结、真实旧actor/AMASS CPU forward、parent leakage、heldout provenance、非有限action pre-physics拒绝、initial-only reset mock时序、30/50Hz与partial分母、四元数符号/90°测量。mock闭环不等于真实仿真。

root实际完成 `motion_tracking_legacy_native_velocity_pilot_v1`；本agent随后用CPU NumPy独立重算其raw arrays。

| pilot结果 | 实际观察 |
|---|---|
| clip | validation motion0，HumanEva/S3/Walking_3，实际完整可测580步 |
| capture | 33帧：1个reset初态 +32个控制步（1.0667秒） |
| initial body position / rotation error | 8.0849e-7 m /1.1201e-6 rad |
| 32步mean body position / rotation error | 0.1883278125 m /0.5925527050 rad |
| maximum step mean position error | 0.3266895409 m |
| bounded falls / tracking threshold crossings | 0 /0 |
| full sequence success | None，完整成功率分母0 |

SHA、finite、reference frames9..41、reset/metric/action mask、episode恒定、冻结12 DOF输出0和两种误差公式逐数组验证均通过。此pilot不是完整序列成功，也未达到论文姿态复现结论。pilot捕获driver SHA为 `282c54dc...`，**早于**最新显式frame传播版本；不把旧protocol补写成child_joint。

CPU证据：[pilot独立重算](paper_motion_tracking_pilot_cpu_verification.json)、[JUnit](paper_motion_tracking_cpu_tests.xml)、[实际actor forward](paper_motion_tracking_real_actor_cpu.json)、[实际preflight](paper_motion_tracking_real_preflight.json)、[速度frame adapter](paper_motion_tracking_velocity_cpu.json)、[汇总合同](paper_motion_tracking_contract.json)。

## 当前冻结版本与下一物理验证

当前driver SHA256 `a89f05f94869695c9f908f8e9681f130a12b7e1f76085b6a5a8b91598efc616e`；pure metric module `da482dcd565d42ef9f6165606428ad510f077bdd68c3fab09c73a840149eac56`；tests `a770fe7c0581801c4fc21f1258cef0308e75432fd011eac97ff21ecb853d6789`。完整22 validation运行的捕获source/protocol/result应以其自身report为准，不假设其他同时工作的模块SHA与旧pilot相同。

接下来用完整clips汇总成功/失败/跌倒与frame-weighted误差，再独立比较新child-frame profile。Torch deterministic配置不提供PhysX/Warp bitwise保证。未公开作者split、shape分布和聚合协议仍是复现缺口，当前测试集与参考资源必须逐项声明。
