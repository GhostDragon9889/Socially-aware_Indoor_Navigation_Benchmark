# 外部 SFM target generator：已完成第一 CPU slice

[源码](sfm_target_generator.py) 复用冻结 `4b1fe12f…` 的完整原 `Social_Force` class AST，不更改力/TTC/clamp 方程；只绕开无关 cv2、绘图和 demo imports。默认 FOV=False。`ttc_prediction_horizon_s=2` 与输出 5 s 分开；输出时间轴为 `float64 arange(51)/10`。

[一次小 fixture](run_CPU_fixture.py) 在目标 Python 3.12 / Isaac Sim 6–Isaac Lab 3 环境中以 CUDA hidden 实际执行，**exit0，13项断言全部通过**，没有导入 Torch、IsaacLab、IsaacSim、Omni 或 pxr。[实际结果](actual_CPU_fixture.json)、[闭合记录](actual_CPU_closure.json)。验证米制 EDT/Y flip/墙面力方向、51 含真实 t0、同快照同步及行置换、排 self/cross-scene、输入和真实 routes 不变、近目标 .1 s 积分、域外停止和 attempted XY、初始域外不夹取、unknown 负样本保留，以及非零 map yaw 拒绝。

保留了两个真实前置失败：ORCA env 没有 SciPy，因此尚未执行预测；Anaconda 实际 12/13 通过，但第一次域外 fixture 的 10 m/s 初值被原边界排斥转向，错误地期待域外退出。原 [失败报告](first_actual_fixture_failed.json) 和 [fixture原字节](first_fixture_source.txt) 保留。只把这个合成域外 case 改为明确的 adversarial 100 m/s 初值，证明真实 t0 不被截断、未来退出分支确实停止；不把该速度当作真实行人运动。这个 case 的 **initial_overspeed=True**（初始速度100 m/s，配置上限1.5 m/s；来源是实际 fixture 源和已通过的 t0 断言）。实际原力+clamp 后触发 `step=1 / t=.1 s` 域外事件，attempted XY 保留在实际结果；不是物理质量证据。adapter 方程没有为断言修改。

这是本地未来目标转换的 CPU 合同验证，没有物理导航/全身稳定性或作者 SFM 参数复现声明。生产185、当前训练源和所有旧冻结结果保持原样。

## 第二 slice：evaluate-only 接缝源码待审

新增 [隔离 runner](evaluate_runner.py)、[SFM factory/runtime](sfm_runtime.py) 和 [评估入口](evaluate_sfm.py)。尚未执行该接缝或其 preflight；等待 root 整读后的一次 SDK-free 检查授权。没有实现 physical CPU reader、owner 或新训练 bundle，也没有启动 SFM 物理实验。

[精确 diff](runner_exact.patch) 仅 11 个必要替换边界（包括 staging ROOT 修正）；[来源说明](runner_source_derivation.json) 记录逆转后整个文件字节和 AST 等于原 `scripts/train_paper_amp.py / d8b85401…`。默认 `target_generator_factory=None` 保留原 DP 分支；physical world/actor/actions/rewards/reference/routes/reset/control body未重新实现。新增 factory 限 evaluate-only + saved actor，SFM 每次 replan 从真实 `world.planar()` 生成一次全场景51 XY，再按原选中 ID/chunk 进入同一个 causal reference adapter。

SFM 外部入口不要求 Diffusion 文件；私有原 CLI 副本只去掉这一个必需项，内部旧 parser 的 Diffusion路径仅为不会访问的语法占位符。原 preflight/actor/native 逻辑在私有 namespace 重用，资源函数只去除无用 DP record，没有修改生产函数。实际 parsed source/runtime 记录 `sfm`、DP checkpoint=None、构造/forward/sample=0；`selected_sample=0` 明确是单个确定性目标索引。完整 AMP 历史训练 DP provenance 保留，实际目标源独立记录，不伪造 `evaluation_planner_change`。

native 使用现有正确 scene factory 初始化后的 provider 原始 `map==254` / .1 m / ROS top-left grid；flat fixture 用同一 cuboids/bounds 的 cell-center footprints 构 raw occupancy，未知/域外 blocked，不用膨胀 `_walkable` 或 actor crop。SFM 顺序 waypoint 预测只动私有 index，原真实 post-physics arrival 和失败计数不变。去掉 DP 构造/采样后不宣称全局随机序列或物理轨迹与 DP bit-identical。
