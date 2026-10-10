# 独立低层课程 AMP 训练审查

## 完成与边界

真实 PhysX 训练完成 **1000 次 PPO/AMP 更新、2,048,000 个物理转移**。64 个隔离环境，每次32控制步；120Hz物理、30Hz控制，共32,000控制步、1066.667秒并行仿真、68,266.667行人秒。运行耗时 1150.033 秒，1780.82 转移/秒（含采样、物理、优化和保存；并非纯物理速度）。原始日志1000条完整且全部数值有限。

这是独立低层命令课程训练；**Diffusion 在训练 rollout 中调用0次**。`step_5000.ckpt` 被绑定用于另外的部署合同，没有消费其规划输出。地面128m、spawn margin48m、episode300步=10秒，5秒命令窗，默认不反射、不修改物理 root。公开 PACER 参数基础上显式使用 smooth 适配：目标速度独立均匀采样0–3m/s，标量/向量加速度上限2m/s²、heading速率2rad/s；不是公开尖转生成器逐位复现。[TRACE/PACER §3.3](https://arxiv.org/html/2304.01893v1#S3.SS3) 明确两级独立训练。

初始化移植公开 PACER actor、critic、task encoder、观测和value归一化矩；判别器为**独立随机初始化 state_action**，2019输入=10×195姿态历史+69应用动作，57个有效动作坐标。没有移植公开1950维state-only判别器。公开权重SHA256：`bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615`。专家动作为下一帧 mocap 姿态的 inverse-PD target 代理；不能解释为真实专家控制指令。

## 跌倒与超时分母

训练全程 **87 次跌倒、6745 次10秒超时、6832 次完成episode/reset**。跌倒定义为pelvis相对地面高度<0.45m或root up-z<0.1；运行最低pelvis 0.353787m。跌倒率 **1.2744/1000行人秒**；完成episode中跌倒占 **1.2734%**。超时不计入跌倒。尚在运行的64个episode未作为完成episode分母。未按“87/2,048,000”冒充独立样本失败概率。

## 分段奖励与风格

每格为250次等大小rollout的均值；事件计数取分段首尾累计差。轨迹组件是对齐的单步位置奖励，不能当作部署路径到达率。

| 更新 | 跌倒 | 超时 | 总奖励 | AMP奖励 | 轨迹奖励 | D expert/policy | 前向速度m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–250 | 45 | 1646 | 0.8030 | 0.4247 | 0.9633 | 0.7870/0.1730 | 1.9446 |
| 251–500 | 16 | 1708 | 0.7435 | 0.3440 | 0.9768 | 0.8304/0.1447 | 1.5751 |
| 501–750 | 21 | 1701 | 0.7039 | 0.3007 | 0.9755 | 0.8503/0.1276 | 1.5122 |
| 751–1000 | 5 | 1690 | 0.6980 | 0.2900 | 0.9780 | 0.8604/0.1235 | 1.4124 |

末次更新 D expert/policy=0.8661/0.1152，AMP均奖=0.2639，轨迹均奖=0.9782。判别器仍明显区分专家和策略，**尚不能确认风格收敛**。总奖励使用AMP+0.5×任务和；signed power 为同一120Hz子步的 projected active joint effort×末端关节速度，4子步时间平均，非隔离motor torque。保留论文打印的direction符号，未在本轮静默替换。

![原始训练曲线](independent_amp_training_review.png)

## 固定 checkpoint 与恢复完整性

审核对象为不可变 `step_1000.pt`，**121,316,621 bytes**，SHA256 **`e0d6ed4a9acb39c581bed92fab722a426bf23c34a3949bc48712cc13bf04c333`**；与immutable manifest一致。CPU `weights_only=True` 读取成功，unsupported globals为空；121个Tensor叶子全部有限。strict加载模型、两个Adam optimizer和四套normalizer后逐Tensor精确相等；policy/D optimizer分别16/6份完整参数状态，Adam step均6000，moment形状匹配参数。更新/转移/runner control counter分别1000/2,048,000/32,000。

课程私有CPU RNG uint8[5056]、plans_generated=140262与物理/历史/计划/episode状态已保存。两个独立课程实例从同一保存状态生成下一批64人命令，所有输出和下一RNG状态逐Tensor相等。Python、NumPy、Torch CPU和corpus RNG用私有实例验证；CUDA RNG只验证保存结构，没有恢复或初始化CUDA。CPU校验不证明PhysX接触缓存/solver的逐位续跑等价。

源码归档的48个文件逐一与冻结manifest的bytes/hash匹配。审查时当前树有2项后续变化（runner与CLI新增causal_accel部署实验）；本次result没有该适配选项，Diffusion调用为0。当前树hash不是运行时hash，本报告保留原启动归档，不追溯改写provenance。

## 4更新实际续训探针

探针共8192物理转移：先2更新再从checkpoint续训2更新，逻辑control_step128、4.2667秒并行仿真、273.0667行人秒。第一进程跌倒0，第二进程跌倒2，超时0。末result的`actual_control_steps=64`及计数描述最后进程；不是8192转移的单次不中断物理过程。课程计划/RNG/训练计数随恢复保留。

## 下一验收

完成且状态有限不等于低层风格收敛，也未测试共享障碍/社交部署。本轮没有scene、没有Diffusion rollout；需要用固定解析命令和相同Diffusion5D/共享场景评估新controller，联合物理contact、跌倒、目标误差与train-only姿态K/MMD。没有论文原始2848条动作ID、作者权重及八场景完整manifest，不能声明作者实验复现。

完整字段/分段component/value/KL、normalizer矩和证据hash：[JSON](independent_amp_training_review.json)。可重跑脚本：[review_independent_amp_training.py](review_independent_amp_training.py)；本次只创建CPU审查工件，没有修改训练代码或checkpoint。
