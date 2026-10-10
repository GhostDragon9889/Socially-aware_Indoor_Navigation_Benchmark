# 共享 Social AMP 2000 更新：真实评价结果

**S0 训练、跟踪、风格和四人 200 s 测量均已完成，模型质量未通过。** 没有达到或超越论文效果，没有替换部署默认模型。S1 Pop-Art 配对实验已通过实际短检查并开始全新训练。

## 数据与执行身份

- 固定真实 AMASS v7：2,848 片段、1,853 个原始动作文件、200 个受试者、523,272 帧。清单 SHA256：`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`。这是本地重建清单，不是已恢复的作者私有动作 ID。
- 全新 seed7、256 人/64 组、32 控制步/rollout，2,000 次更新；实际 16,384,000 条物理转换、8,192,000 个专家窗口、256,000 个物理子步。
- 训练真实退出 0，终态 CPU 审计通过，202 个模型/Adam/RMS/replay/rollout 张量有限。终态权重为独立只读文件，SHA256：`8dcedc67e8b0cafb8fb9d53ca4ca34a09a712ff90e88b8572996892e85d516f8`。
- 独立动作评价使用按原始动作文件与训练分离的完整 validation22；不读取 Test34 数组，不按结果挑选片段或检查点。

执行依据：[训练退出与权重身份](style_improvement_20261005/shared_social_evaluation_v1/training_execution.json)、[终态 CPU 审计](style_improvement_20261005/shared_social_evaluation_v1/training_CPU_audit.json)、[跟踪与风格实际退出记录](style_improvement_20261005/shared_social_evaluation_v1/tracking_style_execution.json)。

## 独立动作评价

| 项目 | 实际结果 | 解释 |
| --- | --- | --- |
| 全身跟踪 | 0/22 成功，22/22 出现摔倒 | 全部 11,982 个后物理控制步保留；片段内不因摔倒重置 |
| 同相位平均刚体位置误差 | 1.865725 m | 不做最佳帧、根平移或旋转匹配 |
| 同相位平均刚体旋转误差 | 2.138779 rad | 原始 30 Hz 参考与实际刚体同步比较 |
| 摔倒帧 | 11,298/11,982 | 摔倒和静止进入主要风格评价 |
| 90D 等支持速度桶平均 MMD² | 0.499120，5 个支持桶 | 固定参考、相同速度桶和已有核；不是论文 MMD |
| 物理根速度 W1 | 0.811324 m/s | 速度分布明显未匹配真实参考 |
| 45D 姿势 MMD² | 0.863276 | 15 关键点、固定尺度/朝向处理、固定参考和抽样上限 |
| 固定 K50/K100 coverage | 0.84 / 0.66 | 不重新拟合中心；覆盖率不能单独证明姿势自然 |
| 固定 K50/K100 entropy | 0.573853 / 0.499683 | 保留全部合法同片段姿势，失败不筛除 |

完整文件：[跟踪结果](../output/diffusion_amp/shared_social_terminal2000_evaluation_v3/full22/result.json)、[90D 风格](../output/diffusion_amp/shared_social_terminal2000_evaluation_v3/style/evaluation.json)、[45D 风格](../output/diffusion_amp/shared_social_terminal2000_evaluation_v3/style/pose45.json)。

NavIsaacLab IV-C 的位置/旋转误差、跟踪成功率和 Table II/III 是方法对齐目标。作者评价动作、八场景任务清单及风格指标参数仍未完全匹配，本地 MMD² 与论文 MMD 不直接数值比较。[NavIsaacLab 原文](https://arxiv.org/html/2606.26265v1#S4.SS3)

## 四人 Diffusion + AMP 的完整 200 s

同一个正式 Diffusion40k、同场景、四人、6,000 控制步全部保存，实际进程退出 0，原生 owner 全部释放。独立 CPU 审计重新核对实际数组的力阈值、episode/reset/终止时相、路线计数和四物理子步：

| 项目 | 实际结果 |
| --- | --- |
| 物理终止及重置 | 956 次；0 次时间上限终止 |
| 四人到达路线点数 | 0 / 0 / 0 / 0 |
| 观测暴露量 | 24,000 actor-controls、800 actor-seconds |
| 1 N 阈值的实际静态接触 actor-controls | 0 |
| 无序同行人接触 pair-controls | 0 |
| 完成片段平均长度 | 25.0513 控制步 |
| 排除跨重置的根速度均值 | 0.9050 m/s；23,040 个同 episode 间隔 |
| 支撑脚部刚体原点速度代理均值 / P95 | 0.26357 / 0.88991 m/s |

零障碍/同行人接触伴随大量跌倒和零路线进度，不能说明导航合格。脚部原点速度代理包含所有有支撑的物理子步，不是接触点滑动。论文跌倒/接触率的私有分母仍未恢复。

依据：[实际退出记录](style_improvement_20261005/shared_social_evaluation_v1/navigation200s_execution.json)、[独立数组审计](style_improvement_20261005/shared_social_evaluation_v1/navigation200s_CPU_array_audit.json)、[CPU 实际退出记录](style_improvement_20261005/shared_social_evaluation_v1/navigation200s_CPU_audit_execution.json)。原退出记录的 `independent_array_audit_performed=false` 是审计前的时间快照，后续审计单独保存，没有修改原记录。

## 生命周期与失败记录

真实 N1 零动作 8 控制步/32 子步和四人 30 控制步均退出 0。N4 的 5 个接触视图、24 个 SDK 传感器及借用 backend 在 App 关闭前的 weakref 存活数均为 0。这证明这些完整构造路径正常关闭，不证明步态或导航合格。

首轮沙箱运行没有 CUDA，部分 App 构造后退出 -11；第二轮完整跟踪被原 Python 来源检查错误拒绝 TorchScript 函数。两次失败保留，第二轮正常清理。修正采用精确 TorchScript 对象、限定名称和固定源文件身份校验，原观测数学、参考时钟与指标不变。[修正和失败证据](style_improvement_20261005/shared_social_evaluation_v1/source_preparation_v2.json)

## 价值训练稳定性诊断

最终保存 rollout 的 CPU GAE 重算与实际训练均值差 `0.0000305`、标准差差 0，按显式 `atol=1e-5, rtol=1e-6` 接纳。

| 最后 rollout 的量 | 结果 |
| --- | --- |
| 实际每步奖励均值 | 0.868132 |
| GAE 回报的奖励贡献均值 | 3.559685 |
| GAE 回报的 critic 自举贡献均值 | -336.899414 |
| 预更新 value 范围 | -32,925.211 至 28,114.498 |
| 预更新 value RMS 历史均值/方差 | 48,332.437 / 17,708,344,474.386 |
| 最后 100 更新实际 PPO minibatches | 平均 2.05；100/100 更新触发 KL 停止 |

这是保存的真实训练数据分解，表明 critic 自举远大于奖励贡献；尚未证明单一原因。原训练在更新 value RMS 后没有同步补偿 critic 输出层，且 KL 停止同时终止 actor/critic 联合优化，分别需要检验。[完整 CPU 诊断](style_improvement_20261005/shared_social_evaluation_v1/training_stability_CPU.json)

## 下一轮唯一数值变量：Pop-Art

[Pop-Art 原论文 Proposition 1](https://arxiv.org/pdf/1602.07714)给出更新目标均值/尺度时保留未归一化输出的仿射补偿。本地实现复用已有 population RMS 和 epsilon1e-5，只补偿 scalar critic 最后一层；Adam 状态保持，未宣称优化器轨迹不变。这是明确的本地改进实验，不是作者未公开训练配方的替代证明。

5 个 CPU 数值测试和 2 个完整 Social trainer 的 CPU fixture 测试已通过；fixture 不包含 AMASS、SDK 或物理训练。后续实际检查已完成：

- 8 人、1 更新、32 条真实物理转换；训练/终态审计/新权重原始观测推理均退出 0。
- 256 人、2 更新、16,384 条真实物理转换；训练/终态审计/256 行原始观测推理均退出 0，202 个保存张量有限。
- 使用 256 人短检查权重，四人 Diffusion 导航 30 步实际退出 0，显式 Pop-Art GPU 前向及生命周期接入通过。
- 独立 full22 和导航入口按已认证训练结果/CPU 审计显式识别权重格式，拒绝数字来源不匹配或隐式 legacy 转换；14 个 stdlib 评价测试通过。

依据：[8 人真实退出](style_improvement_20261005/shared_social_evaluation_v1/popart_smoke8_execution.json)、[256 人真实退出](style_improvement_20261005/shared_social_evaluation_v1/popart_scaled256_execution.json)、[256 人独立加载](style_improvement_20261005/shared_social_evaluation_v1/popart_scaled256_inference_execution.json)、[四人接入](style_improvement_20261005/shared_social_evaluation_v1/popart_navigation_smoke_execution.json)。这些是接线检查，尚未证明动作质量改善。

S1 使用原 `training.json`、固定真实 2848、seed7、256 人、2,000 更新；唯一数值变量为 value RMS 更新时的 scalar critic 输出补偿。全新初始化，不续训 S0 或短检查权重；预先选择终态 `step_002000.pt`，继续用完整 validation22 和固定风格/路线/物理评价。[训练前登记](style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_registration.json)、[实际启动记录](style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_launch.json)。终态训练与独立效果当前尚未完成。

| 工作 | 项目入口 |
| --- | --- |
| 训练，显式选择 legacy/popart | [train_shared_social.py](../scripts/train_shared_social.py) |
| Pop-Art 数值补偿 | [value_normalization.py](../CrowdSim/humanoid/training/value_normalization.py) |
| 来源绑定与可逆训练派生 | [social_popart_training.py](../CrowdSim/humanoid/runtime/social_popart_training.py) |
| Pop-Art 终态审计 | [audit_shared_social_popart.py](../scripts/audit_shared_social_popart.py) |
| 严格 Pop-Art 权重读取 | [social_popart_inference.py](../CrowdSim/humanoid/runtime/social_popart_inference.py) |
| 跟踪、风格和四人闭环 | [独立评价协议](SHARED_SOCIAL_EVALUATION_PROTOCOL_ZH.md) |

既有封存源码、默认权重、真实数据和 SDK 安装目录保持。新格式需要显式读取，legacy 读取不做隐式转换。先做实际短检查和 source/Adam/RMS 审计，再决定长训练；继续用完整动作和实际物理结果判断改进。
