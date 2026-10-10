# 共享 AMP 独立评价协议

目标环境为 [固定环境记录](../env/isaaclab3_sim6/profile.json)：Python 3.12、Isaac Sim 6.0.0.1、Isaac Lab 3 开发版本和 PyTorch 2.11/cu128。实际执行还核对已安装 SDK 的源码与独立原生扩展身份；版本名称本身不能替代兼容性验证。

## 数据与模型选择

- 训练保持真实 AMASS v7 的 **2,848 个片段**，清单 SHA256 为 `68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`。
- 开发评价使用完整 validation22，与训练按原始动作文件分离。Test34 不参与本轮迭代和聚类拟合。
- 本次预先选择 2,000 更新的终态权重，不按较低 MMD 或较高训练奖励挑选检查点。
- 加载前必须有训练进程实际退出 0 的记录、完整终态结果和 CPU 权重审计；模拟步数、正常关闭和动作质量分别记录。

## 与论文的对应关系

| 评价 | 论文公开方法 | 本地固定协议 | 尚未对齐的部分 |
| --- | --- | --- | --- |
| 逐帧全身跟踪 | 同相位刚体位置/旋转；每步平均位置误差超过 0.5 m 即失败 | 全部 22 条、11,982 个后物理控制步；30 Hz；不做最佳帧或旋转匹配；跌倒后继续采集，不在片段内重置 | 作者评价动作 ID、数量、人体形状和物理参数 |
| 动作风格 | 15 关键点、30 Hz、去根平移、朝向对齐、尺度归一化；K=50/100 coverage、entropy、MMD | 固定已有验证参考；复用保存的 K=50/100 中心；45D 姿势与 90D 姿势/速度分开报告，保持已有尺度、核带宽和采样上限 | 作者完整关键点映射、尺度公式、核、聚类与抽样实现 |
| 导航/物理 | 8 场景、每场 3–15 人、200 s；Diffusion 与 SFM 使用同一个 AMP | 下一阶段先做固定四人 200 s 的本地闭环对照，记录实际接触、跌倒、路线进度、误差与重置 | 作者八场景和路线/初态清单；成功/跌倒/碰撞率的精确分母与聚合定义 |

对应依据：[NavIsaacLab IV-B/IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)。论文跟踪均值 0.218 m、0.343 rad 和成功率 64.6%，以及 Table II/III 数值是对齐目标；当前本地评价人群和指标实现未完全一致，不能将它们当作已复现的判定线。尤其本地 **MMD² 不直接等于论文 MMD**。

## 可执行顺序

1. 完成共享训练，记录实际退出状态，运行 [CPU 审计](../scripts/audit_shared_social_training.py)。保存终态权重为独立只读文件。
2. 用 [跟踪入口](../scripts/evaluate_shared_social_tracking.py) 执行 N1 生命周期检查：零动作、8 控制步/32 物理子步。这一检查不评价策略性能。
3. 使用同一入口执行完整 validation22。策略只接收真实当前状态和未来参考根 XY；参考全身姿势仅用于初始重置与事后评分。
4. 使用 [风格入口](../scripts/evaluate_shared_social_style.py) 在 CPU 上运行 90D 和固定中心的 45D 评价。失败、跌倒和静止帧进入主要评价；跨片段差分与初始重置帧按原方法排除，辅助 upright 子集单独标注。
5. 用[四人闭环入口](../scripts/evaluate_shared_social_navigation.py)先运行 30 控制步，验证真实接触和关闭，再运行 6,000 控制步（200 s）。固定同一个 Diffusion 和相同场景，评价新 AMP；随后进行同 AMP 的 SFM/Diffusion 消融及 value guidance 对照。

前三项评价可通过 [串行执行器](../scripts/run_shared_social_evaluation.py) 运行：

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/run_shared_social_evaluation.py \
  --execution-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/training_execution.json \
  --output output/diffusion_amp/shared_social_terminal2000_evaluation_v1
```

上述 `training_execution.json` 需要由实际训练退出与审计结果生成；文件不存在时命令不会运行模拟。每个子进程有独立日志与实际退出码；失败会停止后续步骤并保留输出。源码检查和模拟检查通过均不能证明模型质量通过。

当前 N1 生命周期实现仅处理一个完整、无接触传感器的原 AMPWorld，复用已验证的 SDK 终态释放方法，并将资产数量合同明确改为一个向量化 Articulation。它尚不覆盖构造中途异常或四人接触/脚部视图，须分别验证。

N4 使用[独立接触生命周期](../CrowdSim/humanoid/runtime/social_navigation_lifecycle.py)，要求一个向量化 Articulation、四个同行人视图、一个脚部支撑视图、24 个 SDK 传感器；实际退出前核查这些 native owner 的 weakref 全部释放。S0 的30/6,000步与Pop-Art短检查权重的30步均已退出0；完整构造路径的关闭已验证。SDK安装目录中的代码和方法未修改。

固定正式 Diffusion 为 `step_40000.ckpt`、SHA256 `4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff`：

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_navigation.py \
  --execution-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/training_execution.json \
  --diffusion-checkpoint output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt \
  --diffusion-sha256 4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff \
  --control-steps 30 --output output/diffusion_amp/shared_social_terminal2000_navigation_smoke_v1
```

30 步成功且进程实际退出 0 后，才使用 `--control-steps 6000` 和新的输出目录运行 200 s。`--check-sources` 仅检查源码，`--preflight-only` 仅检查资源；两者均不是实际模型或模拟评价。新入口拒绝未完成、退出非零或 CPU 审计不一致的训练来源。

Pop-Art 评价无需手动转换权重。训练退出记录、CPU 审计和数值来源必须一致，入口据此显式选择新格式 reader；legacy 读取继续拒绝新格式。S1 终态生成自己的实际退出记录后，将上述 `--execution-receipt` 换成该记录并使用新输出目录；不以当前启动记录代替终态退出记录。

S1有[完整串行执行器](../scripts/run_shared_social_post_training.py)：验证预登记终态和固定Diffusion40k，
依次执行CPU加载、full22/90D/45D、四人30步、四人6,000步及CPU数组审计。
执行器记录每个子进程真实退出码，并检查指标结果/清理/源码身份；执行通过与模型质量通过分别记录。
[终态封存工具](../scripts/seal_shared_social_training.py)需要外部进程owner提供实际退出记录，
不会从日志更新数、PID消失或`result.json`推测进程退出。

完整200s结束后，用[独立数组审计](../scripts/audit_shared_social_navigation.py)重新核对所有力/episode/reset/路线/地面支撑数组。主要分母为全部24,000 actor-controls；报告无序同行人pair-controls，包含失败。脚部刚体原点速度仅为支撑代理，论文未公开的率分母和接触点滑动不由该代理推定。

## 下一轮优化的决策依据

先报告 S0 的完整独立结果，再分别进行单变量实验：同行人 encounter 课程、固定 2848 内的窗口采样分层、critic/value 训练稳定性和真实足部接触测量。训练奖励、critic 自举回报或支持样本数量变化不能单独证明效果提升。若回报与跌倒趋势相反，保留原结果并核查价值归一化、实际 critic 更新次数及终止/截断 bootstrap；当前这些仅是待检验原因。

[终态稳定性诊断](../scripts/diagnose_shared_social_training.py)仅在实际训练退出和 CPU 审计之后读取最终保存 rollout。它复用原 GAE，报告真实 reward/values/next_values 分布、奖励与 critic 自举贡献和逐 100 更新窗口；不构造 NN 或 SDK，也不把 reason flags 当作独立评估跌倒率。

方法资料：[AMP 原论文](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)、[TRACE/PACER 作者项目](https://research.nvidia.com/labs/toronto-ai/trace-pace/)、[PACER](https://github.com/nv-tlabs/pacer)、[TRACE](https://github.com/nv-tlabs/trace)。环境生命周期以本地固定源码为执行依据；[Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html)与 [Kit 扩展路径文档](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/extensions_advanced.html)仅提供公开 API 背景。
