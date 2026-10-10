# 本地 AMASS 微调 AMP 与 Diffusion 人群场景测试

## 2026-10-08：真实 Warehouse 新一轮 Diffusion+AMP 优化

已完成固定2848 AMASS / 512次微调 / 2,097,152实际物理样本；四人原始Warehouse完整200秒，零跌倒、零>1N静态和行人接触，54个路线节点，实际退出0。30秒30fps室内+跟随视频已完成。姿态稳定性和部分姿态指标改善；按速度条件的动作MMD与覆盖仍退步。

详见 [Warehouse 当前结果](WAREHOUSE_DIFFUSION_AMP_ZH.md) 与 `warehouse_diffusion_amp_evidence_index.json`。本次一场景工程复现不等于作者完整八场景协议。此前AMASS anchor512和旧Social AMP结果保留为历史对照。


更新：2026-10-08。真实微调及首个 200 秒物理测试完成；真实训练、两个物理场景和三个正式视频均已完成，普通父进程实际退出0，来源摘要与独立核验通过。

## 已完成的真实训练

| 项目 | 实际结果 |
|---|---|
| 初始化 | 作者公开 PACER no_shape 演员、价值网络、任务编码器和完整 AMP 判别器 |
| 数据 | 固定 v7 真实 AMASS：训练 2848、验证 22、测试 34；训练只用 train |
| 预算 | 128 个隔离 PhysX 环境，512 次 PPO+AMP 更新，2,097,152 条实际物理转移 |
| 频率 | 30Hz 动作、120Hz 物理；每次采集 32 个真实控制步 |
| 专家覆盖 | 所有 2848 个训练 motion ID 均被采样；每次更新采样 4096 组真实 10 帧历史 |
| 跌倒与重置 | 6 次真实训练跌倒；总重置 20,567 次，包含真实片段末尾和 10 秒时间上限 |
| 退出 | 普通父进程 Popen.wait 得到实际退出 0；52 项训练源码运行前后不变，SDK正常释放 |
| 推理权重 SHA256 | `c07805d1ae9ab759fe63828a1379d15a36a127d68a52526d4b3e87007c3dcd73` |
| 独立 CPU 审核 | 2848 个 ID、冻结统计及演员/价值/判别器完整导出推理一致性均通过 |

这是已执行的有界微调预算，尚无训练收敛认证。16 更新试训另外保留，不混入正式预算；首次正式启动的缓存校验失败也保留。

## 方法及与论文的对应

依据 [NavIsaacLab IV-B/IV-C](https://arxiv.org/html/2606.26265v1#S4.SS2)、
[Trace and Pace](https://arxiv.org/abs/2304.01893) 和[作者 PACER 代码](https://github.com/nv-tlabs/pacer)。
采用真实轨迹扩散规划、经 AMP 训练的全身 PD 控制器和 PhysX，部署策略根据当前真实状态闭环控制关节。

本轮保留公开 PACER 的 357 维自身观测、1044 维任务观测、69 维动作和 10×195 状态判别器。
原策略作为冻结教师，以同方差 Gaussian KL 约束偏移，系数 1，只计算实际生效的 57 个动作坐标。
四组观测/任务/价值/AMP 归一化统计冻结。策略学习率 1e-6，判别器 2e-6，PPO 每轮 2 epoch，
小批量 1024、判别器批量 512，每轮独立更新判别器两次，保留策略历史回放。
教师不参与部署；导出的策略本身完成控制。

训练轨迹取同一真实动作片段的根部 XY，在每次重置时只平移一次；按 0、0.5、…、4.5 秒取十个目标，
与公开 PACER 原时钟对齐。初始动作姿态仅用于重置，AMP 专家只用于对抗训练。
将来三维姿态、动作 ID、参考关节目标不进入演员观测。片段外的 XY 明确按尾端速度外推作为命令；
正式训练共 20,971,520 个查询目标，其中 11,039,791 个超出真实片段，约 52.64%。
外推不作为 AMASS 真值；物理 episode 在真实片段末尾终止/截断，并用重置前实际最终状态处理 bootstrap。
地面、关节力、速度和姿态均由实际物理产生。

这是明确的本地工程适配：公开 TRACE 使用 action-space unicycle，公开 PACER 为状态判别器；
Nav 文中历史加动作判别器及作者私有五状态权重没有被冒称复现。
本地 2848 是可追溯的真实固定清单，作者完整 2848 ID 清单未取得。训练方向奖励采用已声明的 backwards_penalty。

## 产物与证据

- 权重：`data/pretrained_models/local_amass_finetune/anchor_20261008/Humanoid_AMASS.pth`。
- 训练入口：`scripts/finetune_pacer_amass.py`（物理训练）和`finetune_pacer_amass_verified.py`（普通父进程收据）；教师与导出：`CrowdSim/humanoid/training/pacer_finetune.py`。
- 完整优化器/RNG/runner 保存包：`amass_anchor_full512_v2/last.pt`；不包含 PhysX 求解器缓存。
- `amass_anchor_full512_v2_execution.json`：真实训练终态、模型与日志摘要。
- `amass_anchor_full512_v2_sources/index.json`：完成训练的 52 项源码冻结副本。
- `amass_anchor_full512_cpu_audit.json`：独立推理一致性与来源核验。
- `amass_finetune_actual_banks.json`：三个动作库实际 SHA256；全部与清单一致。
- `amass_finetune_training.png`：全部 512 次真实更新曲线。

以上证据目录：`/home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/evidence/`。

## 复现运行

在 `/project/EXP/SangforAndSaut/ProtoMotions` 中，用当前 Isaac SDK 环境执行：

```bash
PY=/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/finetune_pacer_amass_verified.py \
  --output output/diffusion_amp/new_amass_finetune \
  --updates 512 --num-envs 128
```

输出路径必须全新。固定环境为 IsaacSim 6.0.0.1 / IsaacLab 3.0 beta2 开发版本 / Python 3.12.13 / Torch 2.11.0+cu128。
保持已核验的 SDK 兼容层和原生资源释放实现，禁用生成字节码。实际训练完整性由父进程验证脚本记录，
单独训练进程打印 passed 不作为正常退出证据。固定种子不保证 GPU/PhysX 轨迹逐位一致。

场景按实际微调权重运行；已验收配置`trace_pace_amass_finetuned.json`的全部数值运行字段与实测候选完全一致，只增加验收来源；原候选配置保留原字节。

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/run_trace_pace_verified.py \
  --output output/diffusion_amp/new_amass_crowd -- \
  --config configs/diffusion_amp/trace_pace_amass_finetuned.json \
  --scene configs/diffusion_amp/shared_obstacle_trace_pace_reciprocal.yaml \
  --steps 6000 --measure-contacts

"$PY" scripts/evaluate_trace_pace_reliability.py \
  --run output/diffusion_amp/new_amass_crowd \
  --execution output/diffusion_amp/new_amass_crowd_execution.json \
  --output output/diffusion_amp/new_amass_crowd_reliability.json
```

风格、步态分别用 `evaluate_trace_pace_style.py` 和 `evaluate_trace_pace_gait.py` 的同一参数协议。
所有输出必须新建。后续重复运行采用同一来源，GPU物理可能出现差异，不能只检验日志中的 passed。

## 已验收的 200 秒四人相向场景

`amass_finetuned_reciprocal_200s`：实际普通进程退出 0，输入和 48 项运行源码不变，
6000/6000 控制步、24000 个物理子步，完整 SDK 清理；初始化后没有重置或瞬移。
独立 reliability/style/gait/encounters 均实际退出 0；10 项本地质量门槛全通过。

| 指标 | 微调后实际结果 |
|---|---|
| 跌倒 / 行人接触 / 地图接触 | 0 / 0 / 0；全部24身体、120Hz法向力>1N阈值 |
| 路线节点 | 每人 12 / 13 / 11 / 11，共 47 |
| 真实路程 | 每人 149.51 / 149.77 / 132.37 / 137.61 m |
| 最小根部人际间距 / 地图身体包络 | 1.473 / 1.808 m |
| 最长连续低速 | 最多 3.9 秒 |
| 脚底法向支撑 | 至少一脚支撑99.879%–99.967%；平均载荷约621N/人 |
| 支撑脚速度代理 | 按实际法向支撑筛选的脚身体原点速度0.06442m/s；不是接触斑摩擦滑移 |
| 左右单脚支撑交替起始 | 每人中位1.765–1.846Hz |
| 最高真实根部速度 | 1.526m/s；规划上限1m/s不等于实际物理速度上限 |
| 扩散与制动 | 428批规划中82批显式解析制动，其余真实DDPM |
| 相向通行诊断 | 25个接近区间、23个纵向通过代理；不是Nav总体成功率 |
| 完整动作风格 | 23996合法区间；姿势MMD²0.03471，速度分桶MMD²0.07334，K50/K100覆盖0.44/0.36 |

公开权重基线姿势MMD²0.03180、速度分桶0.07047、覆盖0.52/0.39。
这轮微调后风格分数略差。训练已实际完成且此场景可靠性通过，**不声明风格改善或超过论文**。
两次运行不是逐位相同的配对轨迹；这些是同配置各自完整运行的比较。

## 已验收的 60 秒同向绕障场景

`amass_finetuned_circulation_60s_v2`：普通子进程实际退出 0，48 项运行源码与输入不变，
1800/1800 控制步、7200 物理子步，无初始化后重置；10 项独立质量门槛全通过。
路线节点每人 2/1/2/1，共 6；真实路程 27.09/21.14/26.02/23.18m。
零跌倒、零行人/静态障碍接触；最小根部人际间距 6.483m，最小身体地图包络 2.115m。
最长连续低速最多 0.5 秒；四人至少一脚支撑 99.861%–100%，平均载荷与体重比约1。
支撑脚身体原点速度代理 0.09157m/s，左右单脚支撑交替中位2.07–2.31Hz。
120 批规划均使用扩散候选，解析制动为 0。该同向场景没有相向会车事件。
完整 7196 合法风格区间：姿势MMD²0.04592，速度分桶MMD²0.08272，覆盖K50/K100为0.42/0.29。

首次同向60秒运行虽然保存了完整物理数组并打印passed，进程实际退出为SIGSEGV(-11)，
按失败保留；没有用本次正常复跑覆盖原失败。正式复跑串行执行，没有同时进行RTX录制。

## 录制与查看

视频由同一次通过审核的真实物理状态记录驱动，不重新积分或播放原始AMASS。
源码保留30Hz真实身体/69关节/XYZW根旋转，视频每2个控制步取1帧，为960×720/15fps。
画面永久标注物理源时间，并保存逐帧时间、episode、reset与RGB数据统计；没有跨重置插值。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "$PY" scripts/record_trace_pace_verified.py \
  --run output/diffusion_amp/new_amass_crowd \
  --quality output/diffusion_amp/new_amass_crowd_reliability.json \
  --output output/diffusion_amp/new_amass_crowd_video --steps 1800
```

默认录前60秒；对于6000控制步的已存物理记录，`--steps 6000`可完整录200秒。
`--start-step`和`--steps`只选择已存实际状态区间；不能超过源记录。
录制入口拒绝未经真实退出0/输入摘要绑定/局部物理质量门槛审核的运行。
仅回放使用`trace_pace_recording.kit`最小USD/RTX配置，无新IsaacLab物理环境；
PhysX场景测试仍由原IsaacLab3执行。所有录制必须普通实际退出0，完整释放RGB、Hydra、USD和App。
不会用跳过清理、快速强退或改写退出码来验收视频。

早期录制的USD关闭挂起、退出SIGSEGV和基础体验初始化中止均保留。
最小录制体验中显式提供SyntheticData要求的三帧历史，禁用在线扩展注册，并在扩展卸载前释放Hydra。
这是当前SDK下录制的所有权适配，不构成外部SDK所有退出崩溃已修复的声明。

正式视频（证据目录下；只有录制实际退出0才计入）：

| 视频 | 源物理时间 | 文件 |
|---|---|---|
| 四人相向总览 | 0–60s，900帧 | `amass_reciprocal_overview_video_v2/pedestrian_motion.mp4` |
| 步态近景 | 6–18s，180帧 | `amass_reciprocal_gait_video_v3/pedestrian_motion.mp4` |
| 同向绕障总览 | 0–60s，900帧 | `amass_circulation_overview_video_v2/pedestrian_motion.mp4` |

相向视频取已完成200秒物理测试的前60秒；完整200秒姿态记录保留。
总览可观察四人、实体墙和中心障碍；近景v3调整固定镜位以显示腿脚，源人物、物理时间和动作均不变。v2镜位较近，部分行人出框，单独保留。
这些视频用于查看实际生成动作，不作为独立物理评价的替代。

## 验收范围

微调权重已通过这两项局部物理质量门槛。已经核验四人相向通行200秒和同向绕障60秒，
测量全部身体的行人/地图接触、脚底法向支撑、真实路线进度和完整动作风格。
风格使用冻结的22条验证动作特征，与本轮训练原始动作来源不重叠；主体/会话独立性未证明。公开教师原AMASS训练ID未知，不能保证作者原预训练从未见过这些验证动作。测试34仅做文件完整性校验，未参与梯度或分数调优。

视频使用同一次真实物理测试保存的身体位置、根部旋转和 69 维关节姿态进行忠实 SMPL 渲染。
每帧标注源物理时间，渲染不增加物理步，也不把 AMASS 原始动画作为生成结果。
本地受控场景不是论文作者的八场景、3–15 人总体协议，不能由局部零碰撞声称全面超过论文。

## 终态索引

机器可读来源、每次真实退出、独立报告和视频ffprobe核验见`docs/amass_finetune_evidence_index.json`。
当前训练/回放的定向34项CPU测试通过（Junit：`amass_final_focused_tests.xml`），另外此前116项原训练与可靠性兼容测试通过。
完整训练和正式视频源码均保存冻结副本；生成器不修改原始成功或失败收据。
