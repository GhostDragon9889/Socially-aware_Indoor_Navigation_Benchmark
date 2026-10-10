# 行人 AMP + Diffusion：研究、修正与真实训练

本页保留 2026-10-01 阶段的实验记录；最新 AMASS v7、长训与原生场景进展见 [当前动作风格报告](AMP_STYLE_PROGRESS_20261002_ZH.md)。

更新：2026-10-02（文件及分阶段证据目录保留20261001命名）。当前是依据公开方法进行的独立实现及训练，尚未获得作者定制模型、2,848条动作ID和八场景协议。执行完成、局部误差下降、稳定完成导航是不同验收条件；本文以固定checkpoint和完整物理记录为依据。

## 研究依据与实现选择

| 一手资料 | 与本项目的关系 | 已落实或保留的边界 |
| --- | --- | --- |
| [NavIsaacLab，IV-A/B/C](https://arxiv.org/html/2606.26265v1) | 五状态轨迹扩散、AMP关节控制及评估 | 保留`[x,y,u,v,theta]`、x0 MSE、U-Net/map CNN、五秒局部轨迹及论文奖励系数；默认判别器使用历史状态和动作 |
| [TRACE/PACER论文，3.3及附录](https://arxiv.org/html/2304.01893v1)、[TRACE源码](https://github.com/nv-tlabs/trace)、[PACER源码](https://github.com/nv-tlabs/pacer) | 高层轨迹与低层物理控制分别训练，干净预测端引导 | 新增完整denoiser输入梯度及明确的独立低层轨迹课程；区分公开代码和论文方差因子；程序课程用于训练，部署仍由Diffusion生成轨迹 |
| [AMP原论文与作者项目](https://xbpeng.github.io/projects/AMP/) | 学习动作风格的物理控制 | 保留PPO、真实关节PD动作、专家/策略判别器与梯度惩罚；经典state-only判别器是单独消融配置，与Nav式(2)区分 |
| [Diffuser](https://arxiv.org/abs/2205.09991) | 对生成轨迹施加任务引导 | 引导目标在物理坐标中定义；分批/采样规模不隐式稀释同一行人的目标权重 |
| [PHC](https://arxiv.org/abs/2305.06456) | 大规模物理动作跟踪与失败恢复的相关研究 | 说明长期控制与复位需要独立验证；本项目没有用PHC替换AMP，也没有获得其恢复能力 |
| [PhysDiff](https://arxiv.org/abs/2212.02500) | 生成分布与物理可行性之间的差距 | 轨迹损失下降不能证明闭环物理效果；可选参考限速不是PhysDiff的物理投影，也不保证避碰 |
| [Isaac Lab Contact Sensor文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/concepts/sensors/contact_sensor.html) | 接触测量的物理语义 | 记录法向力、精确目标filters及120Hz子步峰值，排除地面/自身接触；没有将摩擦力或几何距离冒充已测身体接触 |
| [PyTorch 2.11可重复性文档](https://docs.pytorch.org/docs/2.11/notes/randomness.html) | 固定seed与确定性算法的区别 | 评估显式启用严格算法及cuBLAS合同，记录实际数值设置；Torch设置不提供PhysX/Warp或跨版本逐位保证 |

`r_d=1-exp(min(0,u)/1.8)`的印刷公式与“防止倒退”文字不一致。默认保留印刷公式；另有明确命名的方向奖励变体。AMASS没有人的控制命令，动作条件D的专家动作使用下一帧关节姿态的归一化PD目标代理；这项假设不能写成已恢复的真实专家动作。上述尚未定义的细节都记录在profile和checkpoint provenance中。

## 本轮代码改进

- `diffusion/paper.py`：新增`trace_clean`、`trace_clean_variance`；保留旧默认的数值重放。完整反向梯度经过denoiser，只有输入求梯度，模型权重不在推理时更新。
- 同场景联合行人轨迹引导、实际cuboid/domain的signed-distance引导、显式速度/加速度软代价。均为可选参数；没有将软代价写成可行性保证。
- `training/trajectories.py`：独立PACER式训练命令；平滑模式每次独立采样0–3m/s目标，限制速度增量、heading变化和向量加速度。保留公开knot递推对照分支及随机数消耗测试。完整私有CPU RNG进入续训checkpoint。
- `training/reference.py`：可选因果执行参考。普通重规划从上一份已执行参考的实际时间位置/速度继续，生成30Hz稠密参考；保留原10Hz扩散输出，单独记录两者偏差。限速/限加速度约束只作用于参考，不是物理身体保证。
- `application/`：将无SDK的CLI、单一`TrainingConfig`及资源预检集中管理；错误配置、LFS指针、错误尺寸/路径、split身份泄漏及numeric boolean在Kit启动前报错。
- `amp_world.py`：显式有限地面尺寸与出生区余量，精确体质量/PD合同，30Hz控制、120Hz PhysX；真实同步子步功率进入奖励。
- `contact_monitor.py`：每人24身体批量native view；实际V4/V5逐身体/批量测量完全一致。记录全部四个物理子步，self/floor不作为行人碰撞。
- `reconstruction_scene.py`及runner：每个真实控制帧检查到达；规划查询不增加计数。物理复位重新开始路线尝试，复位位置不产生顶点进展。
- 完整optimizer/RNG/normalizer续训合同、旧checkpoint严格加载、实际接触评估、pose cluster/MMD和阶段跌倒记录。
- 显式`--deterministic-evaluation`：仅评估使用，SDK启动前校验cuBLAS环境，启用严格Torch算法并记录实际TF32/数值设置。截图前后克隆物理buffer、检查步数与RNG；目标Sim6缺失的原生时钟接口记为不可观测，不伪造时间或修改SDK。[实现合同](amp_improvement_20261001/deterministic_evaluation_change.md)。

结构与资源位置见[当前结构索引](STRUCTURE_AND_EVALUATION_CURRENT_ZH.md)。详细证据：[干净预测引导](diffusion_amp_20261001/clean_guidance_review.md)、[独立低层课程](amp_improvement_20261001/independent_curriculum.md)、[路线生命周期](diffusion_amp_20261001/route_lifecycle_review.md)、[接触评估](amp_improvement_20261001/contact_evaluation.md)。

## 已完成的训练与物理结果

### 1. Diffusion继续训练到5,000步

`output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt`

SHA256：`63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`。

从真实ORCA社会轨迹模型step1000恢复optimizer/RNG，batch8再训练4000步。训练、验证和测试按场景及episode分割。验证x0 loss由0.09420降至0.08152，测试由0.15961降至0.09742；同32个无引导采样的ADE/FDE反而变差，因此使用闭环评估选择部署方案。这个部署模型使用本地生成的ORCA社会轨迹补充训练，不是作者2,848条AMASS轨迹的精确训练数据复现；原AMASS轨迹模型和split清单另行保留。

### 2. 同一AMP控制器的真实四人200秒对照

| 指标 | 原corrected endpoint1000 | Diff5000/clean+peer+SDF |
| --- | ---: | ---: |
| 终止跌倒 | 16 | 11 |
| 平均局部XY误差 | 0.215m | 0.080m |
| 累计到达路线顶点 | 0 | 8 |
| Pose Gaussian MMD² | 0.10935 | 0.05582 |
| K50/K100覆盖 | 0.84/0.67 | 0.82/0.68 |

新run记录static接触311/24000 actor-frames；peer接触588/36000 unordered-pair-frames。旧run没有真实接触数组，无法对比接触改善。新每人跌倒为`[2,5,0,4]`，到达顶点`[3,2,1,2]`；仍有低速停滞、身体接触和个别人退化。模型、引导及执行参数同时改变，不能单因归因，也不能称完整路线或论文效果已实现。[完整对照、逐段曲线与参数差异](amp_improvement_20261001/clean_5000_physical_comparison.md)。

### 3. 独立低层AMP训练

四步短跑和GPU续训已完成，共8,192条真实transitions。私有CPU RNG在CUDA加载checkpoint后明确回到CPU再恢复；不是仅加载actor参数。源文件冻结存于[训练源码清单](amp_improvement_20261001/independent_curriculum_training_source.json)及相邻tar.gz。

完整1,000次PPO更新已完成，共2,048,000条物理transitions，耗时1,150.03秒；使用`amp_independent_curriculum.yaml`、64隔离actor、每轮32个30Hz物理控制步、128m地面/48m出生余量/10秒episode。训练未调用Diffusion；绑定的固定Diff5000供独立部署，provenance明确`diffusion_used_for_training=false`。

固定模型：`output/diffusion_amp/amp_independent_curriculum_v1/step_1000.pt`，SHA256 `e0d6ed4a9acb39c581bed92fab722a426bf23c34a3949bc48712cc13bf04c333`。121个checkpoint Tensor叶全部有限，模型/两个Adam/四份RMS及私有RNG恢复校验通过。训练87次跌倒、6,745次time limit；末段跌倒率下降，但D仍明显区分专家和策略，不能声明动作风格收敛。[完整训练审计与曲线](amp_improvement_20261001/independent_amp_training_review.md)。

组合新AMP与原始Diff5000轨迹的200秒评估为3次跌倒、`[1,3,4,4]`个实际顶点，完整循环`[0,0,1,1]`；所有12次到达通过实际位置、episode与索引校验。一个循环完成者随后仍跌倒，因此不能将两个循环写成两名行人全程无跌倒。[真实路线验收](amp_improvement_20261001/raw_route_completion_verification.md)。静态接触及姿态多样性退化；后续部署实验单独保存，默认算法没有自动替换。

### 4. 当前验证通过的单候选定时路线与连续参考

固定以上AMP1000和Diff5000，在同一新建共享障碍fixture、seed7、四人各200秒中：**零跌倒、26次真实顶点到达、5个同episode完整路线循环**。每人到达`[8,6,6,6]`，循环`[2,1,1,1]`，26个点均经过物理位置、索引和reset核查；最低pelvis为0.6331m。没有将查询成功或复位位置计为到达。

| 指标 | 原始endpoint部署 | 定时路线＋连续参考部署 |
| --- | ---: | ---: |
| 200秒跌倒事件 | 3 | 0 |
| 完整四顶点循环 | 2 | 5 |
| 排除reset的物理XY路程 | 355.60m | 529.60m |
| Static接触actor-frame比例，strict >1N | 7.00% | 3.85% |
| Peer接触unordered-pair-frames | 1 | 0 |
| Pose Gaussian MMD² | 0.09943 | 0.09215 |
| K50/K100覆盖 | 0.54/0.41 | 0.58/0.47 |

新部署同时改变定时路点、A*合法起点恢复、碰撞引导半径和因果参考，不能将结果单独归因于某一组件。Static仍有90个连续actor接触事件；姿态覆盖和风格还未达到论文验收。这个本地导航条件完成验证，不是八场景或任意时长稳定性证明。[完整逐帧对照、参数及源码hash](amp_improvement_20261001/continuous_timed_physical_comparison.md)。

1704份保存参考计划全部通过1.5m/s速度和2m/s²向量加速度合同（允许float32舍入）；原始扩散目标与参考偏差mean/p95为0.126/0.440m。物理root速度仍可超过1.5m/s，实测区间最大2.20m/s。root到**执行参考**平均误差0.202m，到**原始模型目标**0.116m；两者不能混用。详见[参考适配器合同与测试](reference_adapter_20261001/report.md)。

此前未启用严格Torch确定性算法的单候选和双候选两次30秒**重新运行的渲染**短跑均出现1次跌倒；分别到达2个/1个顶点，均没有完整循环。它们与上面的200秒无渲染运行独立保存，不能据此把问题单因归给候选数，也不能将无渲染零跌倒结论直接扩展到渲染模式。单候选首个扩散计划与200秒记录只存在小的数值差异，差异随后扩大；目前没有证明具体根因。[首帧与后续差异诊断](amp_improvement_20261001/render_s1_divergence.md)。原结果和15fps视频保留在`output/diffusion_amp/independent_amp_diffusion_render_s{1,2}_v1/`。多样本联合选择及记录接口通过核查。[渲染产物核验](amp_improvement_20261001/render_capture_verification.json)、[实际采样接口](amp_improvement_20261001/diffusion_samples_interface_review.md)。后续同源码确定性对照见第6节。

### 5. 已记录轨迹的完整视觉回放

[200秒回放视频](../output/diffusion_amp/recorded_pose_replay_200s_v1/pedestrian_motion.mp4)已完成：1280×960、15fps、3000帧，ffprobe实解码200.000秒。每帧永久标记`Recorded physical state replay`及原始时间；使用原200秒物理记录的真实root姿态和common69DOF进行SMPL皮肤重建。没有重新生成轨迹、加载策略或积分物理。

3000份源索引、episode/reset标记、固定时间线、源NPZ/result及实际源码/rig/neutral模型/UV/纹理hash均通过核验；首、中、末画面已检查。[回放证据](amp_improvement_20261001/recorded_replay_actual_verification.md)、[独立工具合同](amp_improvement_20261001/recorded_pose_replay.md)。该视频呈现已有记录，闭环导航结论仍来自原始物理run。早期回放取帧超时记录保留，当前有界取帧路径已实际通过。

### 6. 相同源码的确定性渲染/无渲染对照

启用`--deterministic-evaluation`，用同一固定模型、单候选参数及seed7分别重新运行四人30秒：两次均零跌倒、到达`[0,1,0,1]`、没有完整循环。两份运行保存的全部物理状态和参考计划数组逐字节一致；450次截图前后，六项已观测公共状态buffer、控制步数以及CPU/当前CUDA RNG均未变化。目标Sim6没有原生时钟查询方法，该项450次都明确保留unknown，未用控制时钟替代。

这两次均有static接触315/3600 actor-frames（8.75%），peer接触27/5400 unordered-pair-frames（0.50%）。这是固定环境/源码下的30秒一致性证据，不证明200秒渲染、高密度、跨平台或PhysX/Warp普遍逐位确定性，也不能证明早期跌倒的唯一原因。[逐数组、截图与来源核验](amp_improvement_20261001/deterministic_render_pair_verification.md)。

## 运行命令

从源码根目录启用已经验证的环境：

```bash
source ../IsaacLab-Protomotions/.venv/bin/activate
```

预检和独立低层训练使用相同参数；预检将脚本替换为`python -m CrowdSim.humanoid.application.preflight`。

```bash
python scripts/train_paper_amp.py \
  --config configs/diffusion_amp/amp_independent_curriculum.yaml \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --trajectory-source pacer_curriculum --curriculum-mode smooth \
  --num-envs 64 --updates 1000 --rollout-steps 32 --episode-steps 300 \
  --terrain-extent-m 128 --spawn-margin-m 48 \
  --checkpoint-every 100 --synchronized-replanning --physics-only \
  --output output/diffusion_amp/amp_independent_NEW_RUN
```

下面是当前完成200秒验证的**单候选**部署参数。它包含明确的项目适配，不能称作者原配置。使用固定controller文件并指定独立新输出目录：

```bash
python scripts/train_paper_amp.py \
  --resume output/diffusion_amp/amp_independent_curriculum_v1/step_1000.pt \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --allow-evaluation-planner-change \
  --diffusion-guidance-mode trace_clean --guidance-weight 20 \
  --agent-collision-weight 5 --agent-collision-radius 0.6 \
  --scene-collision-weight 10 --scene-collision-radius 0.8 \
  --reference-adapter causal_accel \
  --reference-max-speed-mps 1.5 --reference-max-accel-mps2 2 \
  --reference-initial-speed-policy brake --diffusion-samples 1 \
  --scene-config configs/diffusion_amp/shared_obstacle_timed_validation.yaml \
  --route-guidance timed_path --num-envs 4 --evaluate-steps 6000 --episode-steps 6001 \
  --pairwise-contacts --synchronized-replanning --planner-batch-size 8 --physics-only \
  --output output/diffusion_amp/independent_amp_shared_NEW_RUN
```

需要重新运行控制器并渲染时用`--capture --capture-every-steps 2`替换`--physics-only`，将`--evaluate-steps`改为900可录制30秒、15fps视频。加入`--deterministic-evaluation`采用第6节实际核验的数值模式；相同标记也用于配对无渲染运行。截图诊断未观察到额外控制步或公共状态变化，缺失原生时钟保留unknown；并不保证跨环境的逐位一致。此前普通模式渲染短跑存在跌倒，严格模式目前只完成30秒对照。短视频不能替代200秒物理记录。共享物理状态、接触、路线和pose指标需要一起检查。

完整视觉回放的可复跑入口，使用新的输出路径：

```bash
python scripts/replay_paper_pedestrians.py \
  --states output/diffusion_amp/shared_independent_amp_continuous_timed_v1/physical_states.npz \
  --source-result output/diffusion_amp/shared_independent_amp_continuous_timed_v1/result.json \
  --steps 6000 --capture-every-steps 2 \
  --camera-eye 28 -6 24 --camera-target 12 12 1 \
  --output output/diffusion_amp/recorded_pose_replay_NEW_RUN
```

## 尚未通过的验收

1. 当前单seed有界障碍场景已完成200秒无跌倒路线验证；其他seed、高密度、多场景与长期稳定性仍未验证，static接触及动作风格仍需训练改进。
2. 连续参考合同和无渲染闭环结果已验证，但尚未证明重规划跳变是跌倒原因。它会偏离raw目标，且不保证身体速度、避碰或动作风格；历史普通模式单/双候选渲染短跑均有跌倒，严格模式同源码对照只完成30秒。200秒渲染、多seed及跨运行数值敏感性仍需验证。
3. 作者2,848动作清单、八场景/3–15人完整协议、真实专家动作定义、定制checkpoint和论文指标分母未取得；现有结果不能与Table II作数值等价声明。
4. 当前四张大场景地图是CPU几何投影原型，仍需真实collider检验；当前闭环结果来自明确的新建cuboid fixture，不是八个作者场景。
5. 长期自然失败恢复、不同体型/衣着及高密度接触还需要独立验证。
6. 新方法的完整训练/部署入口是`scripts/train_paper_amp.py`。原CrowdSim机器人导航入口仍使用已有SFM/MaskedMimic配置；新的AMP/Diffusion组合尚未接入该入口的多场景机器人交互评测。

## 最终代码检查

核心依赖、扩散引导、AMP奖励与功率、课程/参考、路线与控制时钟、资源预检、真实接触、评估CLI、记录回放和确定性诊断的20份测试文件合并运行：**463 passed**，CPU隔离，无Kit实例；44条警告来自TorchScript弃用及docstring转义。[日志](amp_improvement_20261001/final_core_tests.log)、[JUnit](amp_improvement_20261001/final_core_tests.xml)。本轮核心修改Ruff通过：[记录](amp_improvement_20261001/final_ruff.log)。

当前业务代码、配置及测试的75文件冻结清单见[源码清单](amp_improvement_20261001/final_runtime_source.json)。源码发行包必需路径、代码及指南快照通过独立检查；权重与授权SMPL资源保持外部，[最终打包快照报告](amp_improvement_20261001/packaging_source_snapshot_v4.md)。v1/v2/v3分阶段证据仍保留。

这些测试证明所覆盖的接口与数值合同；长期物理效果依据前述实际运行，不由单元测试替代。
