# Warehouse 行人模拟：实际微调、场景测试与视频

## 当前交付

- 固定2848条真实 AMASS 训练片段，512次更新、2,097,152个实际物理训练样本，2次训练跌倒；完整训练和导出审计退出0。
- 原始 Warehouse USD，四名行人，120Hz物理、30Hz关节控制和真实状态反馈；完整200秒 / 6000控制步，无初始化后的重置。
- 行人分别到达13、13、14、14个路线节点；零跌倒，745个原始静态碰撞体和全部24身体部位测量中，零超过1N的静态/行人接触。十项本地可靠性检查通过，完整SDK清理和父进程实际wait退出0。
- 900帧 / 30秒 / 30fps / 960×720 H264视频；前15秒室内全景，后15秒跟随行人，连续实际源控制步0..899。原始几何完整保留，未隐藏屋顶。视频为已测量状态的忠实渲染，渲染过程不产生新物理。

[播放 Warehouse 视频](/home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/warehouse_optimization/warehouse_diffusion_amp_30s_v2.mp4)

[实际证据索引](/project/EXP/SangforAndSaut/ProtoMotions/docs/warehouse_diffusion_amp_evidence_index.json)

## 对照结果和取舍

下表是相同场景/路线/种子、每组前60秒的比较。闭环实际路线不同；仅是本地描述性结果，不代表统计显著性或论文分数。

| 指标 | 旧AMASS anchor512 | 新姿态/速度AMP，原引导 | 新AMP+速度引导 |
|---|---:|---:|---:|
| 平均速度 m/s | 0.95714 | 0.94388 | 1.11192 |
| 姿态 MMD² ↓ | 0.03052 | 0.03111 | 0.02642 |
| 按速度条件的动作 MMD² ↓ | 0.07047 | 0.08178 | 0.07994 |
| K50动作覆盖 ↑ | 0.46000 | 0.36000 | 0.36000 |
| 接触时足刚体原点速度代理 m/s | 0.06297 | 0.06144 | 0.07039 |

新AMP相对旧AMP的同引导60秒对照中，120Hz平均后退比例从1.569%降至0.878%，最大根倾角从22.79°降至12.99°，XY加速度p95从3.849降至3.685m/s²。速度引导组合的60秒姿态MMD²改善，但速度条件动作MMD和覆盖率变差，不能声称动作风格全面变好。

完整200秒速度组合：平均1.086m/s，中位1.123m/s，p95为1.529m/s，峰值1.937m/s。规划上限1.2m/s是采样软约束与规划验收条件，不是物理速度硬保证；当前仍有峰速和风格改进空间。32次规划包含显式解析制动，需与纯扩散采样区分。

## 可复现入口

工作目录 `/project/EXP/SangforAndSaut/ProtoMotions`；使用 `/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python`。实际环境为IsaacSim6和现有IsaacLab3开发版本，版本/提交见各结果runtime字段。对下列命令选全新输出路径；物理测试与录像顺序执行。

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
PYTHON=/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python

# 可重新训练；模型选择没有使用34条测试动作。
$PYTHON scripts/finetune_pacer_amass_verified.py --output /tmp/amass_locomotion_new --updates 512 --num-envs 128 --optimization-config configs/diffusion_amp/locomotion_finetune_v1.json

# 当前已验证的模型和真实Warehouse配置。
$PYTHON scripts/run_trace_pace_verified.py --output /tmp/warehouse_new -- --config configs/diffusion_amp/trace_pace_warehouse_locomotion_v1.json --scene configs/diffusion_amp/warehouse_trace_pace_speed_v1.yaml --steps 6000 --measure-contacts --seed 7
$PYTHON scripts/evaluate_trace_pace_reliability.py --run /tmp/warehouse_new --execution /tmp/warehouse_new_execution.json --output /tmp/warehouse_new_quality.json
$PYTHON scripts/evaluate_trace_pace_style.py --run /tmp/warehouse_new --execution /tmp/warehouse_new_execution.json --output /tmp/warehouse_new_style.json
$PYTHON scripts/evaluate_trace_pace_gait.py --run /tmp/warehouse_new --execution /tmp/warehouse_new_execution.json --output /tmp/warehouse_new_gait.json

# 室内镜头；原始屋顶和墙体完整保留。
$PYTHON scripts/record_trace_pace_verified.py --run /tmp/warehouse_new --quality /tmp/warehouse_new_quality.json --output /tmp/warehouse_video_new --steps 450 --capture-every-steps 1 --native-scene-visuals --camera-eye 8 -10 7 --camera-target 0 3 0.9
# 近景可增加 --start-step 450 --follow-actor 0 --follow-offset 0 3 3
```

## 代码结构

- `CrowdSim/humanoid/training/locomotion_objective.py`：当前真实训练片段姿态、速度和有效动作变化奖励。
- `CrowdSim/humanoid/diffusion/guidance.py`：可选转弯减速和起步速度平滑；旧配置默认零改动。
- `CrowdSim/humanoid/runtime/trace_pace_native_lifecycle.py`：原始Warehouse查询、接触和相机之外的物理对象生命周期；复用已核对的SDK释放契约。
- `scripts/run_trace_pace.py`：真实地图、原始USD、AMP高度观测与扩散栅格使用同一世界。
- `scripts/evaluate_trace_pace_reliability.py`：从完整状态独立重建地图距离及路线到达，核对原生120Hz接触记录。
- `scripts/replay_paper_pedestrians.py`：静态USD相机坐标读回、实际根坐标跟随、源时刻标注及完整清理。
- `*_verified.py`：冻结源码/输入，记录实际子进程退出和哈希。

## 未完成的论文范围

这是一场景、四行人、单种子的公开方法工程复现，不是作者八场景、3–15行人全套协议。作者定制配置/权重及准确2848动作ID仍不可获得；采用公共TRACE/PACER表示并进行本地微调。尚不能声称达到或超越论文，也不能将本地MMD²、覆盖或接触阈值直接与论文表格比较。风格覆盖、速度条件分布、峰速约束以及更密集对向人群和多种子验证仍需改进。

# Warehouse Diffusion + AMP 优化实验

## 方法与边界

- 对照：此前完成的 AMASS anchor512 权重，固定种子7和四组真实 Warehouse 路线。
- 新 AMP 候选：由公开 PACER 初始化，在固定2848条真实 AMASS 训练动作上，进行512次 PPO+AMP 更新；每次128个环境、32个实际30Hz控制步，物理120Hz。四组归一化统计和教师保持冻结。
- 新训练奖励：原轨迹/能耗/方向奖励基础上，提高AMP权重至1.5，增加当前同一真实训练片段的关节姿态和根速度奖励，以及有效关节动作变化惩罚。姿态权重0.35、速度0.25、动作变化0.01、教师KL0.2。训练学习率3e-6、判别器2e-6。
- 当前ProtoMotions SMPL和PACER的69自由度顺序经实际读取核对一致；有效动作57维。关节动作/姿态目标皆使用同一顺序。既有冻结手/趾的控制契约保留。
- 新扩散引导：目标步行速度1m/s，规划上限1.2m/s；转弯速度目标连续下降，起步0.75秒平滑，路线前视6米。修改只影响软采样目标，物理根位置和关节姿态不会被覆盖。
- 原始场景：`/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Warehouse/warehouse.usd`，固定源SHA `fde4be69816f7f20e0cd1f4b517134e6bb5964fbaed0d8887983e9e8a64fc8ed`。采用既有native_reference_warehouse_floor_owner_v5地图和唯一参考平地，原始非地面静态障碍保留。此地面适配与作者原始地面协议有差异。
- 同模型对照：候选与旧权重各运行60秒，保留原引导和5米前视，以核对AMP效果。速度优化组合再运行200秒。相同初始动作抽样、场景、路线、种子，闭环轨迹会随策略变化；不能宣称严格相同执行动作的配对。
- 独立评估：冻结的22条验证动作/15关键点/30Hz评估基底；真实足支撑和所有24身体部位的静态/行人正常接触力，逐个120Hz子步测量。34条测试动作不用于本轮模型选择。
- 运动MMD是本地评估实现，不能与论文表III数值直接比大小。足部刚体原点的接触时速度是滑移代理，不等于接触斑真实摩擦滑移。
- 公共PACER训练动作列表未知；只能确认本地训练与验证的原始trial分离，不能保证对作者预训练完全独立。

## 原始资料

- [NavIsaacLab论文，IV-B/IV-C](https://arxiv.org/html/2606.26265v1#S4.SS2)：轨迹扩散与PPO+AMP行人模块，2848 AMASS片段，8场景/3–15行人/200秒评估。
- [Trace and Pace论文与项目](https://research.nvidia.com/labs/toronto-ai/trace-pace/)：轨迹扩散、清洁轨迹引导、物理全身行人控制和实际状态闭环。
- 本地参考代码：`/project/EXP/SangforAndSaut/TraceAndPace/trace`、`/project/EXP/SangforAndSaut/TraceAndPace/pacer`。

本实验是公开方法的工程复现与局部优化。公开TRACE轨迹表示和状态AMP判别器，与Nav作者私有配置/动作判别器不同；不声称完整逐项复现论文实验、达到论文全部指标或实时渲染运行。

## 审计

所有训练、场景和视频由拥有子进程的父进程记录 `subprocess.Popen.wait` 的实际退出码。源文件和输入在运行前冻结；退出0、资源身份、全部控制步、真实接触、运动质量分别核查。失败尝试保留，不通过强制退出掩盖SDK清理错误。

视频已由通过物理评估的实际状态逐帧渲染到同一原始Warehouse USD；明确标注源物理时刻。渲染阶段不生成新物理、不提供动画替代。
