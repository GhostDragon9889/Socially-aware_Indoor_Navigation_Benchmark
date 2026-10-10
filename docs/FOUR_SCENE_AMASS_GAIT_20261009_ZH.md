# 四场景 AMASS Diffusion + AMP 步态优化与复现

## 本次交付

固定本地 2,848 条真实 AMASS 专家动作继续微调，四个指定场景各完成一次真实 PhysX 60秒测试及一段60秒录像。YOHO / NewMetro 为4人共享场景，二楼 / 三楼为1人狭窄区域。本地测试与录像已完成；这是有限场景工程复现，不能称为作者八场景总体效果。

- [YOHO：60秒](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/gait_four_scene_optimization/videos_final_v21/YOHO_60s.mp4>)
- [NewMetro：60秒](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/gait_four_scene_optimization/videos_final_v21/NewMetro_60s.mp4>)
- [SecondFloor：60秒](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/gait_four_scene_optimization/videos_final_v21/SecondFloor_60s.mp4>)
- [ThirdFloor：60秒](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/gait_four_scene_optimization/videos_final_v21/ThirdFloor_60s.mp4>)

每段 H.264、960×720、30fps、1,800帧、60.000秒。视频逐帧渲染同一次已验收物理归档，两个连续源区间[0,900)、[900,1800)切换相机。没有插帧、加速、循环旧片段或修改人体姿态。录像不是第二次物理实验。

二楼原始照明较暗，录制明确使用RTX Film ISO200→800（名义+2EV）及色调映射操作6，提高腿部可见性；每帧读取并核对实际ISO。原场景灯光/几何和物理状态保持。参数依据 [IsaacSim RTX曝光文档](https://docs.isaacsim.omniverse.nvidia.com/latest/action_and_event_data_generation/ext_replicator-agent/ext_isaacsim_replicator_agent_visual_noise_example.html)，真实设置和像素亮度对照均保留。

## 实际物理测试

| 场景 | 行人数 | 实际时长/控制帧 | 全帧平均速度m/s | 跌倒 | >1N场景/行人接触帧 | 每人地面支撑子步比例下限 |
|---|---:|---|---:|---:|---|---:|
| YOHO | 4 | 60s / 1800 | 0.651 | 0 | 0 / 0 | 99.97% |
| NewMetro | 4 | 60s / 1800 | 0.663 | 0 | 0 / 0 | 99.96% |
| SecondFloor | 1 | 60s / 1800 | 0.530 | 0 | 0 / 0 | 99.74% |
| ThirdFloor | 1 | 60s / 1800 | 0.366 | 0 | 0 / 0 | 99.42% |

四场均正常 `subprocess.Popen.wait()==0`，完成原生清理、1,800控制帧 / 每人7,200真实120Hz物理子步、初始化后零重置。每场全部11项独立可靠性检查及行走姿态筛查通过，保留全部帧。超过1N的接触按原生法向力判定；它不测摩擦或接触面滑移。几何净空项是根圆盘代理，24身体的真实力档案独立复核。

每个录像区间的父进程实际退出0，全部1,800帧 / 24身体的视觉FK审核误差小于1mm。原始场景USD和AMASS资源保留原位；运行使用散列绑定静态引用层及原生物理网格。

## 与旧部署权重的成对比较

YOHO相同场景、seed7、初始AMASS动作/帧、TRACE权重、引导参数和60秒预算，仅AMP检查点不同。旧权重 `8ee87e…` 保留；新权重 `0cd961343d9828f95b0fcd6ed38ca8fb391140115c1a8fe808ff4ec1884434a8`。完整比较：[成对报告](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/gait_four_scene_optimization/paired_gait_comparison_v21.json>)。

| 指标 | 旧权重 | 新权重 | 相对变化 |
|---|---:|---:|---:|
| 双侧前后摆臂90%范围 | 13.105° | 18.522° | +41.3% |
| 每人较少活动侧屈膝周期范围 | 18.930° | 22.616° | +19.5% |
| 每人较少活动侧脚部原点周期抬升 | 1.969cm | 2.460cm | +24.9% |
| 双腿屈膝范围相对差异 | 0.215 | 0.222 | +3.2% |
| 双侧脚部原点抬升相对差异 | 0.328 | 0.405 | +23.4% |
| 真实单脚支撑双侧相对差异 | 0.289 | 0.186 | -35.5% |
| 含停顿的全帧平均速度 | 0.566m/s | 0.651m/s | +15.0% |
| 全帧身体侧倾p95 | 4.519° | 5.770° | +27.7% |
| 全帧肩部相对扭转p95 | 11.304° | 13.501° | +19.4% |
| 验证22全量姿态MMD²（越低越接近） | 0.040 | 0.038 | -5.3% |
| 验证22共同速度箱姿态/速度MMD²（越低越接近） | 0.074 | 0.075 | +1.9% |

摆臂与抬脚为实际身体几何诊断；周期统计只描述检测到的行走周期。速度与转弯会影响它们。脚部身体原点不是皮肤脚底或接触点，不能据此宣称完全消除拖腿。左右活动幅度和风格分布仍有差异；提升和退步均保留在报告中。

## 本轮训练与代码

- 最新v9：128环境、768次PPO+AMP更新、3,145,728真实物理转移，训练零跌倒；全部2,848专家ID进入采样。导出和四组冻结归一化统计通过独立CPU核验。
- 初始化：v8真实AMASS微调权重 `f17f01edc35a6209983402e9605c3bead65e0d65292fe12d57729c07ea5cd842`，另存可复现副本。原始公开PACER教师只用于训练约束，部署无需教师或AMASS参考姿态。
- 3,314个真实两秒行走窗口来自训练集内1,317源动作，用于物理复位、连续步态及手臂侧向/高度先验。左右共享幅度目标；手臂前后幅度取真实训练分布。专家判别器仍采样全部2,848，而非只使用窗口来源。
- 转弯/停止课程只生成高层XY意图；实际身体由AMP关节动作与PhysX推进。源AMASS当前参考姿态保持；actor类别与视野相关，不声称完整交叉均衡。未使用的后续候选课程没有计入训练。
- 方向相关的MJCF胶囊/盒子XY投影结合实际线/角速度进行保护；它改变制动意图，不覆盖实际姿态。修正原先竖直碰撞体三维半径导致的水平过度保护。
- TRACE使用公开DDPM原权重；本轮调整目标速度、转弯、地图最危险边缘与有界轨迹引导，没有训练新的Diffusion网络。短路线启用静止航向的路线意图，避免undefined heading带来的停滞。
- 二楼静态引用层清除指向停用PhysicsScene的429个归属目标，使原碰撞体进入唯一运行场景；原USD、碰撞形状和变换不变。四张10cm地图以PhysX体积查询、地面射线、逐碰撞体注册证据建立，属于本地 `physx_cell_queries.v1` 协议。

源码入口：`CrowdSim/humanoid/training/walking_turn_curriculum.py`、`temporal_gait_prior.py`、`kinematic_gait_prior.py`、`runtime/body_map_guard.py`、`runtime/trace_pace.py`、`scripts/finetune_pacer_amass_verified.py`。本轮窗口/投影的4项CPU回归测试通过；完整实测另有独立收据。

## 数据隔离及独立测试

训练库实际SHA256：`46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`；训练前后实际读取完整约1.156GB库，文件SHA、大小、inode、mtime/ctime相同。清单SHA256：`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`。

2,848训练 / 22验证 / 34测试源路径互斥。34条测试仅在四场景和成对步态检查完成、模型选择封存后读取一次；没有用于梯度或参数选择。15关键点风格基底沿用冻结验证集，不拟合生成帧。独立测试保留全部合法相邻区间，原生COM/接触与姿态取同一末子步。

| 场景 | 测试34姿态Gaussian MMD² | 共同速度箱姿态/速度MMD²均值 |
|---|---:|---:|
| YOHO | 0.07769 | 0.08091 |
| NewMetro | 0.07969 | 0.07634 |
| SecondFloor | 0.07384 | 0.15132 |
| ThirdFloor | 0.07527 | 0.14560 |

这些是本地诊断而非作者MMD数值，不能直接与论文表格比较。公开PACER预训练与AMASS重叠未知；本地源路径互斥不是全训练历史无泄漏证明。

## 整理后的复现入口

- 正式权重/初始化/发布索引：`/project/EXP/SangforAndSaut/ProtoMotions/data/pretrained_models/local_amass_finetune/walking_turn_20261009`。
- 四场景配置、路线、相机、训练配方/先验/窗口：`/project/EXP/SangforAndSaut/ProtoMotions/configs/diffusion_amp/four_scene_walking_20261009`。
- 完整原始结果、失败历史、脚表面诊断和录像收据：`/home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/gait_four_scene_optimization`。原散列证据不搬迁、不重写。
- [发布索引](/project/EXP/SangforAndSaut/ProtoMotions/data/pretrained_models/local_amass_finetune/walking_turn_20261009/release.json)、[模型选择](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/gait_four_scene_optimization/selected_model_v21.json>)、[独立测试34](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/gait_four_scene_optimization/final_test34_style_v21.json>)。

环境固定IsaacSim6.0.0.1 / IsaacLab3.0开发提交 `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`（产品VERSION3.0.0，Python包版本12.0.0）、Python3.12.13、torch2.11.0+cu128。环境说明：`env/isaaclab3_sim6/profile.json`。保持 `PYTHONDONTWRITEBYTECODE=1`，一次只运行一个原生SDK/GPU任务。

### 重新训练本轮768更新

```bash
cd '/project/EXP/SangforAndSaut/ProtoMotions'
NAV_PY='/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python'
NAV_CFG='/project/EXP/SangforAndSaut/ProtoMotions/configs/diffusion_amp/four_scene_walking_20261009'
NAV_MODEL='/project/EXP/SangforAndSaut/ProtoMotions/data/pretrained_models/local_amass_finetune/walking_turn_20261009'
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/finetune_pacer_amass_verified.py \
  --output /tmp/fresh_amass_walking_train --updates 768 --num-envs 128 \
  --parent-checkpoint "$NAV_MODEL/initialization_compact_v8.pth" \
  --optimization-config "$NAV_CFG/locomotion_walking_turn_v9.json" \
  --gait-prior "$NAV_CFG/gait_prior_compact_v3.json" \
  --temporal-gait-prior "$NAV_CFG/temporal_gait_prior_v2.json" \
  --walking-turn-windows "$NAV_CFG/walking_reset_windows_v9.json" \
  --task-horizons-s 0.5 1.0 2.0 4.5
```

### 重新实测及录制一个场景

以下以YOHO为例，另三场替换场景名，各使用新输出目录。发布配置只改变检查点路径，权重SHA和实际测量时的源配置均保存在发布索引中。

```bash
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/run_trace_pace_verified.py \
  --output /tmp/fresh_YOHO_physics -- \
  --config "$NAV_CFG/YOHO_profile.json" --scene "$NAV_CFG/YOHO_scene.yaml" \
  --steps 1800 --measure-contacts --seed 7
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/evaluate_trace_pace_reliability.py \
  --run /tmp/fresh_YOHO_physics --execution /tmp/fresh_YOHO_physics_execution.json \
  --output /tmp/fresh_YOHO_quality.json
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/evaluate_trace_pace_posture.py \
  --run /tmp/fresh_YOHO_physics --output /tmp/fresh_YOHO_posture.json --require-walking-screen
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/record_native_pedestrians.py \
  --run /tmp/fresh_YOHO_physics --quality /tmp/fresh_YOHO_quality.json \
  --posture /tmp/fresh_YOHO_posture.json --camera-plan "$NAV_CFG/YOHO_cameras.json" \
  --output-dir /tmp/fresh_YOHO_video --tag YOHO
```

应核对父进程实际退出码、源码/输入SHA、完整清理、物理/姿态报告和最终 `ffprobe` 的1800帧/30fps/60秒。历史失败保留，不能通过跳过清理、删除问题帧或降低门槛改写为成功。

## 与论文的范围差异

方法对应 [NavIsaacLab IV-A/IV-B](https://arxiv.org/html/2606.26265v1#S4.SS2) 的Diffusion轨迹与AMP关节控制闭环；使用 [Trace and Pace](https://research.nvidia.com/labs/toronto-ai/trace-pace/) 公开模型及 [AMP](https://xbpeng.github.io/projects/AMP/) 的对抗动作先验。

本轮附加步态奖励、课程、原生网格和保护策略是本地工程选择。作者精确2,848动作清单、定制权重及八场景评估清单未公开匹配；四个本地60秒、seed7、4/4/1/1行人、声明的有限平地与作者8场景/200秒/3–15人不同。路线设定不代表任意拥挤场景泛化。没有声称达到或超过原论文整体指标，也没有证明完全消除拖腿或达到人类感知真实性。

## 完整录像抽样检查范围

完成录像后，实际查看四场景八个镜头的局部第450/800帧，另查看三楼/二楼部分第200帧；对应完整片段15.033/26.700/45.033/56.700秒。全部视觉FK和物理诊断保持全帧覆盖。这是代表帧抽查，不是连续人工播放60秒或人类感知实验。证据：[录像抽查记录](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/gait_four_scene_optimization/postselection_video_review_v21.json>)。

YOHO物理场景有四人，两个局部区域各两人，前后相机分别展示两个区域。第二个视角的部分地面呈灰色，保留源场景引用的实际渲染结果，未补贴图或改造几何；辅助物理参考地面在录像中隐藏。NewMetro同一镜头可见四人，二楼和三楼各一人。

四段成片仍由完整真实物理状态渲染；抽样检查没有触发测试集之后的模型、训练配置或物理参数调整。脚部自然性与人群密度的剩余范围应结合录像和上述成对指标判断，不能仅凭抽样图认定拖腿已消除。
