# 公共 AMP 姿态先验：完整训练、跟踪与原生行人模拟

新配置已合并，实际 fresh2000 完成 **16,384,000 次物理转换**，训练与 watcher 退出0。最终权重通过2,062项独立状态检查；完整22条跟踪及四人200秒导航均已实际结束。原生Warehouse四人与15人200秒后续均已完成，默认部署权重保留原step1000。

## 本轮实现

`amp_style_mixed_state_action_state_prior_terrain_protected.yaml`显式启用公共PACER状态判别器与AMP历史归一化统计的耦合初始化。1950历史输入扩展到2019；69动作列只在初始化时归零，57有效列随后学习，恢复保留已学权重。默认开关false；未加入额外奖励或改变2848 AMASS、验证22／测试34。

[合并后44新增＋224回归](../merged_action_state_prior_CPU_v1/README.md)、[V11生产183文件](../native_runtime_epoch_v11/native_pedestrian_source_v11.json)、[实际934文件隔离训练来源](../../../output/isolated_training/mixed_state_action_terrain_state_prior_v1/source_manifest.json)、[正式终态准入](../state_prior_training_audit_v1/terminal2000_admission_v1.json)均保留。

### 新权重与重运行

[AMP step2000权重](../../../output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_snapshots_v1/step_2000.pt)的SHA256为`19f3341b9bb6118243c0945d839a669360175d93a35dde0f792639178d508f4f`。下列命令从仓库根目录显式选择本轮实验模型，使用新输出目录；默认部署仍采用原step1000。

```bash
PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/simulate_paper_pedestrians.py \
  --amp-checkpoint output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_snapshots_v1/step_2000.pt \
  --config configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --scene-config configs/diffusion_amp/native_warehouse_pedestrians_15_clearance08_v1.yaml \
  --actors 15 --duration-seconds 200 --seed 7 \
  --inference-only --native-dynbench --arm-style-alpha 0 \
  --output output/diffusion_amp/native_warehouse_state_prior_15_review_run
```

这是重运行方法，以上新目录尚未运行。先结束其他Isaac GPU任务；同seed不保证PhysX内部缓存或规划噪声逐位一致。实际已结束的本轮命令与字节来源以[原始启动记录](../state_prior_native_actual_runs_v1/native15_v2_actual_launch.json)为准。

## 完整22条物理跟踪

| 模型 | 成功／22 | 跌倒 | 位置 m | 旋转 rad | 固定90D MMD² | 速度W1 m/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 当前部署state-only1000 | 11 | 0 | 0.300708 | 0.861454 | 0.046821 | 0.143767 |
| 原state-action2000 | 17 | 0 | 0.208382 | 0.725853 | 0.062530 | 0.058332 |
| 地形统计保护2000 | 15 | 0 | 0.247776 | 0.775494 | 0.054529 | 0.052493 |
| 本轮公共先验2000 | **17** | **0** | **0.209859** | **0.810943** | **0.042092** | **0.055477** |

全部11,982计量帧保留；五条失败ID245／254／259／260／261也计入主结果。原工具排除每条初始化帧，共22帧，11960个同clip风格区间；不通过删除失败改善分数。与匹配的上一轮地形保护模型相比，成功率及位置改善、旋转误差变差；不能只挑更好的指标。

[实际跟踪](../../../output/diffusion_amp/state_prior_tracking2000_v1/result.json) · [固定90D](../state_prior_metrics_v1/tracking90/evaluation.json) · [独立1914项参考配对](../../state_prior_tracking_reference_peer_v1/actual_reference_pairing.json) · [跟踪指标QC](../state_prior_CPU_metrics_peer_review_v1/actual_tracking_review.json)。

### 上肢误差定位

[完整24关节并排记录](joint_error_comparison.json)保留所有22条帧。相对匹配的上一轮地形保护模型，左右腕父关节旋转平均误差从0.438746／0.394357升至0.698063／0.682966rad，左右肘亦变差；相对当前部署state-only基线，腕误差又略好。这是具体诊断，不能把各误差相加或据此判断全部动作更自然。

## 完整四人200秒 Diffusion＋AMP

| 指标 | 当前部署基线 | 原state-action2000 | 地形保护2000 | 本轮公共先验2000 |
| --- | ---: | ---: | ---: | ---: |
| 跌倒 | 0 | 2 | 0 | **0** |
| 路线点 | 21 | 2 | 25 | **40** |
| 停滞人秒 | 194.7667 | 591.7333 | 93.8667 | **0.1333** |
| 90D MMD² | 0.07238060 | 0.07372768 | 0.09013555 | **0.06811695** |
| 90D支持速度桶 | 5 | — | 4 | **3** |
| 速度W1 m/s | 0.353368 | 0.738402 | 0.327363 | **0.164985** |
| 固定45D MMD² | 0.02015677 | 0.04759910 | 0.04140942 | **0.04696746** |
| K50覆盖 | 0.86 | 0.68 | 0.58 | **0.58** |
| K100覆盖 | 0.72 | 0.65 | 0.49 | **0.52** |

6,000控制帧、24,000共享物理子步、96,000 actor子步、384,000足部观测。四人路线点均10；零跌倒／时间截断／静态接触／peer接触，接触法向力阈值严格>1N。原始计量203项通过。停滞按同episode XY位移速度<0.1m/s计算，每人0／0.1／0／0.03333秒。部署确实使用Diffusion，生成1736个actor计划；训练为固定AMASS／procedural混合指令，未冒充训练期间使用扩散。

[真实导航结果](../../../output/diffusion_amp/state_prior_nav2000_v1/result.json) · [原始计量](../state_prior_metrics_v1/navigation_raw/audit.json) · [停滞](../state_prior_metrics_v1/navigation_raw/stagnation.json) · [90D](../state_prior_metrics_v1/navigation90/evaluation.json) · [固定45D](../state_prior_metrics_v1/navigation45/evaluation.json)。

## 原生 Warehouse 实际后续

新权重已完成四人200秒真实模拟并退出0：[实际结果](../../../output/diffusion_amp/native_warehouse_state_prior_clearance08_200s_v1/result.json)。6,000控制帧，零跌倒／时间截断，44个路线点、每人2个完整循环，共8轮；[987项独立原始审计](../state_prior_native_actual_review_v1/native4_audit.json)通过，静态／peer接触均零，真实COM重构最大差1.907e−6米。[另一独立80项数组复核](../state_prior_native4_actual_peer_QC_v1/actual4_QC.json)亦通过，原有一次路线索引未取模的QC假设失败保留，修正QC后真实mod4顺序通过，生产与原审计器未改。

本次部署使用生产V11的183文件，训练来自另一个934文件隔离副本；两者分别绑定。旧部署state-only模型的同场景四人记录有57路线点／12循环，当前新模型进展较低。15人200秒也实际结束，6,000控制帧、90,000 actor控制帧、360,000 actor子步、1,440,000足部观测；零跌倒／时间截断／静态接触／peer接触，160路线点、32完整循环。[1,241项实际原始审计](../state_prior_native_actual_review_v1/native15_audit.json)通过。旧15人基线211点／49循环，当前新模型进展仍偏低。首次读取器因tuple／JSON list表示差异拒绝，独立v2仅规范证明记录表示、原数学保持；失败日志保留。

[四人风格比较](../state_prior_native_metrics_v1/comparison4/comparison.json)：共同三桶MMD²基线0.06870245→新0.05244885，固定45D MMD²0.03829664→0.02551202，K50／100覆盖0.34／0.31→0.58／0.47，风格指标改善但路线进展下降。

[15人风格比较](../state_prior_native_metrics_v2/comparison15/comparison.json)也已实际完成，三项CPU命令首轮退出0：[封存运行记录](../state_prior_native_metrics_actual_runs_v1/native15_CPU_completion_v2.json)。原生四人与15人的测量独立于上面的四人fixture；不能跨场景挑选较好的数值。

| 原生Warehouse指标 | 四人旧→新 | 15人旧→新 |
| --- | ---: | ---: |
| 路线到达／完整循环 | 57／12 → 44／8 | 211／49 → 160／32 |
| 跌倒／静态接触／peer接触 | 0／0／0 → 0／0／0 | 0／0／0 → 0／0／0 |
| 共有速度桶数 | 3 | 4 |
| 共有桶等权90D MMD² | 0.06870245 → 0.05244885 | 0.09969936 → 0.06987130 |
| 固定45D姿态MMD² | 0.03829664 → 0.02551202 | 0.02831979 → 0.02273235 |
| K50／100覆盖 | 0.34／0.31 → 0.58／0.47 | 0.76／0.61 → 0.70／0.61 |

共有桶逐一均为新模型MMD²更低；15人K50覆盖下降、K100持平，归一化熵从0.63428／0.60748升至0.71237／0.70950。15人原主要90D均值旧0.09741570／新0.06987130，支持从5桶减至4桶；新1.5–2m/s仅21样本，保持未知。全部89,985同episode风格区间保留，参考中心未重拟合。[15人203项独立NumPy核对](../state_prior_native15_actual_peer_QC_v2/actual15_QC.json)与[指标v2来源审查](../state_prior_native_metrics_peer_review_v2/peer_review.json)通过。动作覆盖与路线进展的取舍仍需继续解决。

[四人／15人实际指标独立QC汇总](../state_prior_native_metrics_actual_peer_v1/actual_comparison_summary_v1.json)复核完整23,996／89,985区间、旧168／174与新183文件来源、固定参考、真实CPU退出及所有未知速度桶。共有桶比较作为补充指标保留，没有替换原主要汇总；也没有通过重拟合基底提高分数。

[四人／15人原始机器比较](native_raw_comparison.json) · [新模型30秒室内视频](../../../output/diffusion_amp/native_warehouse_state_prior_recorded_indoor_30s_v1/pedestrian_motion.mp4) · [首／中／末帧实际可见性](preview_visibility.json)。视频是450帧／15fps的保存物理状态回放，0新physics／policy调用，未修改姿态；[完整媒体CPU QC](../state_prior_recorded_replay_peer_v1/actual_media_audit_v1.json)通过：450 PNG源时间／episode匹配、全部450编码banner裁剪、H.264／1280×960／15fps／30s及53份USD原字节均核验，最终进程退出0。

[生产推理兼容性](../state_prior_native_compatibility_v1/actual_CPU_readiness_v1.json) · [真实四人启动与结束](../state_prior_native_actual_runs_v1/native4_actual_launch.json) · [修正版15人启动器审查](../state_prior_native_wrapper_peer_review_v1/v2_delta_review.json)。

[媒体来源独立复核](../state_prior_recorded_media_peer_review_v1/peer_review.json)亦通过，绑定真实退出和全部450 PNG／MP4原字节；不能据此声称自然度或论文风格优越。

## 选型与限制

预注册完整6000／零跌倒／至少21点／90D≤0.07238060315284697门槛全部通过，随后原生Warehouse四人与15人验证亦已完成。新90D仅支持0.1–0.5／0.5–1／1–1.5m/s三桶；其余样本不足的桶保持未知，主要均值与基线五桶支持总体不同。原主要90D均值下降与45D姿态距离、覆盖率退化同时报告。进一步只读双方共同支持的三桶，基线／新模型等桶均值为0.04098256／0.06811695，三桶逐一均为基线更好；支持总体变化解释了原五桶／三桶汇总的表面改善，不能称同速风格进步。默认模型不自动晋升。[三项指标独立QC与逐桶比较](../state_prior_navigation_peer_review_v1/actual_navigation_review.json)。

45D沿用原AMASS-only `b25dffa…`固定中心，没有重新fit生成动作。足部原点运动仅为滚动／滑动代理；未保存逐body peer原始矩阵，不能据此声称已重构该矩阵。GPU记录的SDK最终清理标记仍为false，不声称完整PhysX solver/cache逐位复现。

[完整机器比较与工件SHA](comparison.json)包含实际计数、支撑／COM、原门槛和历史冻结结果。2848条来自真实AMASS，但并非作者未公开的确切动作IDs；本地同seed／场景／指标不能证明已达到或超过论文八场景效果。[最新公开资源核查](../public_resource_refresh_v1/report.md)。

## 本轮完成状态

[最终汇总](final_iteration_comparison.json)绑定训练、fixture、两组原生风格比较和实际退出记录；原预注册fixture报告保持原字节。[最终源码与输入核验](final_source_input_integrity.json)确认生产183文件、隔离934文件及六份原训练输入均未变。当前四份指南已同步。

本轮实现与测量已完成；严格论文效果仍未达成。下一轮应同时改进上肢动作、共有速度区间的风格覆盖和路线进展，再扩展原生场景。作者完整八场景协议及确切动作清单尚未对齐，不能称达到或超过论文。[研究依据与下一步](STYLE_NEXT_STEPS_ZH.md)。
