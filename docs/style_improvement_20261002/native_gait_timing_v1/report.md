# 原生足力时序诊断：旧 negative 与 AMASS-root

**两份真实四人200秒记录；CPU只读分析，未改变风格指标、数据或模型选择。** [机器报告](report.json)、[全部原始支持 bouts](raw_bouts.csv)、[图表](timing.png)。

## 固定定义与可靠字段

使用120 Hz四子步 `foot_floor_normal_force_w[T,4,B,4,G,3]`，先累加 floor filter 向量，再以每足范数严格 `>1 N` 判接触；左/右侧为各 ankle/toe 的任意接触。没有去抖或删短 bout。固定实际 Pelvis link 水平速度 `<.1`、`.1–.5`、`≥.5 m/s` 分层，边界下含上不含。

每份保留96,000 actor-substeps、800 actor秒；reset/fall样本全部进入分母。bout在 episode 边界截断并标左右 censor；转换只统计同episode相邻子步。时长为采样数×1/120秒的离散曝光估计，转换时刻分辨率为8.33 ms，不能据此确定连续真实接触的精确起止。terminal control的四子步单列，但30 Hz跌倒判定无法定位其中哪一个子步首次跌倒。

原物理/source/COM审计的输入SHA与本次加载数据相同：旧negative使用 `ground_support_200s_actual_verification.json`；AMASS-root使用 `amass_root_step1000_navigation_ground_actual_verification_v1.json`。复用其冻结源码归属，不声称当前源码与历史运行相同。

## 实际读数

| 读数 | negative guide20 | AMASS-root1000 |
|---|---:|---:|
| 跌倒/已到达 route vertices | 0 /21 | 13 /5 |
| 平均实际根速度 m/s | .5643 | .3997 |
| 慢速子步占比 `<.1m/s` | 24.40% | 31.19% |
| 运动子步占比 `≥.5m/s` | 53.28% | 35.27% |
| 全部双支撑 /无足支撑 | 48.02% /.748% | 57.83% /.726% |
| 运动子步双支撑 /无足支撑 | 21.15% /.981% | 25.27% /1.542% |
| 原始左/右 touchdown 转换数 | 867 /867 | 725 /848 |
| 左/右接触 bout 数，含 censor | 868 /868 | 730 /853 |
| 运动均速 bout：左时长 P50 /P90 秒 | .558 /.650 | .492 /.675 |
| 运动均速 bout：右时长 P50 /P90 秒 | .542 /.650 | .550 /.727 |

这里的“运动均速 bout”按**整个原始bout平均根速度**分层；它不是把每个bout截成移动和停留片段。子步比例采用逐子步实际速度，二者分母不同，JSON明确分别保存。

旧negative actor0慢速占58.39%，最长左/右接触bout116.72/115.83秒；新模型最长左/右44.96/69.63秒。慢速段本来可能长期双支撑，说明全部样本的均值会混入站立/停滞比例。即便粗分运动段，新模型该段平均速度为.8273 m/s，旧模型为.9231 m/s，尚未精确匹配速度、转弯、碰撞或episode分布，因此不能把接触差异全部归因于步态风格。

新模型13次terminal actor-controls共有52个子步，全部保留；其中46.15%无足支撑。这里只描述触地状态与终止control的共现，不能证明无支撑导致跌倒。各actor/速度层的曝光、接触转换、真实COM世界高度及censored/uncensored时长分布均在JSON中。

## 可确认的风险与边界

1. 停滞与动作时序必须分开报告；pooled 双支撑比例不能单独代表行走自然性。
2. 两个模型均有8.33 ms的短接触bout；固定1 N原始阈值可能产生力阈值抖动，未经去抖的转换数不是生理步频。
3. 当前没有 AMASS 真实地面力、真实reference COM、足底接触点切向速度或摩擦力。本报告保留这些为null，不把 body-origin速度代理称为真实脚滑。无足支持不等于全身腾空；跌倒后躯干等其他部位仍可能接触地面。
4. 没有作者足力时序目标或统一临床自然性阈值，故本报告不判步态通过/失败，不替代已有 MMD、追踪或导航质量验收。

`native_capture_probe_passed=False`历史metadata原样保留；它与已独立确认的实际数据/传感器有效性范围不同。本次未读取封存34 test输出。

## 重跑

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/analyze_native_gait_timing.py \
  --run negative20=output/diffusion_amp/shared_amass2848_backward_penalty_ground_support_200s_v1=docs/style_improvement_20261002/ground_support_200s_actual_verification.json \
  --run amass_root1000=output/diffusion_amp/shared_amass_root_curriculum_ground_support_200s_v1=docs/style_improvement_20261002/amass_root_step1000_navigation_ground_actual_verification_v1.json \
  --output-dir /tmp/native_gait_timing_fresh
```
