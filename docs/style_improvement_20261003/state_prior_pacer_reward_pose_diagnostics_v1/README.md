# 同帧关节与足部几何诊断

本目录为 CPU 诊断接口，不修改 935 训练源码、主评估指标、默认模型或质量门槛。新 reward `.5` 权重及未来物理结果尚未由本接口评估；旧 `19f3341b…` 的完整 22 条 validation 轨迹用于实际检查接口。

## 如何解释现有指标

固定 90 维 style 特征为 15 个关键点的 pelvis / heading 相对位置，以及乘 `.1 s` 的对应速度。它不使用身体 quaternion，也不含 Hand 关键点；例如腕关节旋转改变手的朝向，却不改变 wrist 原点时，style 距离可能无法看到这项改变。MMD 衡量固定特征的分布差异，不衡量逐帧目标腕部转角。

这是离线 90 维评价的观测限制。训练用的 195 维 PACER AMP 特征另外包含 19 个关节三元组的旋转编码和速度，其中有 L_Wrist / R_Wrist；不能据此推断判别器没有腕部风格信号。训练仍没有唯一同帧参考上身目标。

同帧关节诊断直接复用冻结的 [原 decomposition](../state_prior_metrics_v1/tracking_decomposition/decomposition.json) 和 `12240485…` 工具输出，未重写转角数学：

| 诊断 | 可以观察 | 解释限制 |
|---|---|---|
| world 身体位置 / SO(3) 转角 | 全局漂移与各关节姿态偏差 | 包含根的平移、方向与倾斜 |
| rootcenter + heading 位置 / 转角 | 去掉各自 root 平移和 yaw 后的姿态差异 | 仍按同帧比较，不匹配运动相位 |
| full-root-relative 转角 | 去掉完整 root 朝向后的身体差异 | 额外描述，不能替代主 world 指标 |
| parent-relative 转角 | 各自 MJCF 父关节与子关节的相对旋转差异 | 不能把层次角误差当作可相加的因果贡献 |

旧 state-prior 22 条 validation 的实际均值如下：

| body | world 转角 rad | parent-relative 转角 rad |
|---|---:|---:|
| L_Wrist | 1.373325 | .698063 |
| R_Wrist | 1.215102 | .682966 |
| L_Hand | 1.374211 | .002716 |
| R_Hand | 1.214790 | .004027 |

手相对腕的转角误差很小，手的 world 转角误差接近腕的 world 误差；这支持首先检查腕和上游关节，而不能根据手的全局误差直接判断手关节控制失败。该判断是对保存轨迹的解释，尚未证明具体训练原因。当前 XY-conditioned AMP 没有唯一 3D 上身姿态目标，同帧 mocap tracking 和动作分布质量需同时保留。

AMP 用动作数据学习风格奖励，与任务奖励一起优化控制策略；它允许动作数据无任务标签。[AMP 作者论文](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf) PACER 以二维路径控制物理行人，包含地形和社会条件。[PACER 作者项目](https://research.nvidia.com/labs/toronto-ai/trace-pace/) PACER+ 另行报告 root/global 位置、脚滑和抖动等指标，这说明不同属性需要不同观测；本地诊断没有宣称复现它的未核实指标实现。[PACER+ 论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_PACER_On-Demand_Pedestrian_Animation_Controller_in_Driving_Scenarios_CVPR_2024_paper.pdf)

## 足部诊断的定义

`diagnose_tracking.py` 从已保存 `rigid_body_pos` 与 `reference_rigid_body_pos` 取 L_Toe / R_Toe 的 **world body origin**。完全保持 `b09fa58c…` 的 `StyleSpec`：相邻两个 post-step 帧都满足 `z <= .12 m`，且 `|Δz/(1/30 s)| <= .2 m/s`；报告 `||Δxy||/(1/30 s)`，以及超过原 `.1 m/s` 的比例。ground 固定为原 flat tracking floor 的 `0 m`。

分别报告 actual / reference 的几何 mask，以及同一相邻帧上两者同时满足 mask 的 common 子集。参考速度分层沿用 decomposition 的 `[0,.5,1,1.5,2,3,∞)`，取 interval end 保存的 `reference_root_vel`。这些 mask、分层和阈值没有在新结果上拟合。全部失败、跌倒和负高度保留；空集合明确报告 `None`。

Tracking 未保存接触力、接触点或摩擦力，因此这些值是近地 toe-origin **几何代理**。摆腿、滚动以及 body origin 和接触点的差异均可能产生非零值；本接口不能测量真实接触片脚滑。已有导航 `ground_support.npz` 的受载 body-origin tangential speed 也应保持原观测定义，不能改称接触片速度。

旧 19f 实际 CPU 执行结果（22 clips / 11,960 合法 interval）如下：

| toe | actual mean / median m/s | reference mean / median m/s | actual / reference p95 m/s |
|---|---:|---:|---:|
| L_Toe | .816257 / .037408 | .480766 / .073177 | 3.527974 / 3.028262 |
| R_Toe | .765257 / .008677 | .589887 / .085157 | 4.000847 / 3.439183 |

对应 mocap reference 本身也有很大的代理速度，不能把 actual 均值直接解释为真实接触滑动。原 style 报告的 pooled 15,572 足部 interval 及各统计已逐项重现。保存的 [全 interval archive](baseline19f_actual/toe_intervals.npz) 包含左右足 actual/reference 速度、mask、共同速度差、参考 frame、motion ID 和跌倒标签，可定位任一异常，无删样本。[实际完整报告](baseline19f_actual/diagnostics.json) / [实际 closed0 receipt](baseline_actual_CPU_receipt.json)

## 新权重完成评估后的调用

须先由已独立审查的固定 CPU metrics 完成 `tracking90` 和 `tracking_decomposition`，并生成实际 `actual_source_admission.json`。新接口只接受已注册的完整 validation capture 和这些已通过的结果；不读取模型、训练 bank 或 sealed34 test。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
docs/style_improvement_20261003/state_prior_pacer_reward_pose_diagnostics_v1/diagnose_tracking.py \
--input output/diffusion_amp/pacer_reward_tracking2000_v1 \
--decomposition docs/style_improvement_20261003/pacer_reward_metrics_v1/tracking_decomposition/decomposition.json \
--style-report docs/style_improvement_20261003/pacer_reward_metrics_v1/tracking90/evaluation.json \
--output docs/style_improvement_20261003/state_prior_pacer_reward_pose_diagnostics_v1/new935_actual
```

当前仅完成旧结果的 CPU 运行（`.447 s` / exit0）、三个必要解析 fixture（`3 passed`）与 Ruff。未来命令未执行；新增接口须完成独立审查后用于新结果。
