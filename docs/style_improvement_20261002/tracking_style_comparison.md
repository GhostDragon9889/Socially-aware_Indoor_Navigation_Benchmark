# 三个检查点的真实运动跟踪与风格对照

本报告读取已经完成的 IsaacLab 物理捕获，在 CPU 上计算诊断。全部 22 条留出动作、11,982 个真正的 post-step 帧和所有跌倒帧均保留；没有修改原始跟踪指标，也没有搜索最佳时间或姿势对齐。结果是当前重建方法的对照，不能当作论文实验复现或单变量因果结论。

## 1. 原始指标和冻结风格协议

| 检查点 | 已训练更新 / 实际物理 transitions | 位置误差 m | 旋转误差 rad | 完整成功 / 22 | 跌倒动作数 | 整体 MMD² | 直立子集 MMD² |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 independent 1000 | 1000 / 2,048,000 | 0.377293 | 0.859479 | 6 | 1 | 0.093668 | 0.091640 |
| 新 AMASS2848 child 500 | 500 / 4,096,000 | 0.441347 | 1.027874 | 1 | 0 | 0.050227 | 0.050227 |
| 新 AMASS2848 child 1000 | 1000 / 8,192,000 | 0.430993 | 1.254071 | 5 | 2 | 0.206480 | 0.045422 |

更新数与物理 transitions 均直接读取实际 checkpoint，使用 `weights_only=True`、CPU、mmap。新 500 是新训练的一半预算，已经使用旧 1000 的两倍物理样本；新 1000 为旧 1000 的四倍物理样本。表中的 500 属于中间检查点诊断，不是等训练预算比较。原始成功判定及其阈值完全沿用捕获结果。

三个风格评估使用完全相同的冻结 [heldout_reference.json](style_velocity_adapter_legacy_cpu/heldout_reference.json)，SHA256 为 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`：seed 7、validation 的 256 个 61 帧窗口、15,360 个合法参考区间。每个生成捕获丢弃 22 个 initial reset 帧，得到 11,960 个同动作风格区间和 11,938 个合法加速度三元组，21 个跨动作拼接差分不计入。训练父 source 与留出 source 的交集均为 0；这不证明跨数据集人体身份或公共 PACER 预训练数据独立。

MMD² 使用已有的 90D 特征：15 个节点的 Pelvis 相对、heading 对齐、固定骨链尺度位置，及乘以 0.1 秒的局部速度。核为 Gaussian、sigma=1，含对角项的 biased MMD squared；在固定物理速度 bins 内最多各取 256 个样本，至少 30 个样本才能支持一项，最后对 5 个支持 bins 等权平均。这些是显式重建选择，不能直接与论文中未公布相同 embedding、带宽及采样细节的 MMD 数值比较。完整采样索引见 [三检查点评估 JSON](tracking_style_v1/three_checkpoints/evaluation.json)。

## 2. 后半程退化与跌倒帧

| 诊断 | 旧 1000 | 新 500 | 新 1000 |
|---|---:|---:|---:|
| 直立子集保留区间比例 | 99.2642% | 100% | 95.9448% |
| 跌倒状态结束的区间数 | 80 | 0 | 455 |
| 物理速度 < 0.1 m/s 区间数 | 255 | 194 | 463 |
| 其中结束于跌倒状态 | 55 | 0 | 414 |
| < 0.1 m/s bin 的 MMD² | 0.113467 | 0.074999 | 0.853910 |
| 整体速度分布 Wasserstein，m/s | 0.257637 | 0.213351 | 0.092944 |

直立子集的定义始终为 root-up z ≥ 0.8 且 Pelvis 高度 ≥ 0.7 m。它是解释性诊断，不能代替保留跌倒帧的主结果。新 1000 的最低速度 bin 中，414 / 463 个区间结束于跌倒状态；该 bin 的 MMD² 比旧模型增加 0.740443，等权平均贡献增加约 0.148089，其余四个支持速度 bins 的 MMD² 全部降低。因此“直立动作分布更接近参考”和“整体结果退化”同时成立；减少跌倒后静止姿势对分布的污染是需要实际解决的稳定性问题，不能靠删帧得到合格分数。

新 500 的风格和稳定性在这个捕获上优于新 1000，但其完整跟踪成功仅 1 / 22，不能据此选择它作为全面胜出模型。新 1000 的实际速度分布更接近参考，也出现更多位置判定成功，仍伴随朝向和稳定性退化。没有足部接触或真正 body COM 捕获，toe 近地、滑动速度和质量加权关节原点高度仍是代理量，不能宣称接触物理通过。

## 3. 根位置、朝向与局部骨骼分解

以下只做同一时刻诊断：每一方减去自己的 Pelvis 位置，并分别移除自己的 yaw；没有拟合任意旋转、修改原始世界坐标误差或搜索时间偏移。

| 帧加权平均诊断 | 旧 1000 | 新 500 | 新 1000 |
|---|---:|---:|---:|
| root XY 漂移，m | 0.311570 | 0.364234 | 0.319729 |
| root 高度绝对误差，m | 0.026945 | 0.020517 | 0.060828 |
| wrapped root yaw 误差，rad | 0.618995 | 0.750766 | 0.844917 |
| root-center + heading 局部位置误差，m | 0.135250 | 0.140894 | 0.166442 |
| 移除 heading 的 body 旋转误差，rad | 0.562577 | 0.600131 | 0.816396 |
| parent-relative joint 旋转误差，rad | 0.314163 | 0.325578 | 0.365508 |
| native 瞬时根平面速度，m/s | 0.661669 | 0.706498 | 0.830556 |

共同参考瞬时平面速度为 0.930120 m/s。这里速度来自捕获的 native root velocity；风格速度来自实际位移 / dt，两者统计定义不同。移除 heading 后，新 1000 的 body 旋转误差仍为 0.816396 rad，所以退化并非仅是全局朝向；局部关节误差也增大。旋转 geodesic 不能分解成可相加的“朝向贡献百分比”。

| 局部旋转诊断，rad | 旧 1000 | 新 1000 |
|---|---:|---:|
| L_Wrist heading-relative | 1.04955 | 1.79180 |
| R_Wrist heading-relative | 0.98429 | 1.77264 |
| L_Wrist parent-relative | 0.71143 | 1.04074 |
| R_Wrist parent-relative | 0.64985 | 1.06370 |

手部 body 的较大世界旋转误差主要继承腕部姿势，不能说冻结 hand joint 自身产生同等误差。全部 24 个 body、22 条动作和参考瞬时速度 bins 的分解见 [完整对照 JSON](tracking_style_comparison.json)。新 1000 跌倒来源为 motion 254（Walking001）和 261（Moonwalk001）；旧模型为 259（DanceTurns001）。这些特殊动作属于固定留出清单，主指标继续保留。

另一个明确标为诊断的共同无跌倒动作面板，排除三个模型跌倒来源的并集 `{254,259,261}`，保留相同 19 条 / 10,832 帧。旧 / 新 500 / 新 1000 的 heading-relative body 旋转误差为 0.550812 / 0.592503 / 0.748804 rad；root yaw 为 0.505761 / 0.669887 / 0.754830 rad。即使排除这三条，局部旋转与朝向仍退化。该面板按已观察结果选择，有选择偏差，不属于接受门槛，也不会改变原始 22 条主结果。

三次 reset 的最大“每条动作 24 body 平均”初始误差仅 1.447e-6 m / 1.120e-6 rad，支持映射和初始位姿正确。单独 body 最大值为 3.439e-6 m / 2.325e-6 rad，不能宣称每个 body 都低于 1.5e-6。新 1000 有 286 帧 root-heading 水平投影长度 < 0.1；其中 yaw 在接近竖直投影时解释能力有限，完整原始 SO3 误差仍保留。

## 4. 不能单因归因的变化

旧训练为 319 whole clips / 319 父动作；新训练为 2,848 个不重叠 segments / 1,853 父动作，非作者未公开的精确 2,848 个原始 IDs。旧 / 新实际 TrainingConfig 有 **14 项差异**：minibatch_size、amp_normalization、expert_dof_velocity_frame、discriminator_updates_per_update、discriminator_learning_rate、discriminator_batch_size、warm_start_normalizer_count_cap、warm_start_discriminator、effective_action_likelihood、motion_symmetry_coefficient、discriminator_replay_capacity、amp_reward_weight、target_kl、discriminator_input。具体旧新值和 checkpoint 哈希保留在 JSON。

此外，训练并行 actor 数为 64 → 256；旧捕获的 initial reset 和 AMP 专家速度保持 `legacy_stored` / parent-joint，新的明确使用 child-joint。poses、位置/旋转误差计算和当前物理配置相同，但参考速度初始条件是评估协议变化，不能声称只换模型权重。新 state-only D 是公共 AMP/PACER 基线选择，不是 NavIsaacLab Eq2 的字面 state+action D。

三个测试输入都是“10 个 +0.5…5 秒真实未来 root XY 点”的低层 oracle tracking，不包含高层 diffusion 导航。它们不能视作 shared navigation 场景或论文八场景评估。

## 5. 下一轮有证据支持的实验

1. 保留已验证的 child-joint 速度修复，不用回退坐标错误来追求分数。固定新数据、物理设置及其他参数，做显式方向奖励单变量消融；原论文打印公式与替代负向惩罚都保存独立 profile / checkpoint。现有后退比例趋势支持把它作为待验证假设，尚不证明下一轮提升。
2. 每个检查点继续报告全部 22 条主指标、跌倒和 inclusive MMD²；同时记录直立子集及停滞/跌倒 bin 的解释性变化。500 → 1000 的本次轨迹说明不能仅按训练 reward 或直立风格改善判断收敛。
3. 朝向仍错误时，优先看 wrapped root yaw 和根速度朝向；腕部 parent-relative SO3 误差作为另一条独立信号。分布风格接近参考并不保证逐时刻复现原始腕部相位，不应把这两个目标混为同一指标。
4. 后续独立足部测量需包含真实 floor-only 力、4 个物理子步、接触条件下的速度与真实 COM；当前代理量无法确认脚滑或支撑稳定性的原因。不能把单次 neutral stance 倒地或传感器读取成功当作步态质量通过。

## 6. 文件与复跑

- [完整三模型风格评估](tracking_style_v1/three_checkpoints/evaluation.json)
- [全部主指标、分解、配置差异、来源 SHA 和诊断统计](tracking_style_comparison.json)
- [输入契约和旧分数一致性验证](tracking_style_input_review.md)
- [旧模型分解](legacy_tracking_decomposition.json)、[新 500 分解](amass2848_child_step500_tracking_decomposition.json)、[新 1000 分解](amass2848_child_tracking_decomposition.json)

CPU 汇总复跑：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python \
docs/style_improvement_20261002/summarize_tracking_comparison.py
```

该脚本只读取捕获、冻结评估和本地 checkpoint，并写本目录诊断 JSON。没有启动 Kit、推进物理、改变 checkpoint、训练源码或原指标。
