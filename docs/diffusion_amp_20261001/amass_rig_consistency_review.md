# AMASS 与当前 neutral SMPL rig 一致性审查

## 结论

本次 CPU 抽查未发现骨长、局部 offset 或 AMP 四个 key body 与当前 neutral MJCF 不一致的证据；不支持“个体 shape 差异使 discriminator 可以直接区分真假、因而无法学习”的诊断。此结论限定于所抽查的 expert cache 与选定 rig，不证明训练收敛或动态物理状态无误。

所有入选 319 个 train 与 22 个 validation clip 均检查首帧、中帧、末帧；每个 split 另加 seed=123 的 512 个随机参考窗口当前帧，共 2,047 次帧采样。数据仅安全 mmap 读取，未改 AMASS、模型、训练或 SDK，未启动 Kit/GPU。

| 指标 | train | validation |
|---|---:|---:|
| 样本帧数（含重复） | 1469 | 578 |
| 不同全局帧数 | 1465 | 566 |
| 骨长绝对误差均值 / m | 3.42845804e-08 | 3.29763859e-08 |
| 骨长绝对误差最大值 / m | 4.2448797e-07 | 5.02578688e-07 |
| 逐骨长度最大 population variance / m² | 5.37268746e-15 | 5.18513914e-15 |
| 逐 clip 均值之间最大骨长 std / m | 4.6381644e-08 | 4.63977792e-08 |
| 由 parent rotation 还原 local offset 最大误差 / m | 5.31157725e-07 | 5.61463311e-07 |
| exp_map FK 全 body 最大位置误差 / m | 1.73673652e-06 | 1.17142463e-06 |
| exp_map FK 全 body 最大旋转误差 / degree | 0.000167883588 | 0.000205299983 |
| AMP 四 key body 最大位置误差 / m | 1.61813159e-06 | 1.09769076e-06 |
| AMP 最后 12 项最大坐标绝对误差 / m | 1.61677599e-06 | 9.23871994e-07 |

骨长最大误差约 **0.000503 mm**，四 key body 的最大 FK 位置误差约 **0.001619 mm**，属于浮点计算量级。四 key body 为 `R_Ankle, L_Ankle, R_Wrist, L_Wrist`。将存储的 key 位置换成当前 rig FK 位置并保持其他字段不变后，AMP195 的前 183 项完全相同，最后 12 项仅有表中微小误差。

## Body 顺序与资源证据

- 对比实际 XML 解析结果、`SmplRobotConfig().kinematic_info`、PACER common body order 与转换器 `SMPL_MUJOCO_NAMES`，24 个名字、parent indices 与 local positions 均一致。
- 原 AMASS joint order 先按名字转为 MJCF DFS order；DOF 为每个非 root body 的 xyz exp_map 共 69 项。存储 `grs` 为 xyzw，FK 输入 root quaternion 显式转为 wxyz。
- `.pt` 不嵌入 body names，不能仅按 24 维猜顺序。本次查证转换 producer、COMMON packing、实际 `motion_files` 索引及全部 341 份 conversion sidecar，并用完整姿态 FK 数值验证该顺序。
- 341 份 sidecar 记录的生成资产 SHA256 全部等于当前 MJCF：`58dc997c830b3ded2d2b4528797ed2768f9528b2f25aa008e541fe3a9083e7cc`。
- `prepare_amass_dataset.convert_one` 读取 `poses/trans`；未读取或应用 subject `betas`。`convert_amass_to_motion` 使用固定 neutral MJCF 执行 FK。因此此转换库保留动作姿态，统一了骨架形状，并未保留 AMASS 每个 subject 的骨长。

代码证据：[转换 worker](../../data/scripts/prepare_amass_dataset.py:103)、[命名重排与 FK](../../data/scripts/convert_amass_to_proto.py:151)、[打包 COMMON state](../../protomotions/components/motion_lib.py:750)、[实际 corpus 读取](../../CrowdSim/humanoid/amass_corpus.py:116)、[AMP195 key offsets](../../CrowdSim/humanoid/amp/observations.py:175)。此前独立真实 Isaac Lab 零物理步 FK probe 也匹配 exp_map：[isaaclab_smpl_fk.json](../completion_20261001/isaaclab_smpl_fk.json)。本次未重复启动该 probe。

## 逐骨统计

variance 为抽查帧骨段长度的 population variance（correction=0），不是 subject betas 的方差。详尽 mean/min/max、逐 clip 均值 std、local offset 误差及精确行号在 JSON。

| Body | neutral 长度 / m | train 最大误差 / μm | train variance / m² | val 最大误差 / μm | val variance / m² |
|---|---:|---:|---:|---:|---:|
| L_Hip | 0.115023691 | 0.123798 | 6.433e-16 | 0.083870 | 4.843e-16 |
| L_Knee | 0.376791431 | 0.163765 | 1.134e-15 | 0.091515 | 8.738e-16 |
| L_Ankle | 0.400622829 | 0.213990 | 1.745e-15 | 0.213435 | 2.076e-15 |
| L_Toe | 0.134324569 | 0.225356 | 1.774e-15 | 0.158016 | 1.308e-15 |
| R_Hip | 0.113101854 | 0.135026 | 6.703e-16 | 0.081006 | 5.173e-16 |
| R_Knee | 0.384615220 | 0.199327 | 1.156e-15 | 0.110291 | 9.975e-16 |
| R_Ankle | 0.400950741 | 0.300530 | 1.805e-15 | 0.233839 | 2.03e-15 |
| R_Toe | 0.134765204 | 0.216219 | 1.589e-15 | 0.156596 | 1.197e-15 |
| Torso | 0.112250343 | 0.136679 | 1.873e-15 | 0.123838 | 1.763e-15 |
| Spine | 0.135316290 | 0.137231 | 2.376e-15 | 0.164822 | 2.435e-15 |
| Chest | 0.058701108 | 0.165298 | 1.422e-15 | 0.170491 | 1.553e-15 |
| Neck | 0.218177588 | 0.195258 | 1.853e-15 | 0.179955 | 2.277e-15 |
| Head | 0.082968245 | 0.199622 | 1.444e-15 | 0.189367 | 1.459e-15 |
| L_Thorax | 0.148940052 | 0.155815 | 1.463e-15 | 0.160285 | 1.647e-15 |
| L_Shoulder | 0.096387031 | 0.224058 | 1.797e-15 | 0.178101 | 1.6e-15 |
| L_Elbow | 0.261366136 | 0.257436 | 3.284e-15 | 0.225936 | 3.374e-15 |
| L_Wrist | 0.249365356 | 0.424488 | 5.191e-15 | 0.340031 | 5.185e-15 |
| L_Hand | 0.085704433 | 0.172671 | 1.425e-15 | 0.125083 | 1.164e-15 |
| R_Thorax | 0.149313897 | 0.148867 | 1.509e-15 | 0.182049 | 1.741e-15 |
| R_Shoulder | 0.101791798 | 0.194811 | 1.933e-15 | 0.193162 | 1.731e-15 |
| R_Elbow | 0.254948101 | 0.212756 | 3.171e-15 | 0.213216 | 3.041e-15 |
| R_Wrist | 0.255480493 | 0.379506 | 5.373e-15 | 0.502579 | 5.08e-15 |
| R_Hand | 0.085442731 | 0.185149 | 1.352e-15 | 0.169948 | 1.019e-15 |

## 数据和复现

源 `.pt` 的 byte size 重新核对，与 manifest 一致；SHA256 引用已存在的 manifest 记录，**未重复读取 9.5 GB train 文件重新计算 hash**。小型 manifest、XML、实现源文件、历史 FK probe、每个 conversion sidecar 的当前 hash 已保存在 JSON。

- train: `/project/EXP/SangforAndSaut/Assets/motion/amass_smpl_train.pt`，9,565,828,107 bytes，manifest SHA256 `a9c9191c4bf369427ac05bf1894df14341d7085ecad20dda758b4da04bd532b5`。
- validation: `/project/EXP/SangforAndSaut/Assets/motion/amass_smpl_validation.pt`，678,009,566 bytes，manifest SHA256 `ae0389770ad4a1dfb6d916b80bbe88b8ff89c7ab80a22d71b77597b6c302cf03`。

CPU 复现命令（从仓库根目录）：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ../IsaacLab-Protomotions/.venv/bin/python docs/diffusion_amp_20261001/review_amass_rig_consistency.py
```

脚本真实执行 exit 0；JSON 状态 `sampled_compatible`。脚本静态 Ruff gate 通过。没有额外宽测试或物理试验。

## 不能由此推出的结论

- 每个 source frame 都一致：本次覆盖每个入选 clip 的三个代表帧及随机帧。
- 动态 PhysX policy body 完全刚性或 contact-free：本次没有读取或改动进行中的 rollout。
- 所有 AMP distribution shortcut 已排除：动作、速度、历史、normalization、reset 与 reward 对齐仍需各自检查。
- 新 corrected checkpoint 达论文结果：应由其真实 held-out physical evaluation 判定；较早的 historical 1000-update rollout 存在跌倒，不能用本次资源一致性替代稳定性结果。
- 作者 2,848 clip 实验已精确复现：当前 manifest 为公开 AMASS 派生的 319/22 入选列表，作者原始 IDs 仍未公开。

[完整机器证据](amass_rig_consistency_review.json) · [只读 CPU 复核脚本](review_amass_rig_consistency.py)
