# 实际足部支持采集独立 CPU 验证

输入：`/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_amass2848_backward_penalty_ground_support_pilot_v1`

结果：**passed**。`actual_native_capture_validated=true` 指实际采集完整性与数值一致性；历史 probe 的两个 `false` 标记未改。

## 时间、主状态与来源

- 900 个 30 Hz 控制帧，4 人，30 秒；全部 3600 个 120 Hz 提交步计数连续。每控制恰好四个子步，无补零/丢步。
- 末子步身体位置、Pelvis XYZW、episode/reset/termination 与独立 `physical_states.npz` 逐元素完全相同；其原四个障碍/同行人接触字段保持独立。
- 跌倒 0 次、时间上限 reset 0 次、路线顶点 [1, 0, 1, 0]。不从短时采集推断稳定收敛。
- ground 支持文件 SHA256：`2f23328c2155da699566c6516c9c034c2500896090f5e74c64d8fbf86c357c39`。完整原状态/结果/模型/profile/scene/source 指纹见 JSON。
- 来源边界：结果未包含逐业务源码哈希；当前源码与运行前冻结的 CPU 审查清单一致。控制器指纹是本次重算的指定 resume 文件，未伪称结果发布了该指纹。

## 独立重算

- 正质量总和 63.313949 kg；link 世界姿态 × link-local COM 偏置重构真实 COM，最大坐标误差 9.94e-07 m。
- 足部 sensor 与完整 common-body 名称/路径一一对应；唯一 floor filter 为 `/World/ground/terrain/mesh`，不包含墙、同行人或摩擦。
- 从 raw 四子步 normal-force vector sums 独立重算每足/左右侧载荷、严格 >1 N 接触、双支持/无足支持与全部速度有效 mask。
- 双支持占 actor-substeps 32.4375%；无足支持 1.1111%。这些是给定阈值的传感定义，不是作者未公开指标。

## 接触条件下的足部原点运动代理

支持条件样本 28694；切向速度 mean 0.0994、median 0.0348、P95 0.3635、max 4.1771 m/s。

**这不是接触点滑动测量。** 足部滚动/旋转也会产生 link 原点速度；没有摩擦力、接触点速度或 AMASS 真实 COM，不能称作真实 slip，也不能把代理阈值当论文验收。

| actor | 双支持 | 无足支持 |
|---|---:|---:|
| 0 | 23.5833% | 0.6667% |
| 1 | 29.0556% | 0.6944% |
| 2 | 22.7500% | 0.3056% |
| 3 | 54.3611% | 2.7778% |

## 可复现 CPU 命令

```bash
CUDA_VISIBLE_DEVICES='' ../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/audit_ground_support_capture.py --input /project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_amass2848_backward_penalty_ground_support_pilot_v1 --output /tmp/ground_support_verification.json
```

本检查仅读取现有文件并写新报告，不启动 Kit/GPU，不修改历史结果、业务源码或采集 schema。
