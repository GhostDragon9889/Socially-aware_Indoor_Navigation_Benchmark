# 混合控制器导航独立 CPU 审查

结果：passed；检查2259项。

# 实际足部支持采集独立 CPU 验证

输入：`/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_mixed_curriculum_step2000_ground_support_200s_v1`

结果：**passed**。`actual_native_capture_validated=true` 指实际采集完整性与数值一致性；历史 probe 的两个 `false` 标记未改。

## 时间、主状态与来源

- 6000 个 30 Hz 控制帧，4 人，200 秒；全部 24000 个 120 Hz 提交步计数连续。每控制恰好四个子步，无补零/丢步。
- 末子步身体位置、Pelvis XYZW、episode/reset/termination 与独立 `physical_states.npz` 逐元素完全相同；其原四个障碍/同行人接触字段保持独立。
- 跌倒 11 次、时间上限 reset 0 次、路线顶点 [5, 4, 4, 6]。不从短时采集推断稳定收敛。
- ground 支持文件 SHA256：`e60c32c582eab376941747fc643b40c40e7017f457793ff308ce135ef6d599f8`。完整原状态/结果/模型/profile/scene/source 指纹见 JSON。
- 来源边界：结果未包含逐业务源码哈希；已选源码字节与运行前冻结的 CPU 审查清单一致。显式 source snapshot 可保留历史来源而不依赖后续工作树。控制器指纹是本次重算的指定 resume 文件，未伪称结果发布了该指纹。

## 独立重算

- 正质量总和 63.313949 kg；link 世界姿态 × link-local COM 偏置重构真实 COM，最大坐标误差 1.01e-06 m。
- 足部 sensor 与完整 common-body 名称/路径一一对应；唯一 floor filter 为 `/World/ground/terrain/mesh`，不包含墙、同行人或摩擦。
- 从 raw 四子步 normal-force vector sums 独立重算每足/左右侧载荷、严格 >1 N 接触、双支持/无足支持与全部速度有效 mask。
- 双支持占 actor-substeps 39.3094%；无足支持 1.0344%。这些是给定阈值的传感定义，不是作者未公开指标。

## 接触条件下的足部原点运动代理

支持条件样本 187352；切向速度 mean 0.0988、median 0.0470、P95 0.3217、max 6.7172 m/s。

**这不是接触点滑动测量。** 足部滚动/旋转也会产生 link 原点速度；没有摩擦力、接触点速度或 AMASS 真实 COM，不能称作真实 slip，也不能把代理阈值当论文验收。

| actor | 双支持 | 无足支持 |
|---|---:|---:|
| 0 | 44.2875% | 0.8542% |
| 1 | 41.5125% | 0.8375% |
| 2 | 35.1042% | 1.4958% |
| 3 | 36.3333% | 0.9500% |

## 可复现 CPU 命令

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. /project/EXP/SangforAndSaut/ProtoMotions/../IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/audit_mixed_navigation_ground.py --candidate-dir /project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_mixed_curriculum_step2000_ground_support_200s_v1 --candidate-step 2000 --registered-preparation /project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/mixed_curriculum_step2000_ground_support_200s_preparation_v1.json --source-manifest /project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/mixed_curriculum_navigation_source_v1.json --checkpoint-admission /project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/mixed_curriculum_terminal500_1000_2000_independent_audit_v1.json --output /tmp/mixed_navigation_ground_reaudit.json
```

完整6000控制帧／24000actor帧／96000actor子步全部保留；实际raw/reference计划、重规划、路线、每次跌倒及reset事件见JSON。状态passed是采集与证据完整性，不是质量验收。
