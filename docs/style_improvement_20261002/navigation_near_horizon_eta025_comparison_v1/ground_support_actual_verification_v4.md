# 实际足部支持采集独立 CPU 验证

输入：`/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_amass2848_near_horizon_eta025_ground_support_200s_v1`

结果：**passed**。`actual_native_capture_validated=true` 指实际采集完整性与数值一致性；历史 probe 的两个 `false` 标记未改。

## 时间、主状态与来源

- 6000 个 30 Hz 控制帧，4 人，200 秒；全部 24000 个 120 Hz 提交步计数连续。每控制恰好四个子步，无补零/丢步。
- 末子步身体位置、Pelvis XYZW、episode/reset/termination 与独立 `physical_states.npz` 逐元素完全相同；其原四个障碍/同行人接触字段保持独立。
- 跌倒 2 次、时间上限 reset 0 次、路线顶点 [6, 8, 4, 2]。不从短时采集推断稳定收敛。
- ground 支持文件 SHA256：`3d84f28f8f8d77dc3b371ea5881a9c6c53c4da33f6a4347e90609a19786a356b`。完整原状态/结果/模型/profile/scene/source 指纹见 JSON。
- 来源边界：结果未包含逐业务源码哈希；已选源码字节与运行前冻结的 CPU 审查清单一致。显式 source snapshot 可保留历史来源而不依赖后续工作树。控制器指纹是本次重算的指定 resume 文件，未伪称结果发布了该指纹。

## 独立重算

- 正质量总和 63.313949 kg；link 世界姿态 × link-local COM 偏置重构真实 COM，最大坐标误差 9.88e-07 m。
- 足部 sensor 与完整 common-body 名称/路径一一对应；唯一 floor filter 为 `/World/ground/terrain/mesh`，不包含墙、同行人或摩擦。
- 从 raw 四子步 normal-force vector sums 独立重算每足/左右侧载荷、严格 >1 N 接触、双支持/无足支持与全部速度有效 mask。
- 双支持占 actor-substeps 48.2312%；无足支持 0.8542%。这些是给定阈值的传感定义，不是作者未公开指标。

## 接触条件下的足部原点运动代理

支持条件样本 214647；切向速度 mean 0.0770、median 0.0281、P95 0.2677、max 6.5575 m/s。

**这不是接触点滑动测量。** 足部滚动/旋转也会产生 link 原点速度；没有摩擦力、接触点速度或 AMASS 真实 COM，不能称作真实 slip，也不能把代理阈值当论文验收。

| actor | 双支持 | 无足支持 |
|---|---:|---:|
| 0 | 40.6000% | 1.0792% |
| 1 | 35.6833% | 0.3667% |
| 2 | 58.5750% | 0.6125% |
| 3 | 58.0667% | 1.3583% |

## 可复现 CPU 命令

```bash
CUDA_VISIBLE_DEVICES='' ../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/navigation_near_horizon_eta025_comparison_v1/audit_eta025_ground_capture.py --output /tmp/eta025_ground_capture.json
```

复用原始完整测量审计，仅以固定manifest和archive SHA允许已知112个公开源码成员。configuration仅四个近段guidance新增节点，剥离后whole AST与旧冻结文件相同；其余足部源和完整控制循环严格一致。旧205/321项失效报告保留；不存在测量检查放宽、runtime/evaluator修改或current-tree全部相同声明。
