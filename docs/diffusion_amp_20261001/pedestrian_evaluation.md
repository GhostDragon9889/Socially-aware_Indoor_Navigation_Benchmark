# 行人评估核心：可复现的本地协议

实现：[evaluation.py](../../CrowdSim/humanoid/evaluation.py)。只依赖 CPU Torch；不会导入 IsaacSim、读取策略权重或启动仿真。

## 与论文的关系

论文 IV-C / Table II–III 说明采用 30Hz、15 个骨节点、root-centered、body-heading aligned、scale-normalized 的姿态表示，以及 AMASS cluster coverage、normalized pose entropy 和 Motion-MMD。持久提取见 [论文文本](../full_audit_20261001/navisaaclab_paper.txt) 第 470–517 行。

论文未公开完整 15 节点 ID、归一化尺度、聚类训练/抽样协议、MMD 核及其带宽。下面均为**显式本地选择**；生成结果不得直接与 Table II/III 的数字作复现一致性判断。

| 参数 | 本地定义 |
| --- | --- |
| 15 节点 | Pelvis、双 Hip/Knee/Ankle/Shoulder/Elbow/Wrist、Neck、Head |
| 节点身份 | 通过完整、无重复的 24 个 SMPL body 名称进行映射；不猜 native PhysX body 排序 |
| 中心与方向 | 减 Pelvis 位置；按 root XYZW 四元数的水平 heading 逆旋转 |
| 默认尺度 | 每帧 Head 到双 Ankle 中点的欧氏距离；这是姿态相关的身高近似，不是作者参数 |
| 可替代尺度 | 显式传入正 `height_m`，并将替代定义写进保存的 `embedding_spec` |
| 表示 | 15×XYZ 展平为 45 维，保留 Pelvis 的三个零坐标；无 whitening 或额外统计归一化 |
| 聚类 | CPU kmeans++，seed=7，最多 200 次迭代，centroid 最大移动 tolerance=1e-7；K=50/100 |
| coverage | 生成样本至少占用一次的固定训练 cluster 数量 / K |
| entropy | `−Σ p_k log p_k / log K`；没有占用的 cluster 不参与 log |
| center distance | 到固定训练中心的平均欧氏距离及平方距离；不声称对应作者 Table III 的未定义 Dist. |
| MMD 核 | Gaussian `exp(−||x−y||²/(2σ²))`，默认 σ=1.0，固定带宽；不在 held-out/generated 数据上优化或估计 σ |
| MMD 估计量 | 包含对角项的 biased V-statistic，输出 MMD² 与 sqrt；分块计算真实两两核均值 |

## 固定训练参考与 held-out

`PoseClusterModel.fit` 必须显式传 `split="train"` 和 `source_id`。validation/test/generated fitting 会失败；不允许把 held-out 再拟合成“参考 cluster”。这个接口检查声明，数据真实身份仍需调用者通过 AMASS manifest、源包 hash 和采样 ID 证明。

同一个模型的 `.score(..., split="test")` 或 `.score(..., split="generated")` 只分配类别，不改变中心。JSON `.save/.load` 包含训练 split/source、样本数量、seed、embedding spec、迭代数与精确 float64 中心。载入伪造为 test-fit 的模型会失败。

```python
from CrowdSim.humanoid.evaluation import (
    pose_embedding, fit_pose_clusters, gaussian_mmd, rollout_metrics,
)

train_features = pose_embedding(train_body_pos, train_root_rot, body_names=body_names).reshape(-1, 45)
heldout_features = pose_embedding(test_body_pos, test_root_rot, body_names=body_names).reshape(-1, 45)
generated_features = pose_embedding(body_pos, root_rot, body_names=body_names).reshape(-1, 45)

models = fit_pose_clusters(train_features, split="train", source_id=training_source_sha256)
coverage = {k: model.score(generated_features) for k, model in models.items()}
reference_coverage = {k: model.score(heldout_features, split="test") for k, model in models.items()}
mmd = gaussian_mmd(generated_features, heldout_features, bandwidth=1.0)
```

MMD 使用所有传入样本；若为控制二次计算开销而在调用前抽样，必须保存样本数量、固定 seed/索引和 train/test/generated 的取样协议。生成视频长短及重复样本会影响 coverage/entropy 与 biased MMD，不能隐去这些条件。

## Rollout 时间、重置与碰撞

`rollout_metrics` 接收 CPU `root_pos[T,B,3]` 与 unit XYZW `root_rot[T,B,4]`，默认 dt=1/30。目标 `target_xy[T,B,2]` 必须已经与同一物理时刻对齐；指标计算真实 XY error，不以最近未来轨迹点代替。

重置约定必须明确：

- `reset_mask_marks="before_frame"`：mask[t] 表示记录状态 t 属于新 episode，跳跃 t−1→t 排除。
- `reset_mask_marks="after_frame"`：mask[t] 表示记录状态 t 后终止并重置，跳跃 t→t+1 排除。
- `episode_ids[T,B]` 是独立的第二个边界来源。输出同时报告两个来源的计数及 disagreement；不一致时保守排除两者标识的跳跃，并显式显示不一致。
- 路径距离是相邻有效、同 episode 状态的 XY 距离和。缺失帧两侧不连接；不会把 reset teleport 加入 walked distance。
- `(T−1)dt` 是记录首末样本时间跨度，不等于未记录初始状态时的总物理控制步长。

默认 fall 定义与当前本地 AMPWorld 一致：Pelvis 距地面高度 `<0.45m` 或 root up-z `<0.1`。输出包括 flagged frame 数、同 episode 连续 flag 的事件起点数，以及受影响的已观测 episode 比例；这些 episode 可以是开头/结尾截断的，不自动转换成论文导航 success rate。

碰撞优先传入实际 `map_collision_mask[T,B]` / `agent_collision_mask[T,B]` 和其定义。没有碰撞证据时输出 `None`，不会把未知写成零。

两个可选 CPU 几何代理需明确标记：

- `planar_agent_overlap` 采用给定半径的 XY 圆盘重叠，仅适用于同一 shared world；不是 PhysX contact。隔离并行环境不能互相计碰撞。
- `occupancy_collision_mask` 检查 root 点所在 bool 栅格（row 向世界 +Y、column 向 +X、origin 为左下 cell corner）；不是全身/足迹碰撞。灰度 PNG 必须由调用者明确转换 free/occupied/unknown，并核对纵轴。越界默认计 collision，可显式更改。

## 真实验证

CPU 回归：[tests](../../tests/test_paper_pedestrian_evaluation.py)，JUnit [pedestrian_evaluation_tests.xml](pedestrian_evaluation_tests.xml)。覆盖平移/yaw/尺度不变性、body 名称排列、固定 cluster train/held-out gate 和 JSON 无精度损失续用、K50/100、真实 Gaussian 核解析值/identical MMD≈0、重置前/后两种语义、episode ID、碰撞事件、时间对齐和 missing-frame 距离。

另已运行真实本地 AMASS mmap：train/test 各 512 个实际参考姿态，seed=313/991，train-only K50/100 拟合与 test assignment、Gaussian MMD 前向均通过。精确采样 motion/frame ID、来源 manifest/hash 与结果见 [evaluation_amass_cpu.json](evaluation_amass_cpu.json)。这是数据与 API 兼容验证，没有生成物理 rollout，不能充当最终四人 200 秒评估或作者结果。

最终实际 rollout 的 coverage、MMD、fall/碰撞/跟踪结果由评估 runner 在训练完成后生成；本文不预先声明其通过、收敛或与论文一致。
