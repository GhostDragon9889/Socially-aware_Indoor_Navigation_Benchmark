# AMASS 导航动作整理与原生关节速度修正

冻结日期：2026-10-02。推荐资源为 **manifest_labeled_v7.json**。本次仅整理本地授权 AMASS 数据、增加整理工具和 CPU 测试；未启动 Kit/GPU。

## 最终资源

| 项目 | 实际结果 |
| --- | ---: |
| 训练片段 | 2,848 |
| 原始父试次 | 1,853 |
| 含多个片段的父试次 | 644 |
| 每个父试次最多片段数 | 4 |
| 数据集/subject 目录命名空间 | 200 |
| CMU 官方试次说明支持的片段 | 699 |
| 导航动作文件名支持的片段 | 2,149 |
| 未知数字编号动作推断片段 | **0** |
| 真正从 50 Hz 重采样到 30 Hz | 1,349 |
| 可取完整 Diffusion 上下文的片段 | 1,131 |
| AMP 专用短片段 | 1,717 |
| 验证/测试片段 | 22 / 34，原清单保持一致 |
| 总时长 | 17,347.47 秒，约 4.82 小时 |

- 清单：[manifest_labeled_v7.json](../../data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json)
- 清单 SHA256：`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`
- 动作库：[manifest_labeled_v7_train.pt](../../data/diffusion_amp/amass_style_2848/manifest_labeled_v7_train.pt)
- 动作库大小：`1,155,817,682` 字节
- 动作库 SHA256：`46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`

上述 2,848 是从真实 AMASS 时间区间产生的互不重叠片段，来自 1,853 个原始试次。它们没有重复填充、跨试次拼接或伪造采样率。数量一致不代表作者的动作编号或分段方式一致。

## 与论文需求的关系

论文 IV-B 将行走、慢跑和环顾列为 AMASS 参考动作；IV-C 说明收集 2,848 份动作，并根据标签筛选导航相关的行走、跑步、转向、侧步和过渡动作。作者编号、实际分段方式、质量阈值和数据划分未在本文规范中给出。因此本清单标记 `author_clip_list: false`，属于可核查的方法重建。[NavIsaacLab 论文](https://arxiv.org/html/2606.26265v1)

AMASS 将多个动作捕捉数据集统一到共同的人体参数化，但动作类别仍需标签证据。[AMASS 官方说明](https://amass.is.tue.mpg.de/) 本地未发现 BABEL 的逐帧动作标注；BABEL 提供序列和逐帧语义描述，不能把数字编号或运动学特征等同于这种标注。[BABEL 官方说明](https://babel.is.tue.mpg.de/)

CMU 的数字编号现在通过实际 basename 与官方试次说明关联。混合跳跃、拳击、投掷、举物等描述被排除；空白说明和缺失编号没有用 subject 摘要补齐。证据来自 [CMU 官方动作索引](http://mocap.cs.cmu.edu/search.php)，完整 join 和原始页面哈希见 [CMU 元数据](../style_improvement_20261002/cmu_motion_labels.json)。

## 整理步骤

1. 读取三份既有 tensor-only 动作库，使用 `weights_only=True, mmap=True`，检查当前文件大小；原始大型资源 SHA 使用既有验证记录，没有重复散列 9.5 GB 训练库。
2. 保留原训练/验证/测试的来源数据集划分。全部原始来源具有数据集和 subject 命名空间的独立性；跨数据集是否同一个自然人、公开 PACER 预训练是否看过这些动作仍未知。
3. 描述性文件名必须匹配导航/站立/环顾关键词；CMU 官方说明优先于数字文件名。推荐配置拒绝无说明的数字动作。
4. 对真实 50 Hz 源执行根位置插值和局部四元数最短弧 SLERP，以当前 neutral SMPL 骨架重算 FK 和 30 Hz 速度。DOF 指数映射来自插值旋转，不在线性插值跨越 ±π 的指数映射数值。
5. 优先保留至少 250 帧、8.3 秒的连续窗口供 Diffusion。较短源和剩余时间可提供至少 91 帧、3 秒的 AMP 片段。每个父源最多 4 个片段，连续区间和原帧依赖范围均不重叠。
6. 每一帧检查有限值、四元数、躯干直立比例、速度/加速度分位数等；移动片段必须有实际根部路径和位移，局部手臂动作引起的根部晃动不作为导航位移。
7. 对完整片段消去初始 XY/yaw 和四元数符号，再做量化指纹去重；按标签证据优先、样式/数据集/subject 稳定轮转选择。该方法不能发现所有近似重复或时间裁切重复。

每条清单记录保留 `original_motion_id`、`parent_source_path`、原源帧范围、30 Hz 半开帧范围和时间范围。新库的唯一标识为 `parent_source_path#frames=start:stop`。留出集交叉检查必须比较父来源，避免分段后缀掩盖来源交集。

## 必须采用 v7 的原因

当前 ProtoMotions 的 AMASS 转换公式是 `q_next * inverse(q_current)`，表示父关节坐标系角速度。真实 Isaac Lab D6 同一时刻的测试表明，原生 57 个有效关节速度与 **子关节坐标系** 匹配；其公式为 `inverse(q_current) * q_next`，与公开 PACER 一致。它不是指数映射三分量的直接时间导数。实际探针与原始代码证据见 [速度语义审查](../style_improvement_20261002/dof_velocity_semantics_audit.md)。

v7 保留 v6 的所有选中编号、顺序、姿态、片段范围、全局 `gvs/gavs`、接触代理和时间数据，仅转换 `dvs` 并添加显式坐标系与校验元数据。清单 train entry、bank 和 selection 均声明 `dof_velocity_frame: child_joint`。每段最后一帧沿用倒数第二帧角速度。原始三份 AMASS 库、验证/测试资源和 v6 文件没有修改。

可使用现有接口：

```python
AMASSCorpus(
    "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json",
    split="train",
    device="cpu",
    dof_velocity_frame="child_joint",
)
```

对应 loader 的显式声明检查避免重复转换；元数据与 payload 不一致会失败。

未来直接生成资源的 writer 也已补齐 bank/entry/每条记录的速度坐标声明。该修补防止直接采用 `child_joint` 配置重新整理时，正确 payload 被未声明的旧数据规则再次转换。现有 v7 文件已经具有这些声明，其数据、清单及哈希没有变化；历史训练源码归档保持原快照。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| v6→v7 非 `dvs` 字段 | 12 个 tensor 与 `motion_files` 全部逐元素一致 |
| 选中编号、父源、姿态、顺序 | 2,848 条保持一致 |
| 留出集 | 22/34 条及 frozen reference 来源检查通过 |
| 段边界 | 每父源 ≤4，30 Hz 区间和原始帧依赖不重叠 |
| 当前 neutral FK | 8,544 个首/中/末姿态，最大位置误差 `1.67537e-6 m` |
| 独立 SciPy 子关节角速度 | 8,544 组姿态，RMS `2.67110e-6 rad/s`，最大 `1.46737e-5 rad/s` |
| dtype / 尾帧 | float32；每段尾帧精确 carry |
| 动作库 SHA | 重新散列，与清单及目标元数据一致 |
| 实际 corpus 小批量 | 64 expert、640 history、8 diffusion 读取通过 |
| CPU 回归 | 20 passed，无失败/跳过 |
| Ruff | passed |

证据：[verification_v7.json](verification_v7.json)、[验证脚本](verify_curated_bank.py)、[测试日志](tests.log)、[JUnit](tests.xml)、[独立留出集/元数据检查](../style_improvement_20261002/v7_frame_metadata_verification.json)。本检查没有启动仿真，不等同于控制器收敛或论文物理指标验证。

## 数据偏差与后续评估

运动学类别为：转向样式 1,568、行走样式 720、站立/环顾样式 427、侧步样式 116、跑步样式 17。这些类别来自公开阈值，不是作者逐帧标注。KIT 占 1,233/2,848，源分布与跑步类别均不均衡。644 个父试次提供多个片段，片段仍然相关，不能作为独立试次或独立人的数量。

速度质量门限采用每片段分量 P99 ≤30 rad/s；v7 最大实际 P99 为 8.89408。该门限不是所有瞬时速度的上限：57 个有效自由度中仍有 27 个分量超过 40 rad/s，涉及 17 个片段，最大 69.8322 rad/s，主要为腕关节。具体来源和帧见 [异常速度记录](angular_velocity_outliers.json)。这些数值没有被截断或静默改写。

接触标签和近地面足部速度只作为运动学代理，不能保证没有打滑或动力学失稳。真实训练需在冻结留出集上继续报告闭环姿态误差、失败率、运动风格分布及真实接触。资源数量和坐标系正确，本身不能证明达到论文效果。

## 工具与历史版本

- [整理模块](../../CrowdSim/humanoid/training/amass_curation.py)
- [整理入口](../../scripts/curate_amass_pedestrian_data.py)
- [标签优先配置](../../configs/diffusion_amp/amass_curation_labeled.yaml)
- [显式速度转换入口](../../scripts/convert_amass_joint_velocities.py)
- [最终机器报告](report.json)
- [v6 源码快照](v6_source_snapshot/amass_curation.py)

v1–v3 是逐步审查的中间候选，包含未知动作推断，不能作为推荐训练资源。v4 的标签片段为 2,561；v5 加入合法中等长度片段后为 2,799；v6 恢复未使用的连续尾部片段后达到 2,848。v7 由实际原生速度证据触发，仅修正子关节速度契约。全部历史文件保留，新资源没有覆盖既有 AMASS 或权重。
