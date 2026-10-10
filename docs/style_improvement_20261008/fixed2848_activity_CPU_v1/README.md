# 固定2848 AMASS：完整速度与关节运动统计

CPU实际退出0；全部2848片段、523272源帧及494792合法专家中心保留。
这是一份数据分布诊断，不是候选模型的独立风格或物理评价。
来源、帧数、连续边界、30Hz时序及manifest/bank哈希在执行前后核对。

## 结果

专家中心为每条片段的第9帧至倒数第2帧，包含当前及过去共10帧、下一帧动作目标。
下表同时报告理想的“均匀片段→均匀合法中心”先验，以及所有合法中心等权的统计。
两种权重不同；实际训练曝光仍依赖具体抽样、物理episode与重置。

| 本地描述性条件 | 均匀片段→中心 | 全中心等权 |
| --- | ---: | ---: |
| 当前根部XY速度 <0.1 m/s | 22.4277% | 20.8158% |
| 当前57维关节速度分量RMS <0.5 rad/s | 48.4763% | 47.0642% |
| 当前同时满足上述两项 | 21.3663% | 19.6143% |
| 当前再满足根部角速度范数 <0.3 rad/s | 19.0510% | 17.1256% |
| 连续10帧均满足上述三项 | 14.0034% | 12.5499% |

片段等权的专家中心平均根部速度为0.588677 m/s、关节分量RMS为0.567040 rad/s。
这些阈值是本地诊断选择，不是作者协议或“站立”标签。
低根部速度不保证关节静止；十帧低运动也不证明动作语义、物理支撑或判别器偏好成因。
数据中存在低运动窗口，后续优化应保留合理停顿，避免把所有静止姿势统一当成负样本。

## 核验与复现

- [完整结果](result.json)：两种权重、完整二维直方图、当前／十帧条件概率及输入来源。
- [每片段统计](per_clip_statistics.npz)：2848个motion ID、各片段分母、直方图、均值与条件计数。
- [实际CPU退出与检查记录](actual_CPU_execution.json)：5项测试／3个subtest、Ruff及保存数组的逐值复算均退出0。
- [训练宿主观察](training_host_observation.json)：05:21:12训练605/2000，来源57项／输入7项保持，训练未重启。

统计使用float64计算存储float32速度的范数。所有19个关节三维速度完整保留，范数具有旋转不变性。
实际bank和manifest都明确声明`child_joint`；当前corpus返回同一坐标系，**没有执行父系→子系旋转**。
完整523272帧逐值核对未改变范数、阈值分类或关节直方图桶。
均匀抽样先验抽象了`torch.rand`的有限精度，不声称复原实际抽样轨迹。
当前训练的配置、资源、模型选择及GPU进程保持；全部分析在独立CPU进程完成。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/diagnose_fixed_amass_activity.py --output /tmp/fixed2848_activity_new
```

输出目录必须尚不存在；复现会核对固定manifest/bank及原专家抽样／特征代码的来源。
实现见`source_snapshots/`，身份见[source_snapshot_index.json](source_snapshot_index.json)。
