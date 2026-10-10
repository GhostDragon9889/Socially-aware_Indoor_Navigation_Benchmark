# 全量人体前向轴与重置姿态核对

**实际CPU退出0；已完成观察，训练失败原因仍未定位。**

## 依据与范围

NavIsaacLab [IV-B／式6](https://arxiv.org/html/2606.26265v1#S4.SS2)使用自体坐标的前向根部速度；
当前官方 [ProtoMotions steering reward](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/envs/rewards/task.py)与
[calc_heading](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/utils/rotations.py)采用旋转后的根部+X。
[PACER轨迹观测](https://github.com/nv-tlabs/pacer/blob/main/pacer/env/tasks/humanoid_traj.py)使用逆heading旋转世界目标位移。
这些公开实现支持检查局部轴约定，没有证明本地模型质量或作者私有协议一致。

[完整结果](actual/result.json)核对固定v7训练bank全部2848条、523272帧，并逐动作验证ID、原始路径、帧数、连续帧分区和30Hz。
S1／S2终态各22条、11982评价帧全部保留，原世界坐标评分逐值核对；另保留每条重置帧及第一POST帧误差。
75项实际输入与8项实现来源在执行前后核对。bank没有存储身体名字字段；其顺序沿用原语料转换器的COMMON24声明，
实际捕获还验证了物理报告的完整身体名字顺序。

诊断测量 `L_Hip-R_Hip` 在自身根部坐标系的方向，并用该横向轴叉乘实际根部+Z构造几何前向。
不使用速度定义前向，侧步和倒走动作保留。四元数只在诊断计算中归一化，原始捕获与指标保持；
XY投影不存在时显式标记，保留该帧并将heading统计计数单独报告。本批所有投影均有定义。

## 实际观察

| 完整人口 | 帧数 | 髋横向对局部+Y平均夹角 | 几何前向对根部+X的投影平均绝对差 |
| --- | --- | --- | --- |
| 固定train2848 | 523272 | 1.109473° | 0.018293 rad |
| S1真实全身 | 11982 | 1.109473° | 0.016808 rad |
| S2真实全身 | 11982 | 1.109474° | 0.018612 rad |
| 独立validation参考 | 11982 | 1.109471° | 0.018239 rad |

本批没有观察到固定90°前向轴偏移或左右横向轴反转。
两模型的22个重置帧，全身位置平均误差的最大值为1.446798e-6m、旋转平均误差最大值1.120105e-6rad。
第一POST帧位置误差最大S1／S2为.033087／.031185m，旋转误差最大.192614／.163063rad；
原始完整跟踪仍0/22，S2位置／旋转仍.918030m／1.412125rad。
跌倒时投影角可以放大：S2最大绝对差1.382409rad、最小根部+X水平投影范数.012068；这些帧保留。
上述数据没有证明控制、动作映射或训练分布均正确，也没有证明改善模型效果。

## 复现与验证

入口：[audit_shared_social_heading_geometry.py](../../../scripts/audit_shared_social_heading_geometry.py)。
选择新的输出目录，并使用已登记Python环境、`CUDA_VISIBLE_DEVICES=''`运行。
几何实现：[heading_geometry.py](../../../CrowdSim/humanoid/analysis/heading_geometry.py)。
实际工具会话64029／d2bbee退出0；没有启动SDK或GPU训练。

34项相关检查（新12／原误差分解22）与Ruff通过。叉乘顺序反转触发5项失败、29项通过，精确恢复后34项全通过。
[工具记录](tool_execution.json)与[突变记录](mutation_receipt.json)保留实际退出；初次Ruff的12项单行格式问题已修正。
下一步继续检查训练任务分布和控制学习，并在S3完整退出及独立评价后实际验证future81候选。
本批不改变S3训练配方、固定2848清单、指标或部署默认模型。
