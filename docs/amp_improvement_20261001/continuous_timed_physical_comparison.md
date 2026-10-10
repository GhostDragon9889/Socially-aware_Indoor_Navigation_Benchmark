# Timed + causal reference：真实四行人200秒对照

## 实测效果与边界

两份6000帧×4人、30Hz真实记录各800行人秒；相同独立课程AMP1000和Diffusion5000 checkpoint、物理与真实contact合同。新run **0次跌倒**、pelvis最低0.633056m，26次vertex到达、5个同episode完整四顶点循环；不是八场景benchmark，也不表示pose收敛。

| 指标 | raw endpoint | timed + causal reference |
|---|---:|---:|
| 跌倒 | 3 | 0 |
| 每1000行人秒跌倒 | 3.75 | 0 |
| XY路程m（排除reset） | 355.604 | 529.602 |
| 平均XY速度m/s | 0.444635 | 0.662113 |
| 最低pelvis m | 0.410904 | 0.633056 |
| 同episode完整四顶点循环 | 2 | 5 |
| 实际到达vertices | 12 | 26 |
| Gaussian MMD²，σ1 | 0.099428336 | 0.092151109 |
| K50 coverage/归一化entropy | 0.54/0.504971 | 0.58/0.662422 |
| K100 coverage/归一化entropy | 0.41/0.534021 | 0.47/0.665912 |
| Static actor-frames/24000 | 1680 | 923 |
| Static连续actor事件 | 23 | 90 |
| Peer pair-frames/36000 | 1 | 0 |
| A*失败查询 | 135 | 83 |

新每人vertices=[8, 6, 6, 6]、完整cycles=[2, 1, 1, 1]。26个记录到达点全部匹配声明route，最大root到vertex距离0.748502m≤.75m，无fall或首个reset帧计数。所有四人200秒episode均未跌倒/超时、结束右删失；不是无限时稳定性证明。Static actor-frame比例7%→3.84583%，新仍有90个连续actor接触事件，不能说无碰撞。新peer正常力阈值事件为0。训练与部署权重未修改。

## 原始模型目标、参考和物理root分别报告

旧raw run root→raw目标mean/RMS=0.094125/0.126519m。新 **root→raw模型目标** mean/RMS/p95/max=0.115926/0.173139/0.402528/1.601189m；新 **root→执行reference**=0.202027/0.296748/0.660659/1.299681m。evaluator中的`plan_tracking`在新run指执行reference，不能拿它直接当旧raw目标指标。

执行reference与raw模型目标偏差mean/p95/max=0.125819/0.439691/1.272740m。适配保留原51个10Hz模型knots，生成151个30Hz参考knots；普通重规划从已执行参考的XY/速度延续，仅episode初始化/reset从物理root建立起点，不将物理root搬到参考位置。

## 参考约束与brake标记的CPU复核

保存1704份actor-plan、426次四人联合规划；全部future reference样本分母257304，含未执行future。显式`initial_speed_policy=brake`，**initial_overspeed=0、speed_violation mask=0**；本run没有触发初始超速制动，不将“配置brake”写成实际执行了制动。mask按保存的速度/1.5m/s+1e-6 tolerance逐项重算完全相等。

计划reference max speed=1.500000276m/s、max vector acceleration=2.000002593m/s²（float32误差，声明cap1.5/2）；执行reference同样复核。计划/执行积分残差最大1.35e-06/1.31e-06m；保存deviation重算通过。

**参考cap不等于物理body bound。**实际root的30Hz位移差分速度max=2.200383m/s，60/23996合法区间高于参考1.5m/s；二阶位置差分acceleration max=45.934766m/s²。它们是控制区间离散root指标，不是SDK瞬时速度、子步acceleration或所有body约束。适配也没有障碍/接触/路线到达保证。

## 同时变化的部署参数

保持weight20总guidance，endpoint改为同一次静态A*路径的10个0.5..5秒点，每点weight2；start recovery显式0.5m。恢复仅在实际root有效但clearance cell被占用时连接近邻可达起点，不teleport，初始连接段clearance未认证；successful recovery=149次查询，不是不同episode数或物理reset。peer radius.4→.6、scene radius.5→.8；新causal reference速度1.5/向量加速度2。**多项同时改变，不能把效果单独归给timed或filter。**两run的route到达/reset/forced-replan合同一致。

DDPM100、FP32、replan15、batch8、seed7与XY消费保持；执行状态和额外route触发改变后续随机调用，不是逐位配对轨迹。速度/加速度guidance penalties仍为0，真实约束只对新增reference声明。

## 固定姿态协议与provenance

train/test各2048、generated2048 time-major索引、seed7、15节点45D root/heading/height规范化、σ1以及全部参考IDs相同。K50/100 train-only cluster文件逐字节相同。MMD从.09943到.09215，coverage/entropy有所变化，仍不能判定风格通过；该协议未衡量temporal gait，论文节点/尺度/带宽未知，不声称exact paper。

AMP SHA256 `e0d6ed4a9acb39c581bed92fab722a426bf23c34a3949bc48712cc13bf04c333`；Diffusion SHA256 `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`。启动源归档49文件逐bytes/hash匹配manifest；后续CLI sample-count变化单列，不覆盖原源hash。这次200秒不含后续2-sample演示。墙钟raw604.39s、新627.96s，不能当純planner benchmark。

![分段比较](continuous_timed_physical_comparison.png)

完整tracking/reference masks/几何/route/collision与source证据：[JSON](continuous_timed_physical_comparison.json)。[新逐帧phase](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_independent_amp_continuous_timed_v1/pedestrian_metrics/phase_quality.md)。[可重跑CPU脚本](compare_continuous_timed_physical_runs.py)。本轮未修改训练、部署业务代码或权重。
