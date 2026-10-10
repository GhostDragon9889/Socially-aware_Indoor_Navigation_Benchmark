# 独立课程 AMP1000：真实四行人200秒对照

## 同暴露结果

CPU读取两份6000帧×4人、30Hz真实物理记录，每组800行人秒。保持同Diffusion5000、clean guidance20、peer5/radius.4、signed-AABB10/radius.5、endpoint、FP32、DDPM100、seed7与物理manifest。**控制器和route到达/reset处理同时变化，不能单因归因。**

| 指标 | previous clean_sdf_5000_v2 | independent_amp_raw_v1 |
|---|---:|---:|
| 跌倒终止 | 11 | 3 |
| 每1000行人秒跌倒 | 13.75 | 3.75 |
| 排除reset的总XY路程m | 350.954 | 355.604 |
| 平均XY速度m/s | 0.438966 | 0.444635 |
| Tracking mean m | 0.080121 | 0.0941252 |
| Tracking RMS m | 0.105212 | 0.126519 |
| Tracking p95 m | 0.205204 | 0.261457 |
| Gaussian MMD²，σ1 | 0.055818358 | 0.099428336 |
| K50 coverage/归一化entropy | 0.82/0.682984 | 0.54/0.504971 |
| K100 coverage/归一化entropy | 0.68/0.652375 | 0.41/0.534021 |
| Static actor-frames/24000 | 311 | 1680 |
| Static连续actor事件 | 29 | 23 |
| Peer unordered pair-frames/36000 | 588 | 1 |
| Peer连续pair事件 | 8 | 1 |
| A*失败查询 | 23 | 135 |
| plan endpoint到condition goal均差m | 0.801155 | 0.658264 |

每人跌倒：[2, 5, 0, 4] → [1, 1, 0, 1]。总运动350.954→355.604m近似持平。跌倒减少，但static接触显著增加、tracking误差和姿态MMD增大、覆盖/entropy下降，**不能称综合质量或风格已经改善**。两run都没有超时终止。

## 采样和模型证据

train/test各2048姿态、seed7、15节点45D heading/root/height embedding、Gaussian σ1以及2048个time-major生成索引逐项完全相同。K50/100均只在train fit，两套cluster JSON逐字节相同；train/test参考MMD²=0.035753752。没有val/generated重新fit中心，指标只代表本地显式协议，非论文未知超参数的精确值。

共同Diffusion SHA256：`63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`。

旧AMP SHA256：`1d5051cfc36df71b706e4a136458d32539a8084f15c3d1eb9e4dd2b11568213b`；新独立课程AMP SHA256：`e0d6ed4a9acb39c581bed92fab722a426bf23c34a3949bc48712cc13bf04c333`。新低层1000训练仅消费独立smooth课程，公开PACER actor/critic/task与moments warm start + 随机state_action2019 D；旧低层1000训练消费当时的Diffusion命令。新训练完整性见[训练审查](independent_amp_training_review.md)。

两run真实contact合同逐字段相等：normal force>1N，每30Hz控制帧取4个120Hz子步body-pair最大值，排除floor/self，不包含切向摩擦。旧→新static接触比例1.29583%→7.00000%；peer pair比例1.63333%→0.00278%，并非同一种碰撞量。新static per-actor帧数[1267, 361, 51, 1]；最长连续static接触秒数[25.7, 12.033333333333333, 0.6333333333333333, 0.03333333333333333]。

## Route合同变化与完整循环

旧记录累计vertices=[3, 2, 1, 2]，新=[1, 3, 4, 4]；这两个累计计数合同不同。新runner在每个活着的30Hz post-step姿态检查到达，物理reset把route index归0且reset位置不算进度，到达vertex后下一control立即整组强制replan；旧在更稀疏的规划查询中推进与统计，reset行为也不同。新capture保存route_indices/reached，可逐episode验证12次实际到达，全都位于.75m半径内、无fall/reset帧伪计数。新actor2/3各完成1个四顶点完整循环，共2个；actor3之后仍于156.6s跌倒。旧capture缺这两个数组，旧完整循环数为None，不能宣称从0升到2或作者benchmark成功。

最初DDPM噪声形状、batch设置保持；物理状态和额外到达触发replan会改变后续调用/RNG消费，不能说整个200秒轨迹逐位配对。配置文件虽声明timed_path/recovery.5，本run实际CLI endpoint、无start recovery，`reference_adapter=none`，未使用causal reference限速/限加速度。实际physics manifest完全相同；旧arguments未显式写terrain_extent，新写64m，不将缺失字段当物理改变。

## 跌倒时间关系与滞留

3个fall：66.6s actor0，static前.0667s且近2秒49接触帧；102.1667s actor1，该episode没有已记录static/peer；156.6s actor3，当帧static。全部无近2秒peer。只描述时间关联，不推断接触必然导致跌倒。

新actor0在140–200秒每20秒仅走.0783/.0747/.0811m；少跌倒与滞留可以同时出现。新A*失败135次（旧23），goal可按无路径合同返回current root，因此小tracking/condition-goal误差不能替代全局进展。新actor0末60秒static帧为0，不能把所有停滞归为持续墙contact。

| 时段s | 旧/新falls | 旧/新tracking mean m | 新static actor-frames | 新peer pair-frames | 新均速m/s |
|---|---:|---:|---:|---:|---:|
| [0.0, 20.0] | 2/0 | 0.0760/0.1133 | 28 | 1 | 0.7443 |
| [20.0, 40.0] | 1/0 | 0.0886/0.1078 | 492 | 0 | 0.5226 |
| [40.0, 60.0] | 2/0 | 0.0910/0.1050 | 595 | 0 | 0.4008 |
| [60.0, 80.0] | 0/1 | 0.0816/0.0748 | 184 | 0 | 0.6111 |
| [80.0, 100.0] | 1/0 | 0.0677/0.0779 | 0 | 0 | 0.5099 |
| [100.0, 120.0] | 2/1 | 0.0732/0.0833 | 8 | 0 | 0.4699 |
| [120.0, 140.0] | 1/0 | 0.0838/0.0994 | 11 | 0 | 0.2677 |
| [140.0, 160.0] | 1/1 | 0.0744/0.1066 | 1 | 0 | 0.2921 |
| [160.0, 180.0] | 1/0 | 0.0690/0.0754 | 0 | 0 | 0.2910 |
| [180.0, 200.0] | 0/0 | 0.0960/0.0978 | 361 | 0 | 0.3377 |

![同暴露分段对照](independent_amp_physical_comparison.png)

## Provenance与下一对照

新run源归档49文件逐bytes/hash匹配启动manifest；当前树后续改变单列于JSON，不替代启动hash。墙钟旧613.12s、新604.39s，接触/规划/物理耗时不是纯算法性能benchmark。只有单seed，episode删失和route合同变更限制结论。

下一步保持这些权重与真实contact协议，单列timed-path/causal-reference部署消融，继续报告falls、完整同episode循环、A*失败、static滞留、目标跟踪与固定pose指标。不能把当前少跌倒作为风格收敛、社交导航或严格论文复现的验收。

完整fields/hashes/argument changes：[JSON](independent_amp_physical_comparison.json)。新逐帧与contact证据：[phase_quality.md](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_independent_amp_raw_v1/pedestrian_metrics/phase_quality.md)。CPU脚本：[compare_independent_amp_physical_runs.py](compare_independent_amp_physical_runs.py)。
