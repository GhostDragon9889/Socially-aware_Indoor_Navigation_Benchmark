# Diff5000 clean guidance：真实四行人200秒对照

CPU读取并验证两份6000帧×4人记录；保留历史baseline不重写。AMP控制器相同，planner和运行参数有多项变化，不能单因归因。

## 相同暴露的结果

| 指标 | baseline corrected_endpoint_1000 | clean_sdf_5000_v2 |
| --- | --- | --- |
| terminal falls | 16 | 11 |
| 每1000 actor-seconds跌倒 | 20 | 13.75 |
| reset不计入的总XY路程(m) | 537.695 | 350.954 |
| 平均XY速度(m/s) | 0.672679 | 0.438966 |
| tracking mean(m) | 0.215236 | 0.080121 |
| tracking RMS(m) | 0.262486 | 0.105212 |
| tracking p95(m) | 0.484562 | 0.205204 |
| 路线顶点累计到达 | 0 | 8 |
| A*失败查询 | 68 | 23 |
| plan endpoint到condition goal mean(m) | 5.30737 | 0.801155 |
| Gaussian MMD² / σ1 | 0.109347225 | 0.055818358 |
| K50 coverage | 0.84 | 0.82 |
| K50 normalized entropy | 0.713463 | 0.682984 |
| K100 coverage | 0.67 | 0.68 |
| K100 normalized entropy | 0.654700 | 0.652375 |

每人falls：旧[8, 2, 2, 4]，新[2, 5, 0, 4]。新route vertices=[3, 2, 1, 2]；完整路线成功率未知。

姿态参考：train/test各2048、seed7、显式15节点45D／root heading+height规范化、σ1 biased Gaussian MMD；生成2048个同一time-major索引。两份train-only K50/100文件逐字节完全相同，AMASS样本IDs、fit和embedding协议完全相同。train/test参考MMD²=.035753752。K50覆盖下降、K100略升，不能称所有姿态指标统一改善。

## 新run真实法向接触

| 接触统计 | 接触帧／全部观测帧 | 比例 | 连续事件 | 最大单body-pair力(N) |
| --- | --- | --- | --- | --- |
| Static actor-frames | 311/24000 | 1.29583% | 29 actor incidences | 2910.75 |
| Peer unordered pair-frames | 588/36000 | 1.63333% | 8 pair incidences | 1950.63 |
| Peer actor-frames | 1176/24000 | 4.90000% | 16 actor incidences | 同上peer力 |

严格normal force>1N，控制帧聚合全部4物理子步最大body-pair幅度，排除floor/self；不包含摩擦切向力或全身总冲量。旧run没有这些数组，所以历史actual collision保持未知；不能据新数据断言碰撞改善。

11个terminal当帧static=2、peer=0；同episode前2s有static7／peer2，合并8个。其余3个近2s无已记录接触。这里只提供时间关系。

## 分段与残留问题

| 时段(s) | 旧/新falls | 旧/新tracking mean(m) | 新static actor-frames | 新peer pair-frames | 新速度(m/s) |
| --- | --- | --- | --- | --- | --- |
| [0.0, 20.0] | 1/2 | 0.1727/0.0760 | 22 | 0 | 0.6235 |
| [20.0, 40.0] | 4/1 | 0.1946/0.0886 | 0 | 156 | 0.4285 |
| [40.0, 60.0] | 5/2 | 0.1675/0.0910 | 62 | 89 | 0.5088 |
| [60.0, 80.0] | 0/0 | 0.1931/0.0816 | 71 | 0 | 0.3644 |
| [80.0, 100.0] | 2/1 | 0.2296/0.0677 | 1 | 0 | 0.4357 |
| [100.0, 120.0] | 1/2 | 0.2486/0.0732 | 22 | 0 | 0.5527 |
| [120.0, 140.0] | 1/1 | 0.2416/0.0838 | 29 | 0 | 0.3858 |
| [140.0, 160.0] | 1/1 | 0.2441/0.0744 | 3 | 0 | 0.4444 |
| [160.0, 180.0] | 0/1 | 0.2422/0.0690 | 46 | 0 | 0.3362 |
| [180.0, 200.0] | 1/0 | 0.2184/0.0960 | 55 | 343 | 0.3094 |

Actor2无fall，但80–100s仅走.804m，180–200s仅.421m；末20s actor0/2共343个接触pair-frames。新actor1从2次fall增至5次；总体减少并不意味着每人都改善。低tracking误差不能代替速度、全局路线和身体接触质量。

## 同时变化的模型、采样与参数

两run相同controller：`/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/amp_training_corrected_power/step_1000.pt`，SHA256 `1d5051cfc36df71b706e4a136458d32539a8084f15c3d1eb9e4dd2b11568213b`。

Diffusion旧1000 SHA `69d4336d002c84fa848a93a65dd622e5222a663fac63f61b3e94c6872b8c9d15` → 新5000 SHA `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`。

旧为legacy posterior guidance weight1；新为官方clean目标端梯度（独立actor求和、actor内waypoint均值）weight20＋peer5/radius.4＋本地signed-AABB几何10/radius.5。新参数speed/acceleration penalties仍为0，不能声称推理严格限速限加速度。

旧planner batch64、compiled forward／render+400帧视频；新batch8、eager／physics-only、真实contact monitor。四人仍整组处理；代码路径与数值、运行耗时不能当完全相同。实际DDPM100步、FP32、15控制步重规划、endpoint guidance与seed7保持。

场景cuboids、bounds、起点声明、路线、到达半径和A*尺寸一致；新配置文件额外声明timed_path/recovery.5，但本run CLI仍endpoint且recovery0，不能把它当timed guidance效果。旧physics未显式保存terrain_extent字段；新记录64m，不自行把缺失值当不同时钟或地面变更。

墙钟旧429.86s／新613.12s；本对照是物理效果，不是纯planner性能benchmark。

单seed、多项变化、episode censoring与PhysX非逐位恢复限制因果结论。后续要固定新Diff5000/guidance/物理与起点，比较独立低层课程transfer→500→1000；保持同接触与姿态协议并增加seed。

[新逐帧／phase图与terminal证据](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_obstacle_clean_sdf_5000_v2/pedestrian_metrics/phase_quality.md)。完整source hashes、参数diff、测量与采样合同见相邻JSON。

![分段对照](clean_5000_physical_comparison.png)
