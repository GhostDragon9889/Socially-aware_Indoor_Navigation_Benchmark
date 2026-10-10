# 两份真实 200 秒导航采集：冻结风格与停滞诊断

两份采集均为 4 人、6000 个 30Hz 控制帧，保持全时段 inclusive 指标，不删跌倒/低速/不直立片段。冻结 validation 参考、窗口、特征与指标均未修改；reference SHA256 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`。

| 指标 | previous_best | backward_ground200 |
|---|---:|---:|
| 训练 transitions | 2048000 | 8192000 |
| 跌倒 | 0 | 0 |
| 到达路线顶点 | 26 | 21 |
| 完整四顶点 cycles | 5 | 4 |
| inclusive 速度分层姿态+速度 MMD² | 0.09274088529776789 | 0.07238060315284697 |
| 物理速度分布 W1 (m/s) | 0.25128750533861177 | 0.3533679939881452 |
| 直立敏感性保留比例 | 0.9987914652442074 | 0.9995415902650442 |
| 实际静态接触 actor-frames | 923 | 0 |

局部 MMD² 下降，但速度分布差距增大、路线顶点 26→21、完整 cycles 5→4。不能判定所有方面都改善或论文风格收敛。旧 controller SHA `e0d6ed4…`，新 `62c4edc…`，Diffusion 同为 `63e7cad…`；完整 source/config 字典见 JSON。

旧池 319 sources，新池 2848 非重叠 segments / 1853 parents；训练量 2.048M→8.192M，同时 D/RMS/reward/profile 与确定性设置不同。因此这不是单一方向奖励消融；真正 sign-only fulltracking 对照在另一个报告。

## 停滞诊断

定义：同 episode 连续位置差分速度 <0.1 m/s；排除跨 reset 跃迁。严格连续段不合并中途小速度波动。

| actor | idle intervals / valid | idle比例 | 最长连续段 (s) | 该段执行 target误差均值 (m) | 双支持 |
|---|---:|---:|---:|---:|---:|
| 0 | 3486 / 5999 | 58.11% | 38.267 | 0.04227 | 100.00% |
| 1 | 302 / 5999 | 5.03% | 1.767 | 0.05261 | 100.00% |
| 2 | 906 / 5999 | 15.10% | 5.233 | 0.05914 | 96.84% |
| 3 | 1149 / 5999 | 19.15% | 8.600 | 0.06925 | 99.90% |

Actor 0 最长严格 idle 为 133.800–172.067s，净位移 0.0430m；执行 target 误差均值 0.0423m，native 末子步 root速度均值 0.0212m/s。
该段原始 Diffusion .5s 位移均值 0.0342m，5s 位移 0.9933m；实际路线顶点仍相距约 13.36m。
原始模型的近段命令已缺乏前进，远端 5 秒移动不等于每 .5 秒重规划会执行该移动。执行 reference 与 root 很近，不能仅据 idle 将原因归于低层 terrain RMS。这里只定位闭环命令/运动关系，未建立归一化因果。
该段 root 点到最近 box 的几何距离最小 2.0518m；它不是全身体碰撞距离。90/105/120s 的 root/原始与执行 target/路线顶点/box距离原值在 JSON。
`global_goal_xy` 在 timed_path 中是 A* 五米 lookahead 端点；实际完整路线顶点使用 `route_vertex_xy`，不能将前者误报全局到达。

## 新增真实足力 / COM 与旧代理的边界

冻结风格 CLI 只读取原 physical_states 通道，因此其中 actual-foot/true-COM 保持 null；新 raw ground_support 的独立核验另列，不伪改旧指标。AMASS 对应真实 COM、接触力和摩擦仍未知。
新四子步支持占比：双支持 48.0156%，无足支持 0.7479%。支持条件下足部 link原点切向速度均值 0.07314m/s。
后者不是 contact-point slip；滚动/旋转可产生原点运动，不能与旧 near-ground toe 差分代理硬比，也不生成 AMASS 摩擦/接触真值。

![冻结风格分层与速度混合](comparison.png)

完整证据：[冻结原评估](evaluation.json)、[本比较 JSON](comparison.json)、[独立 ground 核验](../ground_support_200s_actual_verification.json)。同 source-parent 与 heldout 无交集已验证；subject/公开 PACER 预训练身份重叠仍未知。
