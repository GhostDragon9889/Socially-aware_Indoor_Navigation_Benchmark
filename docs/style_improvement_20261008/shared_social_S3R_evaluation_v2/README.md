# S3-R12500恢复终态的独立评价

本目录保留首次失败执行；[单独续跑](../shared_social_S3R_navigation_retry_v1/README.md)
于2026-10-08 01:34:43完成导航200s和原CPU审计，两子进程实际退出0。
成功前缀与新导航的[严格组合对比](../shared_social_S3R_comparison_v1/README.md)已完成；本目录的失败记录保持。

2026-10-08 00:50:15（Asia/Shanghai）实际终态：pipeline退出1；navigation200s写完6000步数据后退出-11。
CPU actor、N1生命周期、全部22条跟踪／冻结风格与导航smoke均实际退出0。
正常导航raw-CPU成功审计未执行；54份输出只读保留，失败进程的数据不作为完整SDK成功凭据。

完整11982帧跟踪为14/22成功、零跌倒，世界坐标位置.2767467m、旋转.7496124rad。
45D MMD²=.0230230940，K50/K100覆盖=.58/.52，熵=.63984/.65735；S2覆盖=.72/.63，覆盖存在退步。
物理速度W1=.1690263m/s、90D支持5个速度桶；后续严格比较确认4桶的样本数／参考索引匹配，
等权MMD²=.0684345→.0471716。1.5–2m/s桶因样本数及参考子集不匹配为unknown。

[根部／身体完整CPU重算](../shared_social_S3R_tracking_diagnostic_v1/README.md)实际退出0；
[失败导航数组取证](../shared_social_S3R_failed_navigation_forensics_v1/result.json)实际退出0，
保存数据为零跌倒／接触、4路线节点，84.81%区间低速，最后150秒全部低速。
该取证明确保留实际SDK退出-11，不替代原成功审计或认证质量。
[GDB完整6000步诊断](../shared_social_S3R_navigation_gdb_v1/completion.json)
于01:12:47正常退出0，未复现段错误；故障根因未定位，没有修复证据。

## 来源与协议

只选择已完成恢复训练的`step_012500.pt`，SHA256为
`97ba4c6475817e1bbde4aa56377a45191058a0725cad48fcbf3a9b769accb675`。
恢复完成凭据来自独立OS owner的实际`Popen.wait()`；新凭据使用恢复专用schema，
保持原S3退出未知、物理World重建、未保存454次更新回退的事实。

| 评价 | 固定人口与规则 |
| --- | --- |
| CPU actor准入 | 实际保存rollout首控制步256行，严格只读权重读取，69维／57有效动作，重复确定性、输入／RNG保持 |
| 跟踪 | 全部parent-disjoint validation22，11982个POST物理控制步，原世界坐标身体误差、0.5m失败锁存、全部失败帧保留 |
| 风格 | 原冻结45D参考和K50/K100中心、90D动作参考及共同速度桶；不重新拟合，不使用Test34 |
| 导航 | 原四行人场景，同一Diffusion40k，30步smoke后6000步／200秒；原接触／重置／路线评分 |
| 原始数据审计 | 独立CPU重新统计全部24000 actor-controls、真实法向接触、路线／episode／支持数据 |

论文要求动作风格和动态路线／物理表现分别评估，导航协议包含八场景、N3–15和200秒运行。
本次N4运行用于与本地S1/S2的相同协议比较；作者八场景初态及风格参数仍未恢复，
不直接比较本地MMD²与论文Table III数值。[NavIsaacLab论文](https://arxiv.org/html/2606.26265v1)

## 实现和验证

显式元数据准入：`CrowdSim/humanoid/application/recovery_evaluation_admission.py`。
单一评价适配入口：`scripts/evaluate_shared_social_recovery.py --stage ...`。
适配器在私有模块命名空间改动来源准入和子进程入口；tracking／navigation本体零替换，
完整入口逆向字节检查通过。原脚本、严格权重reader、数值／物理／清理与退出检查保持。
风格输出key明确为`S3_R12500`，没有改动风格抽取或评分公式。

14项CPU检查与Ruff通过，含错误退出／预算／权重／物理恢复声明／原新训练凭据等拒绝检查。
首次冷预检把导航输出放在docs下，被原位置准入正确拒绝；
v1失败文件只读保留，v2仅修正请求输出位置，并按两个原reader各自的schema验收。
两项真实冷预检均退出0，没有Torch或SDK执行；21来源／5输入登记，五份源码快照只读保存。

运行命令（首次启动使用新输出，禁止覆盖失败或已完成记录）：

```bash
PYTHONDONTWRITEBYTECODE=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/own_shared_social_recovery_evaluation.py \
  --plan docs/style_improvement_20261008/shared_social_S3R_evaluation_v2/owner_plan.json --check-plan
```

本次已通过同一入口的`--detach`启动并失败保留；不应覆盖或再次提交同一输出。
owner独立于聊天连接存活，通过OS wait记录完整pipeline实际退出，原pipeline记录每个子进程退出。
启动器退出0只代表成功启动。至少4096MiB空闲显存；外部IsaacSim进程保留，未声明GPU独占。

## 证据

- [终态选择与协议登记](registration.json)
- [恢复训练实际凭据](training_execution.json)
- [实际冷预检退出](preflight_execution.json)
- [CPU检查执行凭据](CPU_preparation_actual_execution.json)
- [OS owner计划](owner_plan.json)、[实际启动](actual_launch.json)
- [源码快照与可逆改动证明](source_snapshot_index.json)

输出：`output/evaluation/shared_social_S3R_budget12500_v2/`。
只有实际终态完整、各子进程退出0、来源保持和独立评分结果齐全后，才可报告本批完整评价。
执行通过仍不等于质量通过；默认部署权重保持。
