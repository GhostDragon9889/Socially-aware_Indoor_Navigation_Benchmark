# 固定分槽的 AMASS / procedural 命令课程

本方案由既有 validation、200秒导航和已冻结源码合同提出，未依据34段正式test选择配方。当前保留negative baseline为部署默认；新的训练/评估输出使用独立目录。

## 方法与两个方案

[NavIsaacLab IV-B](https://arxiv.org/html/2606.26265v1#S4.SS2)以未来XY引导物理动作和AMP；这里继续只给actor真实self-state、地形和未来XY。[TRACE/PACER](https://arxiv.org/html/2304.01893v1#S3.SS3)将规划器与低层控制器分开训练。[CMP](https://arxiv.org/html/2608.03234v1)研究任务语境与通用motion prior的冲突并引入兼容性适配器；本方案没有实现该适配器，不能称CMP复现。

| 方案 | 命令 / episode | 验证成本与风险 |
| --- | --- | --- |
| A，当前实现 | 前128槽真实源相位AMASS，后128槽原程序化命令；clip-end只终止AMASS组 | 恢复程序化任务覆盖；两组各8.192M transition。整体16.384M预算是旧1000update的两倍，不能单独归因于混合。 |
| B，仅设计 | 全AMASS，取消真实源末端reset，线性XY尾部继续到300步cap | 大量实际控制阶段将使用工程尾部；需重证所有合法f0在末控制相位+5秒的floor bound和跨源末端resume。改变episode时域和意图分布，不能称仅隔离短episode。未实现/未运行。 |

## A 的可执行合同

- 固定global前半为AMASS，后半为procedural，无按reset概率重分配、无branch ID进入actor。256槽每32步rollout、2000updates时每组128×32×2000=8.192M真实transition，包括跌倒transition；单独记录episode/reset数。
- AMASS使用10帧newest-first history、至少31个真实未来30Hz帧；f0+实际episode age确定相位，world translation只在reset绑定一次。真实片段内151@30Hz命令严格取源帧；只有源外future采用明确的最后11位置/10间隔线性延拓，不称AMASS groundtruth。到真实源最后帧立即truncated，下一value在reset前测量。
- procedural保持negative baseline实际配置：smooth、2m/s、1.5m/s²、15步replan、300步cap、seed7、192m floor、48m spawn。原51@10Hz曲线升采样供公共151缓存，reward/task仍直接按原canonical51 lerp计算，保留live-root reanchor语义。
- AMASS reset独立CPU generator seed+1，浅拷贝corpus共享不可变bank；procedural reset/expert仍共用原corpus generator；程序化command使用旧private seed。分组draw不等于旧全batch draw序列，全部保存并披露。默认旧模式不创建新generator/采样分支。
- `MixedTrajectoryCurriculum`拥有global↔local映射、各组episode/sourceframe/phase/offset/endvelocity/canonical-plan、两child状态和RNG、真实exposure counters。checkpoint严格要求完整mixed schema；载入GPU映射检查点时显式转回三CPU-generator buffer。缺状态或单组旧checkpoint不得当mixed resume。
- fresh使用相同公开PACER actor/critic/state-D初始化，baseline全部TrainingConfig与uniform expert不变。旧训练weight/优化器没有隐式移入新课程。warmstart新阶段需要另行明确设计，当前未支持跨source resume。

## 验收与边界

业务before快照在 `mixed_curriculum_before/`；完整物理control For AST保持原样。CPU门禁涵盖真实reset/runner方法、native源相位、原程序化插值、两组clip-end/cap/GAE、三RNG/严格resume、默认tensor/RNG/CLI字段。真实v7证明与独立floor/source审计另存。CPU合同不构成SDK/物理成功。

后续由root运行4+4真实resume pilot、fresh2000、固定完整22 validation和真实4人200秒导航；记录训练预算差异、两组exposure、falls/route cycles/contacts与style。训练改善和长时域导航成功均待实际验证，无future3D pose、teleport、外力支撑或最小速度强制。
