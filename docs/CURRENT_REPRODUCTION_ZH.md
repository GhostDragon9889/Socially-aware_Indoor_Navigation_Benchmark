# 行人模拟复现：当前入口

## 2026-10-09：YOHO 姿态与速度优化已完成

固定2848真实AMASS新增128更新/524,288物理转移。四人YOHO完整200秒、零跌倒、无初始化后重置、实际退出0；11项本地物理门槛和行走姿态筛查全部通过，起步与全程无手臂过水平动作。30秒30fps真实物理状态录像已完成；全900帧FK误差小于1mm。

成对60秒AMP比较中，行走侧倾p95由6.795°降到4.862°，扭转由8.004°降到7.443°；摆臂幅度接近，风格MMD与支撑脚原点速度代理退步。Diffusion公共网络权重保持，推理引导与有界轨迹优化有调整。当前只是单场景/四人/seed7工程复现。

详见 [YOHO结果与复运行](/project/EXP/SangforAndSaut/ProtoMotions/docs/YOHO_DIFFUSION_AMP_ZH.md) 和 [来源证据](/project/EXP/SangforAndSaut/ProtoMotions/docs/yoho_diffusion_amp_evidence_index.json)。以下为保留的历史结果。


目标是IsaacSim6／IsaacLab3上的Diffusion＋AMP。当前已有真实训练和物理评价，**尚未证明达到或超越论文完整效果**。作者动作ID、八场景清单和风格指标参数仍未匹配；本地协议的数值不能直接与论文排名。

## 本轮最新进展

[S3-R12500训练与组合独立评价](style_improvement_20261008/shared_social_S3R_comparison_v1/README.md)已完成。
2026-10-08 01:34:43：导航200s及原raw-CPU审计实际退出均0，6000步／24000 actor-controls、清理成功。
组合复用已实际成功的跟踪／风格前缀，旧失败保持；72份资源／人口／抽样严格比较CPU退出0。
对比S2，导航跌倒9→0、行人接触32→0、路线节点1→4；仍存在长期路线停滞和覆盖下降。

[首次v2终态独立评价](style_improvement_20261008/shared_social_S3R_evaluation_v2/README.md)
跟踪／风格已实际退出0：全部22条／11982帧，14/22成功、零跌倒，位置.276747m／旋转.749612rad。
45D MMD²=.0230231、K50/K100覆盖=.58/.52；比S2距离更低、覆盖更窄，完整质量目标仍未通过。
导航6000步数据写完后实际退出-11，2026-10-08 00:50:15 owner记录pipeline退出1；原raw-CPU成功审计未执行。
失败数据独立CPU取证退出0：零跌倒／接触、4路线节点，84.81%区间低速，最后150秒全部低速；未认证SDK成功。
根部／身体完整帧分解实际退出0，原世界坐标评分保持；GDB6000于01:12:47正常退出0，未复现故障或定位根因。
01:20:09无GDB的[导航独立评价续跑](style_improvement_20261008/shared_social_S3R_navigation_retry_v1/README.md)
已完成，仅补导航200s和原CPU数组审计，54份保留输出只读保持。
恢复来源与原新训练凭据明确分开，原数值／物理／失败帧及冻结风格参考保持；14项CPU检查与Ruff通过。
原数值和实际退出准入保持；26项CPU检查、34个subtest及Ruff通过。
真实无GDB退出及正式raw-CPU审计已通过；完整路线与论文质量尚未达到，默认不变。

[episode候选终态来源准入／独立评价入口](style_improvement_20261006/shared_social_episode_evaluation_preparation_v1/README.md)
已完成CPU接线：修正重复预检的私有helper缓存冲突，15项检查及11个subtest、Ruff和来源预检通过。
真实S2凭据先经过原退出／CPU审计准入，再被新候选终态选择正确拒绝，预期退出1；Torch／SDK均未导入。
10份证据只读封存并独立冷核验，57项来源绑定／3项真实S2元数据输入／23项S3guard保持。
完整跟踪／风格／N4 Diffusion评价继续调用原数值流程；新候选训练、终态接受及真实评价尚未执行。

[episode固定参考课程／World入口CPU准备](style_improvement_20261006/shared_social_episode_reference_runtime_preparation_v1/README.md)
已实际CPU退出0：真实2848、8人／256人两种规模的私有入口和World准入通过，
原AMASS轨迹／reset来源及每种规模4096专家来源窗口逐值保持；66项检查与Ruff通过。
程序路径只在重置生成，按episode年龄查询；物理上限加5秒参考及RNG曝光差异明确登记，AMASS仍future31。
31份证据只读封存并独立冷核验，41项实现来源／4项输入／23项S3guard保持。
三份配置预检退出0，缺少完整S3证书的启动准入退出1；SDK、独立2000更新训练和效果测量尚未执行。

[程序轨迹重锚定全量CPU对照](style_improvement_20261006/shared_social_procedural_anchoring_CPU_v1/README.md)
已实际退出0：固定2848各取最早合法帧9的真实位置／姿态，声明静止／初始XY速度匀速运动，
三机制共2563200奖励行；实际S2程序配置／退出、7项输入与10项直接实现来源保持。
静止平均轨迹项为周期重生成.964443、同首路径周期重启.964326、episode时钟.234614；
该机制持续保留高静止奖励，尚未证明实际训练失败因果或物理改善。
新EpisodeProceduralReference只在重置生成路径，查询不依赖当前位置；真实2848的455680目标查询、
原首51节点／前150个POST目标逐值保持和参考状态恢复通过，CPU实际退出0。
49项检查与Ruff通过；10秒episode加5秒未来是明确的本地课程设计。该历史模块批次未接World，
上方新入口批次已完成World的CPU接线；SDK和独立训练尚待执行。
21份证据只读封存并独立冷核验，7项实际输入／10项直接来源及23项S3guard保持，原始均值与候选目标逐值核对通过。

[实际权重的路线条件响应](style_improvement_20261006/shared_social_trajectory_response_CPU_v1/README.md)
已实际CPU退出0：S1/S2共享S2的11982评价／8192训练观测，六条件共242088策略输入行前向。
只更换路线20维；身体／地图／RMS保持。两个actor有非零响应，本批有效动作均无限幅，
27项检查及Ruff通过；尚未证明转向、步态和物理闭环质量。后续重锚定CPU对照见上方，候选仍需实际训练。
15份证据已只读封存并独立冷核验，81项实际输入／10项直接实现来源和23项S3guard保持。

[全量前向轴／重置姿态核对](style_improvement_20261006/shared_social_heading_geometry_CPU_v1/README.md)
实际CPU退出0：固定2848的全部523272帧及S1/S2各22条／11982评价帧，75项输入与8项实现来源保持。
本批未观察到固定90°人体前向偏移或左右横向轴反转；22个重置帧最大位置／旋转平均误差仅
1.446798e-6m／1.120105e-6rad。34项检查与Ruff通过，原评分、跌倒／失败帧及部署模型保持。
这是排查证据，尚未定位训练失败原因或产生新动作／物理效果。
15份证据已只读封存并独立冷核验，S3的23项来源guard保持。

[完整跟踪根部／身体诊断](style_improvement_20261006/shared_social_tracking_root_body_CPU_v1/README.md)
现已实际CPU退出0：S1/S2各22条、11982评价帧，保留全部跌倒和失败后帧，原始世界坐标评分逐值保持。
S2根部XY误差0.853973m、根部旋转1.252676rad；去除根部平移／完整旋转的身体诊断为0.218184／0.132644m，
不能替换原始0.918030m和0/22成功，也不能与论文0.218m等同。后续重点同时检查轨迹／朝向跟随。
22项CPU检查及Ruff通过；闭合条件反转触发6项失败并恢复。18份证据只读核验，74项原始输入保持；
S3的23项guard未变，本轮没有新增物理模拟效果或训练失败因果结论。

[完整捕获命令／相位核对](style_improvement_20261006/shared_social_tracking_commands_CPU_v1/README.md)
也已实际CPU退出0：两模型各119820目标点，PRE时钟／相位／根部和原始20维前缀一致，未发现本批错帧。
末端保持14.9040%；同相位参考根部公式离线0.296733／0.343848，不是实际训练reward trace。
38项CPU检查及Ruff通过，条件反转触发3失败并恢复；17份文件只读核验、83项输入保持。
下一步审查训练源剩余时长、episode长度和共享组重置；没有新的SDK模拟或论文效果结论。

[实际episode时长诊断](style_improvement_20261006/shared_social_episode_horizon_CPU_v1/README.md)
已实际CPU退出0：完整S1/S2组episode平均1.699144／1.933537秒（扣除末尾未完成年龄），
独立完整validation22平均18.154545秒；训练canonical50未来节点的来源外比例48.6179%／49.6418%。
全部2848原始bank来源/帧数/30Hz对应清单，最短91帧；可在不丢弃动作的前提下要求future81。
32项检查及Ruff通过；该时长差异未证明失败因果，缺少全程重置ID/帧档案与episode直方图。
[独立future81课程CPU准备](style_improvement_20261006/shared_social_minimum_future81_CPU_v1/README.md)
37项检查及真实256人来源接口通过，同合法绑定轨迹/10目标和4096专家来源窗口逐值保持。
两批共20份文件只读封存并冷核验，S3的23项guard保持；条件反转均被检查发现并精确恢复。
[独立World／entry接线](style_improvement_20261006/shared_social_minimum_future81_runtime_preparation_v1/README.md)
已完成CPU验证：真实2848／256人接口退出0、40项实现来源保持，23项检查及Ruff通过。
新World仅四处来源／类型／future81准入替换，逆向恢复原物理实现字节；三份原配置逐字复制。
SDK smoke、2000更新候选训练、风格和物理评价仍待S3完整结束；启动准入实际退出1且没有启动SDK，尚无性能改善证据。

[原生串行工具owner](style_improvement_20261006/shared_social_native_tool_owner_preparation_v2/README.md)
已完成CPU准备：42项pytest、13项V8契约及四地图登记/冷读取通过，10个子进程退出0。
命令/路径与只读登记逐字段比较，反转条件被检查捕获，源码精确恢复；71份文件只读核验。
桥接的56份证据另已封存，S3的23项guard保持。真实工具链因缺少S3完整完成证书而实际退出1，
没有启动SDK；仍待S3及后续评价结束后验证21步原生流程、准确初态和200秒物理/风格效果。

[S3-R优化器恢复训练](style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/README.md)已完成：
2026-10-07 23:46:12（Asia/Shanghai）终态训练／CPU审计实际退出均0，恢复段500/500／4096000转换；
有效总计12500更新／102400000转换，预选`step_012500.pt`与完整rollout已生成，28份输出只读。
保留原12000保存点，未保存454更新回退并重算。模型／两Adam／四RMS／回放／训练RNG恢复，
物理World／课程私有RNG／历史重建；不是原S3连续执行或完整物理续训。
真实256人单更新smoke实际退出0、CPU审计通过，8项检查及5个subtest、Ruff通过。
每25次保存权重、125次保存rollout；20权重／4rollout保持，202张量与完整计数／RMS／Adam审计通过。
[外部冷核验](style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/owner_capacity_v3/external_verification.json)
实际退出0，16来源／4输入及原权重／日志／固定AMASS资源哈希保持；终态清理成功。
外部IsaacSim进程保留、非GPU独占；原[丢失会话证据](style_improvement_20261007/shared_social_S3_session_unavailable.json)及原数据保持。
恢复训练及终态审计完成，跟踪／风格已完成、导航退出问题正在调试，默认不变；原cell581完成证书没有补造。
最近完成的[同AMP新鲜DP／SFM配对](style_improvement_20261005/shared_social_S1_paired_actual_v1/README.md)
两臂全部实际退出0：DP跌倒更少／风格距离更低，但路线／覆盖更低，均未通过质量要求。

新增[原生共享AMP接入](SHARED_SOCIAL_NATIVE_EVALUATION_ZH.md)：四地图52个固定资源案例中48准入、
4个Hospital高人数配置因间距不足拒绝；2058段A*与48个SFM合成快照预测通过，实际CPU退出0。
原生DP/SFM入口、Pop-Art准入及N+1接触视图释放已通过30项CPU检查；SDK-free两臂资源预检通过。
真实原生SDK smoke、准确初态配对、多seed及完整物理/风格审计仍待执行；CPU准备不证明论文效果。
[原生初态最终CPU接线](style_improvement_20261005/shared_social_native_initial_preparation_v3/README.md)新增N3–15
完整人体/历史/RNG与原地图四通道的相同采集点；严格两臂预检退出0，最终79项相关检查通过。
准入突变被3项测试发现并恢复原字节；临时回调在成功/异常时释放。真实World初态与200秒评价尚未执行。
新增[原生raw CPU接线与封存](style_improvement_20261005/shared_social_native_raw_audit_preparation_v2/README.md)：
最终62项检查通过，N3/N15的30/6000步来源检查实际退出0；人数、子步、四元数、脚部轴分别校验。
旧S1 DP/SFM两份真实N4数据的所有指标保持；准入突变实际21项失败，源码恢复后全通过。
已修正driver参数ABI假设并保留v1失败，88份文件只读封存；这不是新的原生SDK/质量结果。
原seed7的raw reader准备完成，实际原生捕获及独立风格评价、多seed和作者八场景协议仍需验证。
新增[原生独立风格CPU准备](style_improvement_20261005/shared_social_native_style_preparation_v2/README.md)：
53项检查、两项来源预检、历史真实DP/SFM的45D/90D及全部共同桶数值回归通过，外部实际退出0。
不重新拟合参考或中心，不读取Test34。修正仅涉及证据runner的tuple/list JSON表示比较，
所有数值严格相同、无容差；v1失败保留。92份证据只读封存，新的原生风格数据尚未评价。
新增[显式seed原生入口及矩阵](style_improvement_20261006/shared_social_native_seeded_preparation_v1/README.md)：
48项CPU检查及17个子进程通过，类型校验突变触发26项失败并精确恢复，88份文件已只读核验。
本地seed7/19/42展开156组合，144资源准入、12原有拒绝；288次两臂200秒运行均为计划，尚未执行。
实际driver保持原字节；严格S1兼容预检仅改变seed/output，不改变S3训练或其终态模型选择。
该阶段的raw/风格接口缺口已由下方独立版本补齐；真实World/RNG配对和完整独立评价待执行。
该矩阵是本地鲁棒性扩展，未恢复作者八场景/seed协议。
新增[显式seed独立raw／风格CPU准备](style_improvement_20261006/shared_social_native_seeded_admission_preparation_v1/README.md)：
195项相关检查、24项raw与6项style冷来源检查通过；33个子进程和外部流程实际退出0。
seed相等条件突变触发13项失败并精确恢复。历史真实两臂的全部原始指标、45D/90D及共同桶保持。
119份文件已只读核验，S3配方/23项来源guard与默认权重保持；尚无新的原生SDK或质量结果。
[分项计数修正](style_improvement_20261006/shared_social_native_seeded_admission_preparation_v1/test_inventory_correction.json)
确认新接口95／原回归100、总195项；封存README的97/98误写保留并纠正，不改变通过结果。

已修正 AMASS 清单被误要求只读及 PhysX 空过滤列表的兼容问题。
共享 World 已实际完成 8 人／9 控制步／36 物理子步，整组重置和跌倒探针通过；
真实 AMASS 的 8 人短训练也已完成 1 次 PPO／AMP 更新、32 条物理转换，两个进程均正常退出 0。
权重／Adam／replay／RMS／rollout 的 CPU 审计通过；这只是训练闭环检查，不是动作效果达标。
256 人／64 组检查已完成 2 次更新／16,384 条转换，真实退出 0 且 CPU 审计通过；
全新初始化的 2,000 更新 pilot 已完成 16,384,000 条物理转换、8,192,000 个专家窗口，
主机实际退出 0；202 张量的终态 CPU 审计通过。独立 validation22 与固定风格测量也已退出 0，
但质量失败：0/22 成功、22/22 摔倒，位置/旋转误差 1.865725 m/2.138779 rad。
四人 200 s 实际退出 0，独立 CPU 数组审计确认 956 次摔倒重置、四人路线点均为 0；
保持默认权重。Pop-Art 的真实 8 人/256 人短训练、审计及 CPU 加载已通过，
S1同数据/seed/预算全新2,000更新及全部独立流程已完成，实际退出0；
仅改变value RMS的critic输出补偿。跟踪仍0/22，跌倒动作7/22；位置/旋转1.008892m/1.332191rad。
共同速度4区间90D MMD² .509862→.068384，固定45D MMD² .035621，
但K50/K100覆盖率降为.64/.49。四人200s为5次跌倒、37个peer pair-controls、3个路线点。
稳定性改善，质量仍不合格。S2仅关闭KL提前停止；全新2000更新及全部独立评价现已完成，实际退出0。
严格准入通过，但仍0/22跟踪；导航跌倒5→9、路线点3→1。覆盖率提高未带来整体改善。
[S2完整结果](SHARED_SOCIAL_FULL_EPOCH_RESULTS_ZH.md)保留风格、速度、路线、物理各轴。
[S2运行与自动接续](style_improvement_20261005/shared_social_full_epochs_v1/README.md)记录完整预登记及外部owner。
详见[S1完整对照](SHARED_SOCIAL_POPART_RESULTS_ZH.md)及[S0原始结果](SHARED_SOCIAL_PILOT2000_RESULTS_ZH.md)。
当前命令、配置和实际记录集中在[共享行人 AMP 入口](SHARED_SOCIAL_AMP_CURRENT_ZH.md)。
新增[独立原始运动诊断](style_improvement_20261005/shared_social_locomotion_CPU_v1/README.md)
确认S1有效位置间隔80.46%低速、90.44%双支撑、最长连续双支撑149–176s；
低跌倒未伴随完整步态和路线恢复。新增[严格准入层](style_improvement_20261005/shared_social_admission_v2/S0_S1_supplemental_admission_v2.json)
补齐实际child绑定、完整跟踪协议、底层风格来源和90D样本可比性，原S0/S1数值仍通过。
[终态D离线动作敏感性](style_improvement_20261005/shared_social_D_input_diagnostic_v1/README.md)
为负向诊断，暂不支持动作代理主导失败；不是训练消融。
[公开SFM的同AMP高层接口](style_improvement_20261005/sfm_shared_amp_preparation_v1/README.md)
已完成22项CPU测试；未执行SDK／物理接入，作者八场景协议仍未恢复。
现已补上[SFM fixture评价接入代码](style_improvement_20261005/shared_social_sfm_integration_v1/README.md)：
5项CPU bridge测试及实际S1资源预检查通过；原始post-control输入成功生成raw51和共同causal151。
现已完成新鲜两臂SDK smoke／200s、完整原生CPU审计、初态及闭环风格配对。
新增[两臂共用的规划器构造前初态接线](style_improvement_20261005/shared_social_paired_initial_v1/README.md)：
7项CPU契约、两个source inverse及真实S1两臂资源预检查通过。记录完整身体／关节／AMP历史、native link／COM、路线、资源SHA及全局／AMASS私有RNG；初态比较拒绝缺失或漂移。
新鲜S1 DP/SFM各30步均实际退出0，32通道完整初态／资源／RNG精确匹配。
两臂6000步及全量native CPU审计已退出0：DP为5跌倒／0静态／37peer pair-controls／3路线点，SFM为20／21／31／6。
完整32通道初态仍精确相同；[完整结果](style_improvement_20261005/shared_social_S1_paired_actual_v1/README.md)保留风格／速度／覆盖与物理取舍。
[实际运行状态](style_improvement_20261005/shared_social_S1_paired_actual_v1/live_execution.json)记录工具退出，历史DP不充作本次基线。
新增[smoke／200s共用原生验收](style_improvement_20261005/shared_social_high_level_raw_audit_v1/README.md)：7项CPU契约通过，历史短测与完整S1实际退出0，全部24身体位置匹配末原生子步；完整原指标不变。新入口为实际新鲜两臂准备了初态→首plan/reference→原生clock的验收链，没有新增GPU效果证据。
新增[固定2848原先验／实际专家末批](style_improvement_20261005/shared_social_expert_coverage_CPU_v1/README.md)：5项CPU契约、完整494792短窗和S1真实4096行核对已退出0；区分clip/window分母、整段／短窗proxy和BABEL显式时间覆盖。fast原采样概率1.5801%，末批46行；没有改变训练先验或由单批推断全程曝光／质量原因。

新增[实际PRE任务上下文](style_improvement_20261005/shared_social_task_context_CPU_v1/README.md)：8项CPU测试、8192控制行／4096专家的真实读取退出0，动作前后phase／episode／native速度核对通过。程序和AMASS目标分支的未来段中位位移率≥1.8m/s分别1628／20行（各4096），身体均速1.148517／.542247m/s；目标位置／位移率不等于即时速度命令，且未与expert逐行配对。失败和reset行全部保留，未修改先验或证明低速原因。

新增[独立BABEL dense跨度约定](style_improvement_20261005/babel_dense_sequence_convention_CPU_v1/README.md)：6项CPU测试及完整2848构建实际退出0；304个明确单动作dense父源对应399个clip，与原844个显式timed union全clip覆盖分开保存。无frame的多动作、extra-only及未知项保留，不改动作ID／类别proxy／训练prior。

S2完整训练、独立动作／风格／200s导航、原生CPU复算及严格补充准入均已完成。
模型质量未通过，默认权重保持。新增[闭环风格入口](../scripts/evaluate_shared_social_navigation_style.py)：
直接分析实际规划驱动的完整全身姿态，冻结AMASS验证参考与聚类中心，保留失败；8项CPU回归检查及实际两臂测量通过。
共同五桶固定平衡样本数和参考索引，90D MMD²为DP .112125／SFM .131224；DP的速度／覆盖／路线更低，不能称整体达到论文效果。

## 当前模型选择

| 部分 | 权重／实验 | 当前结论 |
| --- | --- | --- |
| 默认AMP | `62c4…/step_1000.pt` | 保持默认；已有Warehouse4／15人200秒零跌倒、零接触记录，Office三人10秒验证也完成 |
| 状态动作耦合先验、weight2 | `19f334…/step_2000.pt` | 完整22条17成功；部分动作和native风格改善，未替换默认 |
| AMP外层weight0.5 | `09e802…/step_2000.pt` | 全22条15成功，四人200秒3次跌倒／8个行人接触pair-frames；物理要求失败，后续native分支未执行 |
| 默认Diffusion | `63e7…/step_5000.ckpt` | 保持默认 |
| 完整两域40k EMA Diffusion | `4581f29d…/step_40000.ckpt` | 训练及严格终态检查已完成；固定主200窗口ADE降低53.77%、FDE降低51.95%，但行人碰撞点率3.06%→3.20%；同AMP四人200秒完整闭合：1跌倒／0静态／45个行人接触pair-controls／24个路线点，物理要求未通过，未替换默认 |
| 新1,000地图Diffusion | `b563…/step_5000.ckpt` | 100场景配对ADE／FDE改善；同源四人200秒1次跌倒／70个行人接触pair-controls／9次到达，旧模型全零物理失败／21次到达；CPU核对与完整配对均已完成，默认不替换 |
| 新Diffusion的10候选扩展 | 默认AMP62c4＋DPb563，S1→S10 | 同源四人200秒完整闭合；**3次跌倒涉及2人**／0静态接触／279行人接触pair-controls／18次到达；idle19.539%低于S1的30.035%，物理要求失败，后续S10 native分支不执行 |
| 新Diffusion的方差引导 | 默认AMP62c4＋DPb563，S1／variance | 同源四人200秒GPU及203项raw核对完成；0跌倒／0静态／315行人接触pair-controls／9次到达；跌倒减少伴随接触增加，物理要求失败，不晋升默认、不执行该设置native扩展 |
| AMP专家分层先验 | `98a49058…/step_2000.pt`，只改变专家采样 | fresh2,000步／1,638.4万条物理转换完成；全22动作20成功／0跌倒，四人200秒0跌倒／0接触／32次到达；风格有取舍，默认保持 |
| AMP状态判别器消融 | `97b2d278…/step_2000.pt`，固定q采样，D输入1950 | fresh2,000轮／1,638.4万条转换、终态审计及全22动作完成；18成功／0跌倒，位置.218970m／旋转.751018rad／跟踪90D MMD² .05209593；四人200秒另有3次跌倒／0接触／31次到达，203项raw核对通过，未替换默认，也不是Nav作者状态动作判别器的等价实现 |

新DP的[轻量失败事件诊断](style_improvement_20261003/default_newDP_failed_event_diagnostics_v1/README.md)：141.833秒行人3跌倒，当步无足部支撑且无静态／行人接触标记；这是同帧关系，尚未证明原因。

## 结果与运行路径

- [新社交控制器实际CPU数值检查](style_improvement_20261004/social_actual_runtime_v1/CPU_numerical_gates_actual_root_owner.json)：三个独立进程共10项通过；分别0.037／1.813／2.093秒。实际检查共享CNN布局、PPO／D更新、完整Adam／RMS／replay／RNG恢复、actor-only权重加载及critic候选梯度；两次32样本合成update，不是AMASS／物理训练。
- [真实8人两组接触检查的失败记录](style_improvement_20261004/social_actual_runtime_v1/group_physics_failed_actual_root_owner.json)：57610／f3fb8d实际退出139；必填experiment_name缺失导致config断言，未创建群组场景，不能声称碰撞验证通过。原失败目录和日志已封存；新probe及training入口同样缺失，修订后重新验证。
- [必填模拟器配置修订](style_improvement_20261004/social_simulator_config_revision_v1/README.md)：八人gate、shared probe和training各补一个experiment_name关键字；原PINS／物理／PPO／AMP／cleanup保持。编译／逆还原／AST／Ruff、独立审阅与封存已完成。实际20529／dbd6e1中config断言消失，64子步碰撞检查全部通过（同组正探针1355.844N、跨组raw0），但进程仍退出139；[记录](style_improvement_20261004/social_actual_runtime_v1/group_physics_config_v2_actual_root_owner.json)已封存，不能晋升SDK gate。下一修订仅补SDK的SimulationContext资源释放。training必须显式指定旧封存profile，未来评价metadata仍需接纳新实际入口身份。
- [新增ORCA-Interact官方格式原始数据](style_improvement_20261003/orca_interact_raw_actual_runs_v1/README.md)：1,000场景、10,608行人实例、800／100／100分割，全量CPU检查通过并封存；不是作者提供的数据。
- [显式清理SimulationContext后的真实失败](style_improvement_20261004/social_actual_runtime_v1/group_physics_cleanup_v3_actual_root_owner.json)：45011／b05ce8实际退出139；物理检查、stop、clear_instance与app.close均返回成功，仍不能视为正常退出。[原生GDB诊断](style_improvement_20261004/social_actual_runtime_v1/group_physics_gdb_v4_actual_root_owner.json)在Python最终GC中捕获SIGSEGV，栈经过PhysX tensor插件、tensor Python绑定及traceback/frame析构。调试器退出0不代表被调试SDK进程退出0。下一修订释放入口拥有的tensor API引用并在扩展仍存活时回收；未启动新社交训练。此方向也得到[Isaac Sim 6.0官方已知问题](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/known_issues.html#crash)的支持。
- [拥有的原生视图释放检查](style_improvement_20261004/social_actual_runtime_v1/group_physics_tensor_release_v5_actual_root_owner.json)：20297／3b52bd仍退出139；8个contact弱引用中有1个在容器清空后仍存活，检查明确拒绝。随后[引用来源诊断](style_improvement_20261004/social_actual_runtime_v1/group_physics_referrer_diagnostic_v6_actual_root_owner.json)实际66636／5060cb再次观察139，定位index7由已返回的_contact_views帧最后循环变量view持有。具体异常缓存的拥有者尚未确定；下一修订释放helper本地API别名，仍以真实退出0作为SDK准入条件。
- [helper别名修正后的实际检查](style_improvement_20261004/social_actual_runtime_v1/group_physics_helper_release_v7_actual_root_owner.json)：92584／7560f3仍退出139，但8个contact弱引用已全部死亡，64物理子步及cleanup标志全部通过。第二次[GDB原生栈](style_improvement_20261004/social_actual_runtime_v1/group_physics_second_native_gdb_v8_actual_root_owner.json)定位到timeline绑定的class析构，已不同于原tensor帧析构；正审查版本限定的终态接口缓存释放，未据报告passed启动训练。
- [终态Timeline修订的实际失败](style_improvement_20261004/social_actual_runtime_v1/group_physics_timeline_compat_v9_actual_root_owner.json)：29564／4d6c58实际退出139；64子步接触通过、8个contact弱引用全部释放，但联合SDK身份检查拒绝，未调用Core终态关闭或清空Timeline缓存。源码已定位：安装版SimulationContext保存的是ResolvableString管理器代理，代理直接与类作is比较必定不等；下一修订固定代理源码／字符串并要求解析结果仍为同一个canonical PhysxManager。此定位不能代替真实正常退出。
- [精确代理解析后的真实失败](style_improvement_20261004/social_actual_runtime_v1/group_physics_timeline_resolver_v10_actual_root_owner.json)：75237／890bca实际退出139；上下文、代理来源／字串及canonical解析全部通过，下一项Lab清理状态检查拒绝，仍未进入Core／Timeline释放。安装版PhysicsManager.close也用sim.physics_manager is cls判断拥有者，与该代理机制冲突，会跳过Base上下文／配置释放。下一修订仅在已停止且精确拥有的上下文关闭前，将公开physics_manager字段规范为已验证的canonical类，由原SDK close执行正常STOP／Base清理；不直接改SDK文件或Base私有指针。
- [规范化管理器后的真实失败](style_improvement_20261004/social_actual_runtime_v1/group_physics_manager_normalization_v11_actual_root_owner.json)：25781／1cf318仍退出139；本轮已逐字段实测Base上下文／配置清空，Core关闭及两个Timeline缓存释放均成功，8个contact弱引用全部死亡、64子步通过且cleanup_errors为空。报告passed不能覆盖进程失败。[第三次原生GDB记录](style_improvement_20261004/social_actual_runtime_v1/group_physics_third_native_gdb_v12_actual_root_owner.json)在模块最终GC捕获PhysX tensor析构SIGSEGV，调试器0不是被调试进程0。源码另发现PhysxSceneDataBackend仍缓存SimulationView，正在通过公开simulation_view setter补清并要求实际弱引用销毁；此缓存是否为剩余崩溃原因尚待实测。
- [公开后端释放后的真实失败](style_improvement_20261004/social_actual_runtime_v1/group_physics_scene_backend_v13_actual_root_owner.json)：86319／3c377c实际退出139；后端公开setter清空两个缓存字段，实际既有SimulationView为1个、is_valid=false，但原post-context GC后仍存活1个，新增检查正确拒绝。8个contact弱引用已全部释放，Base／Core／Timeline清理及64物理子步通过；不能据这些字段称进程修复。下一步仅对该既有弱引用目标记录引用者类型、源码位置或匹配字段，保留失败并继续定位。
- [限定目标的真实引用诊断](style_improvement_20261004/social_actual_runtime_v1/group_physics_backend_referrer_v14_actual_root_owner.json)：25082／798bc4实际退出139，原弱引用失败检查保持；目标的8个直接引用均为字典中的_physics_sim_view字段，一跳识别为4个Articulation和4个ArticulationData。诊断列表本身只是查询期间的临时引用，不称生产持有者；未遍历全局原生对象或序列化任意字段值。下一修订仅在停止模拟后、原SDK关闭前，针对本次创建且与同一后端view严格匹配的资产／数据字段释放终态引用；源码修订不能代替真实正常退出。
- [八个资产字段释放后的实测](style_improvement_20261004/social_actual_runtime_v1/group_physics_owned_asset_release_v15_actual_root_owner.json)：38103／8c6daa仍退出139；4个Articulation及4个Data字段清空、8个contact弱引用和后端SimulationView弱引用均为0，原64子步物理、Base／Core／Timeline清理通过且cleanup_errors为空。[第四次原生栈](style_improvement_20261004/social_actual_runtime_v1/group_physics_fourth_native_gdb_v16_actual_root_owner.json)实际76734／700795调试器退出0、被调试进程SIGSEGV；此次栈是core simulation-manager插件→C __run_exit_handlers→exit→Py_Exit，已不同于此前Python模块GC的tensor／Timeline析构。具体C回调或原生析构原因仍未确定，正在核对指令与内存映射；不据局部passed启动训练。
- [原生退出指令与映射核对](style_improvement_20261004/social_actual_runtime_v1/group_physics_native_exit_mapping_v17_actual_root_owner.json)：51089／96e2c9调试器退出0、被调试进程SIGSEGV；PC仍位于已映射core插件，但call *0x10(%rax)访问的地址不在进程映射中，不能称插件函数本身已卸载。具体C++对象尚未被符号证明。[官方v6.0.0插件源码](https://github.com/isaac-sim/IsaacSim/blob/v6.0.0/source/extensions/isaacsim.core.simulation_manager/plugins/isaacsim.core.simulation_manager/PluginInterface.cpp)显示StageUpdate节点句柄未保存、两个全局智能指针未在onShutdown释放；当前仅据此准备独立源码修复候选。三个公开接口头与安装版逐字节一致，rc.59／Kit110.1.1／扩展1.15.4匹配；不修改安装SDK，不把源码候选称真实退出修复。
- [独立原生修复后的真实正常退出](style_improvement_20261004/social_actual_runtime_v1/group_physics_native_overlay_v18_actual_root_owner.json)：93877／e8ad50直接SDK进程退出0，未使用GDB或fast_shutdown；实际加载路径、完整Python来源和未改动的cp312绑定均指向独立扩展副本。原64子步、同组1355.844238N正探针、跨组raw力严格0及全部弱引用释放检查保持并通过；35个安装扩展文件前后SHA相同。源码只保存／删除本插件StageUpdate节点，并在原onShutdown释放所持智能指针；四CPP编译和链接已实际成功。此结果验证当前八人探针正常退出，尚未验证共享World、训练或论文动作效果。下一步在新的入口修订中复用这个扩展与加载校验。
- Formal EMA40000固定风格指标已实际完成：[90D CPU记录](style_improvement_20261004/social_actual_runtime_v1/formal_EMA40000_style90_actual_CPU_owner.json)96705／657bb7退出0，5个支持速度区间的MMD²均值0.05592538；[45D CPU记录](style_improvement_20261004/social_actual_runtime_v1/formal_EMA40000_pose45_actual_CPU_owner.json)33799／d886a7退出0，固定b25中心、2048×2048确定性抽样MMD²为0.02297688，K50／K100覆盖率0.86／0.71。读取全部6000控制帧／24000行人帧，45D仅排除4个首帧及1个跨reset端点，不筛跌倒、接触或静止。仅使用固定validation22参考，未读取Test34数组，也未重新拟合中心；这是本地指标，不能直接与论文Table III数值比较。当前风格reader的实测足接触速度与true COM字段为null；真实子步接触／COM保留在独立raw审计中，不能将风格代理量称为这些实测指标。
- [实际可用Interact语料](style_improvement_20261003/orca_interact_usable_actual_runs_v1/README.md)：时间戳、速度方向代理、缺失地图与19邻居接口已适配；全部1,001文件检查并封存，原筛选视图训练窗口150,439；正式混合训练使用后述完整时间窗口视图。
- [官方ORCA Maps流程的可用本地语料](style_improvement_20261003/orca_maps_official_usable_actual_runs_v1/README.md)：1,000场景、2,001文件全量检查并封存，原筛选视图训练窗口78,149；45个空地图和原始重叠诊断均保留。原两域筛选视图228,588窗口作为历史试训保留。
- [包含零近邻样本的完整时间窗口视图](style_improvement_20261003/orca_all_temporal_view_staging_v3/README.md)：实际构建与CPU检查均退出0，训练233,046／验证29,952／测试29,466窗口；新增4,458个训练窗口，全部原窗口保留。复用原states/maps，仅新建2,000个独立索引及两份manifest，均只读封存；两次构建失败和partial保留。
- [两域Diffusion的EMA与梯度累积实现](style_improvement_20261003/orca_official_mixed_training_staging_v2/README.md)：原筛选视图batch400显存不足，200×2四轮及恢复五轮已完成。[完整窗口实际训练](style_improvement_20261003/orca_official_all_temporal_actual_runs_v1/README.md)另已完成四轮及恢复五轮；**40,000次更新已实际结束（50502／f10f35退出0），累计16,000,000窗口曝光，终态CPU检查和固定配对已完成；完整效果仍待物理及风格评价**。正式训练从原始4b6权重重新初始化，不续训试训权重；EMA、验证与完整窗口指标采用已冻结的新配置。40k固定800窗口／域的EMA损失为Maps .1071854、Interact .0811223，较早期有回升；不能以更新数认定效果改善。
- [固定配对评价与独立旧权重副本v3](style_improvement_20261004/paired_validation_readonly_baseline_v3/README.md)：源码adeaea41经compile／Ruff／独立审阅并封存；固定main200和另504诊断、种子／模型／loss／指标保持。旧63e加载字节相同的0444／nlink1副本，原0600／nlink2文件及别名保持。首个实际29675因缺少仓库import路径退出1，已保留；明确PYTHONPATH后91992／23e92b实际退出0，245.187秒，181个输出文件与实际owner均已只读封存。[完整报告](../output/diffusion_amp/formal_EMA40000_vs_63e_validation_readonly_baseline_v3_retry1/comparison.json)保留所有主200／另504诊断窗口及失败碰撞，诊断重叠20窗不合并成704独立样本。
- [同AMP的40k EMA主物理评估](style_improvement_20261003/formal_EMA40000_same_AMP_fixture_actual_runs_v1/fixture200s_actual_launch.json)：实际GPU89246／824a99退出0，完整6000控制步／200秒。CPU92420／8e8952退出0，203项raw核对通过；[封存审计](style_improvement_20261003/formal_EMA40000_same_AMP_fixture_metrics_v1/raw/audit.json)保留全部24,000 actor-controls与96,000 actor-substeps：1跌倒／0静态接触／45个行人接触pair-controls／24路线点。OLD为0／0／0／21，未达到物理零事件要求，不晋升默认。
- [EMA常量归一化的配对评价修订v2](style_improvement_20261004/paired_validation_EMA_normalization_v2/README.md)：原EMA对包含常量的完整state dict做FP32 blend，纯标量复核确认1.5可能变为1.5000001192092896。新入口只接纳config或原配方精确支持值，记录实际delta／ULP且不回写buffer；四项派生逆还原、11函数源码／AST及独立审阅通过，root已封存。该修订的准备阶段未读取权重；实际v3配对已通过合法EMA常量接纳，逐值delta／ULP保存在报告，不回写buffer。原固定selection、loss、plan、metrics及终态门保持。
- [Diffusion验证回升的源码与原始资料核对](style_improvement_20261004/formal_diffusion_validation_source_review_v1/README.md)：核对BN micro200×2、whole-state EMA及五状态MSE边界；固定每域800次抽样监控与29,952唯一窗口、逐场景paired的分母不同。先执行已准备的固定主比较和全量诊断，当前未判定完整验证过拟合或改变训练目标。
- [全量验证损失诊断](style_improvement_20261004/full_validation_loss_diagnostic_v1/README.md)：CPU实际核对Maps10,566＋Interact19,386共29,952个唯一窗口、200个validation episode及原始文件字节；不读NPZ数组或模型。四项标量检查、Ruff及独立审阅通过并封存。模型执行待固定40k主比较完成；共同逐窗口t／noise、尾批加权、两域分别及整体窗口加权用于解释固定800窗口曲线，历史结果仅作诊断。
- [同AMP的SFM目标生成适配](style_improvement_20261003/sfm_same_AMP_staging_v1/README.md)：13项CPU几何检查和10项接口检查通过；[真实SDK-free资源检查](style_improvement_20261003/sfm_same_AMP_fixture_actual_runs_v1/SDK_free_preflight_actual_receipt.json)已实际退出0，确认8类资源、AMP62身份及SFM/checkpointNone/samples0接线。完整SFM＋AMP物理对照尚未运行，参数与未来目标构造属于本地补齐。
- [运行命令和默认权重](../FULL_PIPELINE.md)。
- [奖励0.5完整对照与失败原因](style_improvement_20261003/state_prior_pacer_reward_complete_review_v1/README.md)。
- [新Diffusion真实训练与配对指标](style_improvement_20261003/orca_maps_diffusion_actual_runs_v1/README.md)。
- [Office原始物理／支撑验证](style_improvement_20261003/local_Office_baseline_actual_review_v1/README.md)。
- [下一轮专家采样的唯一变量、ESS和具体命令](style_improvement_20261003/state_prior_stratified_training_preparation_v1/README.md)。
- [专家采样试训的实际退出与检查结果](style_improvement_20261003/state_prior_stratified_actual_runs_v1/README.md)。
- [状态判别器试训与完整训练实际状态](style_improvement_20261003/state_only_matched_q_actual_runs_v1/README.md)：训练、终态审计、全部22条动作及四人200秒物理均实际退出0；三个快照及完整捕获已封存。[风格及旋转分解](style_improvement_20261003/state_only_matched_q_tracking_metrics_v1/README.md)保留全部四条跟踪失败；闭环3次跌倒，native扩展不执行。
- [专家采样原生场景的实际评价](style_improvement_20261003/state_prior_stratified_native_actual_runs_v1/README.md)：四人／十五人200秒GPU、原始CPU及90D／45D均实际闭合；均0跌倒／0静态／0行人接触，到达38／146次。90D主MMD²为.07734562／.08017306（支持桶3／4），45D为.03760272／.03683265，覆盖.58/.46与.68/.57；共同速度桶对照保留，进展及部分风格未超过旧参照，不晋升默认。
- [同一份185源码上的新旧Diffusion物理配对计划](style_improvement_20261003/default_newDP_physical_preparation_v1/README.md)。
- [物理配对的实际状态与原始结果](style_improvement_20261003/default_newDP_physical_actual_runs_v1/README.md)。
- [10候选完整物理负结果与停滞取舍](style_improvement_20261003/default_NEW_samples10_actual_runs_v1/README.md)：进展改善伴随更多跌倒和行人接触，全部失败保留。
- [方差引导完整物理负结果](style_improvement_20261003/default_NEW_variance_actual_runs_v1/README.md)：数据核对通过，物理零接触门槛失败；同一新模型原引导70对方差315个行人接触pair-controls，到达数均9。
- [项目模块与已有协议结构](STRUCTURE_AND_EVALUATION_CURRENT_ZH.md)。
- [论文指标与本地统计范围](style_improvement_20261003/motion_evaluation_protocol_mapping_v1/README.md)：全22动作的帧加权、clip等权、step均值最大值分别报告；薄CPU汇总已实际退出0，不改变主指标或失败分母。作者最大值、rotation及导航rate的完整协议仍未知。
- [实际D6／exp-map旋转核对](style_improvement_20261003/actual_D6_rotation_representation_CPU_v1/README.md)：22条已录制validation、276,092个局部关节观测，当前表示最大角差7.99e-7rad，CPU退出0；保留现有旋转接口。
- [原始／可执行参考轨迹的真实CPU净空诊断](style_improvement_20261004/causal_reference_clearance_CPU_v1/README.md)：全部三个封存对照的12,500项plan/pair均保留，CPU实际退出0。执行目标前缀的peer unsafe相位OLD S1为239→128、NEW S1为676→625、NEW S10为864→576；NEW S10 static为61→76。约束后存在局部新增不足，总体结果有取舍；这些参考目标不是实际身体接触，也不能据此判定跌倒原因。
- [实际身体、参考目标与真实接触的完整CPU对照](style_improvement_20261004/actual_tracking_contact_CPU_v1/README.md)：真实会话17342退出0、1.57447秒，三组全部72,000 actor-controls及全部接触onset保留。S10参考unsafe576小于S1的625，但实际root unsafe1069大于794、peer279大于70；S10有118／279个接触pair-controls的末端参考代理safe。全帧跟踪均值.14138／.14190m近似，接触帧mean .43088／.23289m；仅观察关联，不判因果。真实接触汇总四物理子步，末端位置不能当作接触子步位置；CSV完整性、原相位及独立审阅通过并封存。
- [实际行人接触力完整分布](style_improvement_20261004/contact_force_severity_CPU_v1/README.md)：真实CPU工具059378退出0，三组每组36,000 pair-controls及原来源均核对并封存；OLD／NEW S1／NEW S10的>50N分别0／48／202，>1N条件P50分别null／57.73260／73.33865N。S10 P95／峰值较低但较强接触更频繁，既有物理负结果保留。50N受PACER启发，本地四子步body-pair法向max与PACER排除足部后的净力向量和范数不等价，不能作为论文碰撞率、终止复现或因果结论。
- [全程参考／身体速度响应诊断](style_improvement_20261004/reference_response_CPU_v1/README.md)：真实会话27686／d834e8退出0，耗时4.217851秒，三组全部72,000行及phase、末子步Pelvis、来源与分母核对并封存。OLD／NEW S1／NEW S10全程速度误差均值.231013／.258558／.245856m/s；新两组peer接触行人帧误差.427503／.467333m/s。全程误差减少与接触增加并存，不能宣告物理改善；没有估计真实动作延迟或判定原因。
- [社交AMP的3–15人原生场景入口](style_improvement_20261004/social_native_evaluation_staging_v1/README.md)：新producer7d2a5156／entrybe9f4126经root及独立完整审阅、compile／Ruff、逆还原／AST后设只读。沿用原native预检、绝对坐标、同一World／控制／接触／失败清理及4训练USD准入；仅补齐人数、观测与报告接缝。尚未执行CLI、NN、SDK或物理评价，不把既有旧策略的native结果当作新策略效果。
- [三通道社交栅格几何接口](style_improvement_20261004/social_raster_staging_v1/README.md)：已实现64×64×3高度／相对XY速度、同组与自身选择、稳定重叠归属及镜像Vy反号；4项NumPy检查和独立源码审阅通过并封存。20轨迹坐标＋栅格共12308维，与旧1044策略／normalizer不兼容；共享行人训练、社交网络接入及实际物理效果尚未完成。
- [Torch设备端社交栅格](style_improvement_20261004/social_raster_torch_staging_v1/README.md)：已实现同device Tensor数学、按scene分组及ego chunk，避免N×N×64²临时分配。真实CPU会话30104／2c34a8退出0，7项跨后端检查1.460秒；覆盖1024完整egos及混合256行，测试owner像素全相等、栅格atol2e-6通过。Ruff和独立源码审阅通过并封存；GPU等价、同步开销和实际吞吐尚未验证，不宣告提速或物理效果。
- [原始社交任务组装接口](style_improvement_20261004/social_task_observation_staging_v1/README.md)：已实现12308维rawtask、显式root/task/grid坐标与高度参考、单次批量terrain查询、镜像及仅替换候选20维轨迹前缀。真实CPU工具c35147直接退出0，4项检查/.012秒通过，CUDA前后未初始化；源与检查已封存。完整self/action镜像另见下方新接口；实际SDK当前状态／clock接线与物理训练仍待完成。
- [社交actor／checkpoint／value的未来CPU数值检查](style_improvement_20261004/social_inference_future_CPU_v1/README.md)：一次复用合成32-example fixture，覆盖B1／B4均值与57个实际动作、错误SHA提前拒绝、一次严格CPU mmap加载及候选20坐标的值／梯度。源码159f8741经编译、Ruff、root与独立完整审阅后设只读；四项尚未执行，合成root_z几何不代表真实SDK接线，不能作为效果证据。
- [中性SMPL社交CNN与actor／critic](style_improvement_20261004/social_policy_staging_v1/README.md)：已编写独立网络类，共享三层CNN、map256＋trajectory20及独立633输入actor69／critic1；critic-only value保留输入梯度合同。仅compile／Ruff和源码审阅，网络构建／forward／backward及三项测试尚未执行，不是已有训练权重；旧1044 checkpoint、RMS、trainer与inference不兼容。
- [八人共享World真实SDK探针](style_improvement_20261004/shared_group_world_SDK_probe_staging_v1/README.md)：源码已完成独立审阅、compile／Ruff并封存（87600495，root1f82af）；复用真实train2848、单App、末子步法向力及重置前全N归档，默认8个控制步。仅完成源码检查，尚未启动SDK；零动作和合成nextV用于接口验证，不代表策略或论文效果。
- [共享社交CNN／PPO＋AMP训练入口](style_improvement_20261004/shared_social_training_entry_staging_v1/README.md)：已连接固定train2848、单App、四人组World／normal views、真实专家与collector→update，源码f12a5004及两配置经compile／Ruff、独立审阅并由rootaa119d封存。八人32transition短检查与256人2000更新flat-floor本地pilot分别配置；外层AMP1／task.5，方向backwards_penalty明确为本地选择。尚未运行CLI／NN／SDK／训练，完整物理resume不支持，2000不是作者完整预算。
- [共享社交AMP完整接入合同](style_improvement_20261004/social_training_integration_v1/README.md)：集中256逻辑actor／64四人环境、物理双向映射与组间过滤、接触终止、reset前next value、history／curriculum同步、新RMS／checkpoint／mirror与value guidance边界。只写尺寸或scene IDs不能完成共享训练；实际SDK、训练及评价尚未接通。
- [四SMPL场景配置factory](style_improvement_20261004/group_articulation_scene_staging_v1/README.md)：按本机IsaacLab3 queued ReplicateSession接口生成四独立深复制articulation、N/4环境、local collision0／组间filter及global ground−1。2项stdlibmock检查/.001秒通过，Ruff／compile和独立审阅通过并封存；未调用真实SDK或验证碰撞。
- [四articulation张量桥](style_improvement_20261004/group_articulation_tensor_bridge_staging_v1/README.md)：已实现四[G,...]与canonical[N,...]状态／69动作双向排列，以及完整组reset转四组env indices。5项实际CPU检查/.010秒通过，CUDA未初始化，源与检查已封存。仅actor行映射；body／DOF命名、COMMON／SIMULATOR／XYZW转换与读写已由独立状态适配器连接源码，实际SDK物理仍待验证。
- [新社交训练配置](style_improvement_20261004/social_training_settings_staging_v1/README.md)：SDK／Torch-free校验12308特征、50镜像系数、完整task RMS、57有效动作及显式group／grid／height/self选择。4项stdlib检查/.001秒通过并封存；旧1044校验、warm start及protected suffix保持拒绝不兼容配置。新trainer／checkpoint factory已另写独立源码，尚未执行运行接入。
- [完整原始社交镜像](style_improvement_20261004/social_symmetry_staging_v1/README.md)：self357／task12308共同镜像，原action69映射、57有效维归约及外部50一次保留。4项CPU张量／梯度检查/.020秒实际5b2c14通过，root4f94bf封源与检查；没有策略NN、标准化分支或物理对称性验证。
- [独立社交CNN／PPO／AMP训练器](style_improvement_20261004/social_trainer_staging_v1/README.md)：新633-input CNN策略与D2019、四RMS、两个Adam及完整新checkpoint接口已连接源码；原PPO／D／GAE数学仅作明确factory／mirror接线。旧1044及warm start拒绝，fresh self／task／value RMS count绑定physical transitions。compile／Ruff、派生逆bytes／AST及独立审阅通过；尚未导入、构建NN、执行更新／resume或连接真实共享runner，不是训练成功证据。
- [四人共享物理驱动源](style_improvement_20261004/shared_group_world_staging_v1/README.md)：四子步120Hz、当前末子步net-normal、COMMON命名状态、整组末态／history／reference reset及规划中断守卫已连接代码；三项grid语义声明与配置强制匹配，实际quaternion保留。源码35097B/b608cae6经compile／Ruff和独立审阅，由root0780fd封存；四项fake-boundary CPU driver检查实际2d9f97→1de562退出0/.207秒，通过末步力／四子步功率／末态归档reset与中断守卫；检查及记录由root f3b74f封存。尚未执行SDK，不能称共享物理已验证。
- [社交策略rollout采样循环](style_improvement_20261004/social_rollout_collector_staging_v1/README.md)：保留Gaussian likelihood与施加动作、pre-history state_action奖励、四子步功率、reset前全员next value及各类独立flags；冻结四RMS，中断保留局部诊断并fault。源码27252B/f7d27270已审阅封存；四项fake-boundary CPU检查实际8dd007退出0/.034秒，检查完整collect奖励／末态nextV／RMS冻结和失败保留。未运行真实NN或物理；CLI／专家已另有独立源码接线，实际运行与checkpoint物理恢复仍未完成。
- [原生净法向力view工厂](style_improvement_20261004/social_normal_force_view_factory_staging_v1/README.md)：复用现有PhysX manager视图，核对全员24身体的真实路径／ReportAPI／数字env顺序后建立空filter view，dt1/120并拒绝停止或替换后的旧绑定。三项stdlib模拟检查69328e退出0/.012秒，root9c7538封存；没有实际SDK力值或reset同步验证。
- [AMASS专家批次接线](style_improvement_20261004/social_amass_expert_staging_v1/README.md)：复用原单次采样与专家公式、history10×195／下一帧命名PD代理／57有效动作；增加idle／实时限位／生成器守卫及采样后中断故障处理。四项合成CPU检查567839→33fabb退出0/.674秒，48bc4976源／baa3f62a检查与记录已封存。仅读取真实v7清单及bank stat，没有真实AMASS值、NN／物理或正q采样验证。
- [新社交策略推理接入](style_improvement_20261004/social_deployment_integration_v1/README.md)：仅从已训练81a实例复制encoder／actor／两组RMS和57有效mask，支持任意B的357／12308输入及确定性mean→clamp动作；源码05261849已compile／Ruff、独立审阅并由root6e22a2封存。尚未运行NN；checkpoint reader另已补独立源码，实际加载及22动作／四人200秒运行仍待完成；两项评估producer源码已补，native接线仍待补齐；逐次完整权重检查的真实延迟尚未测量。
- [新社交检查点推理加载器](style_improvement_20261004/social_checkpoint_inference_staging_v1/README.md)：强制完整SHA及独立只读文件、Linux FD绑定CPU／mmap／weights_only加载、构建前strict新schema，并核验全模型与四RMS后仅保留actor路径。源码cd227f5d经compile／Ruff、独立审阅及root6707b2封存；尚未加载权重或执行NN。计数为checkpoint声明，不代表独立物理训练观测，Adam／replay／RNG／SDK恢复不由此接口验证。
- [评估用同帧社交原始观测](style_improvement_20261004/social_evaluation_observation_staging_v1/README.md)：固定原N1／shared N4 World，一次owned COMMON状态生成357／12308与实际clock／quat／heading诊断；明确reset通知、同episode真实fall的奇异朝向延续及静态sampler语义。两项源码Required已修正，2379d658经compile／Ruff、独立审阅与rootb992fb封存；没有NN／SDK／数据执行。
- [新社交AMP完整22动作评估入口](style_improvement_20261004/social_tracking_evaluation_staging_v1/README.md)：严格cd227权重加载及2379观测接入原N1／frame9／真实future／120Hz四子步与原误差算法，固定全22／11982帧并保留失败；异常另存部分捕获，不补帧或参与成功聚合。源码79fd24b2经精确逆派生／AST／compile／Ruff、独立审阅并由roote0dba3封存；没有加载权重或执行跟踪。
- [新社交AMP四人导航入口](style_improvement_20261004/social_navigation_evaluation_staging_v1/README.md)：独立evaluation-only配置、真实共享训练report＋模型／USD准入、同帧357／12308输入与partial reset通知；保留原Diffusion／guide20／replan15／causal reference／6000控制及接触采集。源码c31c8739经compile／Ruff、精确逆派生和独立审阅，root e207b1封存；尚未运行preflight／NN／SDK／200秒评价。
- [冻结critic可微价值接口](style_improvement_20261004/social_value_adapter_staging_v1/README.md)：仅复制已构建的新encoder／critic／三个RMS，保持候选20维local米制轨迹的输入梯度并反归一化value，当前self／raster固定。源码12573B/9875c58b已compile／Ruff、独立审阅与封存；forward／梯度未执行，positive count不证明critic质量或避碰效果，不自带Diffusion指导。
- [四articulation状态读写适配器](style_improvement_20261004/group_articulation_state_adapter_staging_v1/README.md)：明确24 body／69 DOF名称映射、COMMON转换、当前link状态独立快照、PD位置目标及完整组的部分reset写入。5项fake-asset CPU检查/.120秒通过，实际工具d7ee93退出0；源码与检查由root 6e49d6封存。未调用SDK或推进物理，实际接触、功率、history／clock reset仍未接通。
- [八人／两组真实PhysX检查入口](style_improvement_20261004/group_articulation_physics_smoke_staging_v1/README.md)：已编写独立单App入口，检查实际名称／接触路径顺序、同组正接触和跨组原始力严格为零，保存两阶段全部子步。compile／Ruff及源码审阅通过，root 6e49d6封源；尚未启动SDK，不能称碰撞隔离已经验证。
- [排除足部的净法向接触判据](style_improvement_20261004/nonfoot_net_normal_contact_staging_v1/README.md)：按实际body名字排除四个足部，对剩余净法向力向量求和再判断norm>50N及progress>1。4项纯CPU张量检查/.025秒实际61ce0b通过，源／检查4ff6b0封存；输入要求同物理帧all-partner net normal，历史peer-pair最大值不能替代。当前World与normal factory已连接采样／early termination源码，实际SDK时序和Gym摩擦通道等价仍未确认。
- [当前帧全伙伴净法向力采样器](style_improvement_20261004/social_net_normal_sampler_staging_v1/README.md)：现有每actor24传感器view的路径／名称／分组预检后，调用get_net_contact_forces(physics_dt)，立即复制再转COMMON身体顺序与canonical actor行。4项fake CPU检查/.027秒实际db516f通过，root052ba9封源及检查。未调用原生SDK、验证floor力或真实时钟；现有per-actor24接口已由normal factory与World连接源码，实际终止和规模开销尚未验证。
- [四人分组与reset掩码](style_improvement_20261004/social_group_reset_staging_v1/README.md)：保持256扁平actor行、64组及每组2 AMASS＋2 procedural；真实fall保持terminated，连带组成员作为truncated而不伪称跌倒或time limit。5项NumPy检查、Ruff和独立源码审阅通过并封存；本组件仅布局／掩码算法；过滤、runner与episode/history/plan reset已由独立World连接源码，实际共享物理仍待验证。
- [BABEL真实训练语义标注连接](style_improvement_20261004/babel_train_semantics_v1/README.md)：发现本地既有完整release ZIP，纠正此前“未发现BABEL”的历史资源盘点。仅decode train／extra_train，与1853训练parent转换JSON精确连接；真实会话68758／a88240退出0，全部2848原行／顺序及引用核对通过，37.5MB独立标注文件已封存。1646段匹配到标注，其中844段有连续帧标注时间覆盖；1202段未匹配，另802段匹配但无完整timed覆盖。多动作／多标注者分歧、空category及一条反向时间区间均保留，不投票、不补造；现有v7／style5／采样／权重未改，尚未训练语义风格模型。
- [共同五秒时域的原始／可执行参考候选选择](style_improvement_20261004/reachable_candidate_selection_staging_v1/README.md)：新增独立可执行评估入口，保留单次原S10 NN生成、原52点引导及全部候选；matched_raw50与reachable50仅评分几何不同，均采用同50未来节点和20／5／10成本。8项几何＋2项callback CPU检查、两模式真实资源预检及独立源码审查通过，7片段隔离runner接线完成。[候选／物理raw审计及补充计划v2](style_improvement_20261004/reachable_candidate_selection_staging_v1/commands_and_audit_plan_v2.json)已独立审阅并封存，绑定场景SHA、全候选及6000控制步并单独显示真实跌倒／接触；新reader和物理实验尚未执行。历史52点S10不充作matched50对照，同seed也不保证闭环分化后的候选银行相同。
- [实时SMPL RGB／深度采集入口](style_improvement_20261004/live_pedestrian_capture_v1/README.md)：复用原有实时蒙皮和唯一物理驱动，新增公开采集入口及深度文件／状态不变性审计；真实SDK-free资源预检和33项CPU接口测试通过。真实RGB-D渲染、帧同步、标定与新增入口的物理诊断尚待GPU串行验证。

## 姿态误差的已有完整分解

以下复核已有22动作报告，未新增NPZ或模型计算；每组11,982个原metric frames、24 body，全部失败保留。

| 模型 | world主旋转误差（rad） | 去自身heading后的body误差 | parent-local关节误差 |
| --- | ---: | ---: | ---: |
| [默认62c4](style_improvement_20261002/backward_penalty_tracking_decomposition.json) | .861454 | .573434 | .301823 |
| [q935专家分层](style_improvement_20261003/stratified_prior_metrics_v1/tracking_decomposition/decomposition.json) | .782873 | .526608 | .272275 |
| [q936状态判别器](style_improvement_20261003/state_only_matched_q_tracking_metrics_v1/decomposition.json) | .751018 | .521384 | .274913 |

parent-local的24均值包含Pelvis identity，属于额外诊断，不替换world主指标；没有相位搜索或去掉失败。作者旋转坐标及聚合未完全公开，不能挑较小的诊断数字宣告超过论文。默认62c4这份旧报告／捕获目前仍有写权限，不称全部只读封存；q935／q936已封存。

## 下一步

官方ORCA两域完整时间窗口、200×2梯度累积、EMA的40,000次更新已完成，实际50502／f10f35退出0；40k独立终态372,121,656B、SHA4581f29d…已设只读。初次77025检查因JSON将两配置tuple转list而退出1；[显式类型恢复](style_improvement_20261004/formal_terminal_configuration_adapter_v1/README.md)只修正这两个容器类型，实际31365／999fed退出0，原严格raw／EMA／Adam、语料及生产加载检查保持。完整实际报告明确记录adapter来源。14k及15k–39k独立历史副本已保存，watcher39421／bc5843实际退出0；仅做header与字节归档，未评价历史权重、不用test选优。正式训练退出由独立f10f35证明，40k header不代替它。

固定配对已闭合0：[实际comparison](../output/diffusion_amp/formal_EMA40000_vs_63e_validation_readonly_baseline_v3_retry1/comparison.json)与[实际CPU owner](style_improvement_20261004/paired_validation_readonly_baseline_v3/actual_CPU_root_owner.json)已封存。主200窗口ADE 1.160940→0.536668m，FDE 2.733504→1.313420m；Maps障碍点742→198／5000、障碍轨迹44→22／100。行人碰撞点合并306→320／10000，Interact155→198／5000，主clean loss .057463→.091358；不能声称全面改善。零当前邻居诊断另504窗，和主集合重叠20窗。原63e mode600／nlink2保持，实际只读副本字节全同；不改固定窗口、数学或种子。这是5秒开环评价，不能推出200秒闭环或动作风格效果。

[同AMP的终态接线与raw指标适配](style_improvement_20261003/formal_EMA40000_same_AMP_fixture_source_plan_v2/README.md)已实际接纳终态fbe188；资源预检和185启动前全SHA均通过。[四人200秒真实GPU89246](style_improvement_20261003/formal_EMA40000_same_AMP_fixture_actual_runs_v1/fixture200s_actual_launch.json)已实际退出0，完整捕获与203项CPU raw核对完成；F1／static0／peer45／routes24，物理要求未通过。[同AMP历史风格对照](style_improvement_20261004/formal_EMA40000_style_historical_comparison_v1/README.md)已整理并封存：OLD→EMA的90D MMD² .07238060→.05592538、45D .02015677→.02297688，K100覆盖.72→.71，两个足速度代理均变大；不能称整体动作风格更优。社交CNN／PPO／checkpoint／value共10项实际CPU检查随后全部通过；真实8人SDK全部64子步接触检查通过、8个contact弱引用已全部释放，但实际进程仍退出139。精确管理器规范化已实测恢复Base正常清理，Core及两个Timeline缓存也已释放；v14已实证4个Articulation／4个Data字段持有后端view；v15释放这八个字段后contact及backend弱引用都为0，但实际仍退出139。v16／v17定位到core插件的C退出阶段后，独立官方6.0.0原生修复已在v18真实退出0，加载来源和原64子步检查通过，35个安装扩展文件SHA保持。共享World及训练入口正迁移到这一已验证扩展，随后运行固定train2848的真实World／短训练；尚未开始新训练。全量29,952唯一窗口损失及历史诊断仍后置。SFM同AMP对照已通过资源预检，待GPU串行执行。TRACE允许零近邻样本，本次已恢复原筛选排除的4,458个训练窗口（约1.91%，真实train分割计数）。

AMP分层采样及状态判别器消融的完整训练／评价均已完成；状态判别器闭环3次跌倒，90D／45D为 .06529463／.03235954，不进入native扩展。下一轮[状态判别器物理步后时序对照](style_improvement_20261003/state_only_q_poststep_timing_preparation_v1/README.md)已有三片段独立改动、独立审阅及一次有界CPU时序检查；[实际937文件隔离副本](style_improvement_20261003/state_only_q_poststep_training_preparation_v1/README.md)已完成配置／资源／导入检查；[完整checkpoint审计入口v3](style_improvement_20261003/state_only_q_poststep_training_audit_preparation_v1/root_source_freeze_finalization_v3.json)已独立审阅并封存，准确绑定真实工具会话和CUDA环境；SDK快速退出的日志标志按真实文件核对，原退出、导入及数值验证保留。[GPU背景检查](style_improvement_20261003/state_only_q_poststep_GPU_background_preparation_v1/registration.json)已准备，启动前才实际观察。新训练及模型payload审计尚未执行；[完整22动作与四人200秒评价接线](style_improvement_20261003/state_only_q_poststep_evaluation_source_plan_v1/README.md)已固定原协议，未来权重／审计SHA仍为null。[未来四人200秒原始CPU审计入口](style_improvement_20261004/state_only_q_poststep_fixture_audit_v1/README.md)已补齐并封存；复用原物理计算，仅补实际终态、预检和工具闭合来源接纳。pending模板被真实CPU检查拒绝，尚未审计未来模型或捕获。采用相同q、奖励、PPO、PD及旧Diffusion评价，fresh初始化。公开AMP／PACER为此机制提供依据，论文式2仍写状态历史＋动作，该消融不冒充严格作者模型。新Diffusion的[方差引导](style_improvement_20261003/default_NEW_variance_actual_runs_v1/README.md)和[10候选对照](style_improvement_20261003/default_NEW_samples10_actual_runs_v1/README.md)均未通过物理要求。各实验依据公开PACER／TRACE机制，不冒充Nav作者未披露的参数；[AMP动作代理与旋转目标的源码研究](style_improvement_20261003/AMP_action_rotation_research_v1/README.md)记录目标差异。

已根据TRACE/PACER与Nav论文继续补齐[两模式候选评分实验](style_improvement_20261004/reachable_candidate_selection_staging_v1/README.md)：优先固定候选／时域／成本后检查实际reference几何，保持原NN生成、AMP与完整失败分母。几何代理有取舍，不预设物理改善或直接晋升默认。

源码核对还确认：当前62c4完整checkpoint带有critic与value RMS，部署的[AMP推理接口](../CrowdSim/humanoid/runtime/amp_inference.py)验证后只保留actor，没有执行value guidance。[TRACE/PACER官方value接口](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/utils/guidance_loss.py#L816)要求固定当前身体／环境，仅让候选轨迹进入critic。本地1044维任务为20个XY坐标＋1024静态terrain值，没有作者动态peer高度／速度地图或shape条件；本地critic的奖励和课程也不同，不能把它视为作者避碰概率模型。源码支持后续匹配本地critic的独立评分接口研究，当前尚未实现或运行。

进一步逐项源核发现：既有q训练把256个单人放在相互碰撞过滤的独立env中（脚本默认64个，CLI可指定），`shared_world`只随evaluate步骤开启；mixed/AMASS课程还显式拒绝shared模式。Diffusion虽获得peer历史，AMP的357维自身状态＋1044维任务没有peer动态通道，mixed课程训练重规划也绕过Diffusion。现有左右镜像正则已实现，社交信息及真实组内交互训练是另一处待补缺口。新增栅格／任务组装已通过有界CPU检查，独立actor／critic CNN已写源但尚未执行。后续依据新接入合同完成共享物理、reset／curriculum、normalizer／checkpoint及PPO镜像，再实际训练／物理比较；缺口本身不证明既有接触的原因。

NavIsaaclab2.0的本地干净提交d742f0c5仍是[SFM到MaskedMimic](https://github.com/yh83305/NavIsaaclab2.0/blob/d742f0c5b421bbc281e2112e7e50427e0336c16a/CrowdSim/nav_manager.py#L751)，未找到论文Diffusion／TRACE版本或S>1候选排序。官方TRACE的终态filter没有乘cfg.weight，不能据此确认Nav作者过滤方式；本地共同五秒实验仍固定两臂20／5／10，不混入另一个变量。上述是源码界限，不是新增训练或物理效果结论。

历史冻结实验保留原位置、原绝对路径和原始失败文件；本页集中当前状态，避免把历史训练中／prepared-only登记当作实际完成记录。训练reset、离线碰撞、几何代理和PhysX实际接触各有不同语义，报告明确保留。封存34条测试输出未用于本轮优化。
