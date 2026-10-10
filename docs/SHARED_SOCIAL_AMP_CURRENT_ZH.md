> 2026-10-10：项目已重命名为 **Socially-aware_Indoor_Navigation_Benchmark**，核心源码迁至 `src/`。
> 入口与目录说明见 [项目结构](PROJECT_STRUCTURE_ZH.md)；历史路径通过兼容链接保留。

# 当前状态：Diffusion + AMP 行人模拟

## 2026-10-10：速度与路线优化、四段新60秒视频已完成

本轮AMP真实训练4,587,520次交互，选用655,360次交互的小学习率候选用于YOHO/NewMetro，窄场景保留已验证AMASS v9父权重。四场景4/4/1/1人各完整60秒，11项物理门槛、姿态、巡航速度及10秒空间停留检查通过；所有30Hz实际速度样本不超过1.5m/s，巡航均值0.721–0.864m/s。换向和避障减速保留。四段新1800帧/30fps视频完整解码与全帧FK核验通过。

新权重、配置、逐场覆盖范围、证据及复现命令见 [本轮完整报告](/project/EXP/SangforAndSaut/ProtoMotions/docs/FOUR_SCENE_SPEED_20261010_ZH.md) 和 [发布索引](/project/EXP/SangforAndSaut/ProtoMotions/docs/four_scene_speed_release_index_20261010.json)。本轮训练有1次探索跌倒，交付物理源零跌倒/零重置/零>1N接触。当前没有覆盖全部场景，也未加入机器人或证明达到作者整体效果。以下历史记录保留原时间与结论。


## 2026-10-09：四场景 60 秒步态优化及录像已完成

固定2848真实AMASS新增768更新/3,145,728物理转移，训练零跌倒。四场景各完整60秒，11项物理门槛和独立姿态筛查通过；四段1800帧/30fps/60秒原生录像完成。成对步态与独立测试结果、实际人数组4/4/1/1及剩余差异见 [本轮报告](/project/EXP/SangforAndSaut/ProtoMotions/docs/FOUR_SCENE_AMASS_GAIT_20261009_ZH.md)。这是本地工程复现，未证明作者八场景总体效果。以下为保留的历史结果。

## 2026-10-09：YOHO 姿态与速度优化已完成

固定2848真实AMASS新增128更新/524,288物理转移。四人YOHO完整200秒、零跌倒、无初始化后重置、实际退出0；11项本地物理门槛和行走姿态筛查全部通过，起步与全程无手臂过水平动作。30秒30fps真实物理状态录像已完成；全900帧FK误差小于1mm。

成对60秒AMP比较中，行走侧倾p95由6.795°降到4.862°，扭转由8.004°降到7.443°；摆臂幅度接近，风格MMD与支撑脚原点速度代理退步。Diffusion公共网络权重保持，推理引导与有界轨迹优化有调整。当前只是单场景/四人/seed7工程复现。

详见 [YOHO结果与复运行](/project/EXP/SangforAndSaut/ProtoMotions/docs/YOHO_DIFFUSION_AMP_ZH.md) 和 [来源证据](/project/EXP/SangforAndSaut/ProtoMotions/docs/yoho_diffusion_amp_evidence_index.json)。以下为保留的历史结果。


## 2026-10-08：真实 Warehouse 新一轮 Diffusion+AMP 优化

已完成固定2848 AMASS / 512次微调 / 2,097,152实际物理样本；四人原始Warehouse完整200秒，零跌倒、零>1N静态和行人接触，54个路线节点，实际退出0。30秒30fps室内+跟随视频已完成。姿态稳定性和部分姿态指标改善；按速度条件的动作MMD与覆盖仍退步。

详见 [Warehouse 当前结果](WAREHOUSE_DIFFUSION_AMP_ZH.md) 与 `warehouse_diffusion_amp_evidence_index.json`。本次一场景工程复现不等于作者完整八场景协议。此前AMASS anchor512和旧Social AMP结果保留为历史对照。


## 本轮完成：固定2848 AMASS微调与实际场景验收（2026-10-08）

本地AMP已完成**128环境、512更新、2,097,152真实物理转移**，2848训练motion ID均进入专家采样。
权重、冻结归一化统计和完整导出推理均通过独立CPU核验。
四人相向200秒及同向绕障60秒分别普通实际退出0，各通过10项独立物理门槛，均零跌倒、零行人/障碍法向接触。
初始化后重置0次；两场共7800控制步/31200物理子步，不包含训练重置。

- 微调权重：`data/pretrained_models/local_amass_finetune/anchor_20261008/Humanoid_AMASS.pth`。
- 明确选择本地微调配方：`configs/diffusion_amp/trace_pace_amass_finetuned.json`。
- 训练、场景、独立风格/步态、录像和SHA来源：[完整操作与证据](AMASS_FINETUNE_ZH.md)。
- 录制读取同一次真实PhysX状态，逐帧标注源时间；使用IsaacSim6 RTX纹理SMPL，可复录完整200秒。
- 相向场景47路线节点、23通过代理；脚底支撑99.879%以上，支撑脚原点速度代理0.06442m/s。
- 风格MMD及K50/K100覆盖略差于公开基线，**没有证明风格改善或达到原论文整体效果**。

这只是seed7、两种四人平地实体障碍场景的局部质量验收；不是原论文8场景/3–15人的总体认证。
同向场景原退出-11及早期录像失败完整保留，复跑和正常退出录像分别使用新目录。
完整物理/录制父进程收据而不是子进程打印passed决定是否验收。

## 此前公开权重基线与历史记录

下述“最新/当前/未微调”对应此前公开权重基线，局部微调终态以本页顶部和AMASS说明为准。


更新：2026-10-08。当前目标是参考 Trace and Pace，实现可核验的真实物理行人规划与动作生成，并与 NavIsaacLab 方法对应；不要求指标超过论文。

## 训练终态

- episode2000 训练已于 10:54:27（上海）完成：2000 更新、16,384,000 物理转换，训练和 CPU 完整性审计均实际退出0。
- 全部22条独立动作跟踪为0/22成功、6条发生跌倒；导航短测实际退出-11，200秒阶段未启动。这个候选没有提升为部署权重。
- S3-R 已完成12500有效更新；其已完成独立评价与早前失败记录分别保留。零跌倒仍伴随路线停滞，不能认定已达论文效果。

## 本轮可执行方法

新增[TRACE + PACER 实现与验收说明](TRACE_PACE_REPRODUCTION_ZH.md)：`scripts/run_trace_pace.py`。
使用作者公开 TRACE100步DDPM 与 PACER AMP，实际身体反馈、定时路径点、地图/邻居引导、候选联合检查与显式安全制动。AMP输出真实PD动作，未播放姿势或重置人体来隐藏失败。

本地固定AMASS v7 的2848条真实动作保留；该入口使用公开权重，尚未在本地2848上重新训练。Nav作者定制权重与八场景清单仍缺失，公开基线和本地Nav历史+动作判别器支路分别标记。

## 最新可核验结果

最新`trace_amp_bounded_reciprocal_200s_v3`：6000/6000控制、24000物理子步、普通退出0，
48项执行源码/输入不变；全部10项本地路线/物理门槛通过，48节点、零跌倒、零实测行人/障碍接触，
25次相向通过代理，最小根部间距1.585m，最长低速0.567秒。
至少一脚支撑子步99.875%–99.971%，平均载荷约621N/人/体重比约1.0，支撑交替中位1.76–1.875Hz。
完整23996风格区间45D MMD²=0.03180、90D速度分桶MMD²=0.07047、K50/K100覆盖0.52/0.39。
49/402次规划有解析制动，平均规划1.594秒；瞬时根部速度最高2.030m/s，高于1m/s规划上限。
人体没有初始化后重置。当前验证只覆盖seed7的受控四人靠右往返场景；相同种子复跑不逐位一致，
实际掉头仍有短暂反向移动，不认证全部场景或主观自然度。原始证据及资源SHA在TRACE说明。


| 实验 | 实际终态与独立结论 |
|---|---|
| 原PACER时钟，单人12秒 | 退出0，零跌倒；路线误差0.216m，停步0.0355m/s |
| 原时钟，同向环行200秒 | 退出0；24节点，零跌倒/实测接触，所有实际轨迹来自DDPM |
| 作用域修正，靠右200秒v2 | 退出0；6000控制/24000物理子步，10项独立本地门槛通过，24节点，零跌倒/实测接触；脚部支撑≥99.96%，体重载荷约1.0倍，左右支撑交替约1.76Hz |
| 无交通约定交叉场景 | 60秒零节点后退出1，持续停滞；不算可靠导航 |
| 旧靠右200秒v1 | 子进程日志完成但父进程退出−11，不准入成功 |
| 往返走廊60秒v1 | 退出0，但两条12m目标被全局成本绕成40m、实际相向通过0；不算会车证据 |
| 窄走廊代价修正 | 全局往返路径12.62–13.24m、相向路线分离；75项CPU检查及Ruff通过，60秒普通退出0/10项门槛通过/12节点/8次相向通过/零跌倒接触 |

作用域修正先释放已结束的控制帧/locals缓存/闭包临时对象，再正常关闭SDK，
7类对象以弱引用核验、同步CUDA；没有强退制造成功。完整200秒退出0仍不能证明所有原生SIGSEGV根因已消除。
靠右200秒v2的完整23996风格区间45D MMD²=0.03580、90D速度分桶MMD²=0.08398、
K50/K100覆盖0.34/0.28，属于本地冻结参考指标。只记录两次局部相向通过，多数交通实际环行；
后续因此修正全局代价并增加同侧走廊与最大长度测试。

当前仍是受控场景/seed7证据；最新平均规划约1.59秒，未验证实时。
接触斑滑移、真实摩擦、人类观感、多种子与3–15人、多地形/原八场景协议仍需评价。
固定2848的候选训练完成但可靠性未通过；本轮公开PACER权重没有在本地2848上微调。
全部失败、原始接触与实际父进程收据保留。详情与复现命令见TRACE说明。

## 历史记录

以下保留此前执行、审计与诊断的原始叙述。文中的“正在训练”“等待”“当前”等描述对应各自记录时间；本页顶部及 TRACE 说明给出最新终态，旧失败不会改写为成功。

---

# 共享行人 AMP：当前可执行入口

目标运行环境：当前固定的 Isaac Sim 6／Isaac Lab 3 开发环境。
训练数据保持 AMASS v7 的 2,848 条真实动作片段；这是本地重建清单，
作者未公开的动作 ID 尚未恢复。

**后续episode候选已进入2000更新训练**，见[本轮执行与复现说明](style_improvement_20261008/shared_social_episode_after_recovery_v4/README.md)。
2026-10-08 02:48:35（Asia/Shanghai）：8人／256人训练及两项CPU审计实际退出均0；终态清理成功。
owner473700／正式SDK487451正在执行2000更新；2026-10-08 06:10:58宿主实际核对：
819／2000更新、6,709,248转换，完整日志连续且有限，进程命令／父子关系／owner启动身份保持。
最近20次平均14.06秒／更新，剩余训练约4.61小时；尚无完整训练终态。
首个[500更新检查点CPU恢复完整性核验](style_improvement_20261008/shared_social_episode_checkpoint500_CPU_v1/README.md)
实际退出0：原严格加载器读取完整模型／两Adam／四RMS／65536回放后逐值一致，651041646B独立副本保持。
未恢复全局RNG／物理World／课程，未重启训练；不能代替2000终态或质量评价。
新增同一500更新归档的[关节速度](style_improvement_20261008/shared_social_episode500_joint_velocity_CPU_v1/README.md)／
[位置旋转向量](style_improvement_20261008/shared_social_episode500_joint_pose_CPU_v1/README.md)批量核验，两个CPU进程实际退出均0。
完整8192状态、96重置控制步保持；原时钟／PRE←POST连续关系核对，AMP57逐值等于实际原生速度。
全69子坐标角速度最大差6.26505e-6 rad/s；188416关节旋转比较最大差6.26320e-7 rad。
4项独立坐标测试／4个subtest及Ruff通过；本批未发现关节参数迁移差异，不能推广为全训练或独立风格质量通过。
修复专家／收集器旧World来源绑定，并实际复现CUDA float32 `150/30=5.0000004768`造成五秒查询失败。
使用显式可逆的float64除法→float32舍入，保持严格边界；CPU／CUDA真实2848的8人和256人资源检查实际退出0。
10项CPU测试、21个subtest、四项冷预检及Ruff通过；57来源／7输入固定。
两次SDK失败均0更新／清理成功／实际退出1，原失败目录保留；两轮物理预检通过后自动训练及独立评价。
独立[冷核验](style_improvement_20261008/shared_social_episode_after_recovery_v4/smokes_external_verification.json)实际退出0，
原参考配置、实际专家来源、保存provenance、只读输出及57来源／7输入匹配，未启动Torch／SDK。
该候选从新模型训练，与下方已完成12500的S3-R恢复结果分开；尚未有候选质量改善证据。
新增只读查询`scripts/shared_social_episode_status.py`，6项检查／7个subtest、Ruff及宿主查询实际退出0；
正式57来源／7输入冷复核实际退出0，训练来源保持。
[同预算独立比较](style_improvement_20261008/shared_social_episode_comparison_preparation_v1/README.md)
登记／冷预检与8项CPU检查／11个subtest实际退出0，固定S2与候选各2000更新／16.384M转换。
CPU观察器521166已等待真实owner完成，会自动比较完整validation22、冻结风格与路线／物理结果；
06:14:29宿主核对实际命令／原启动ticks2110428保持，该原观察器仍存活等待。
目前比较尚未执行，test34不用于选择，作者八场景协议尚未恢复。
新增[episode终态原生DP/SFM入口](style_improvement_20261008/shared_social_episode_native_preparation_v1/README.md)：
9项CPU检查／3个subtest、Ruff、登记及N3／N15两项来源预检实际退出0。
原四地图156个cohort／144准入／12拒绝完整保持，预登记两臂同AMP与200秒预算。
通过独立模块绑定本轮真实完成记录，原旧S3证书准入保持；未完成训练仍被拒绝，未启动新SDK。
原生物理／初态配对／数组与风格实际评价仍待执行，本地四图不等于作者八场景。
新增[完整78个原生源码接口实际检查](style_improvement_20261008/shared_social_episode_native_interface_execution_v1/README.md)：
N3–15×seed7／19／42×DP／SFM的真实CLI均实际退出0，人数／种子／源码逆转换核对通过。
未加载Torch、候选权重或启动SDK；SDK与raw审计仍需真实工具终态，不生成替代chunk。

新增[S3-R时间历史／动作条件CPU诊断](style_improvement_20261008/shared_social_S3R_temporal_discriminator_CPU_v1/README.md)：
实际退出0，原10帧／2019维判别器和奖励函数评分73728对输入，参数与RMS保持，未启动CUDA／SDK。
物理策略原历史平均AMP奖励.409293，重复当前帧为2.204078；过去帧逆序／置换有响应，动作错配响应较弱。
重复当前帧保留速度及动作，可能物理不一致，不能等同真实站立或认定路线停滞因果。
全部保存分数统计逐值核对；6项CPU检查／5个subtest和Ruff通过。
本批不是独立风格评价，当前2000训练来源／配方保持；待真实终态质量结果后决定下一候选。

新增[静止姿势／刚体平移CPU对照](style_improvement_20261008/shared_social_S3R_kinematic_discriminator_CPU_v1/README.md)：
实际退出0，49152对输入；将速度置零后，策略窗口平均AMP奖励由.409293升至3.074379，刚体平移为2.990079。
原历史／重复帧评分逐值保持；8192个真实身体快照的静止重编码差0，平移差仅位置舍入3.49e-6。
6项CPU检查／6个subtest与Ruff通过，全部新保存统计逐值核对；未启动CUDA／SDK。
运动学一致不证明物理支撑／力矩平衡；不将真实AMASS合理停顿一律视作负样本，当前训练保持。

新增[固定2848完整AMASS活动统计](style_improvement_20261008/fixed2848_activity_CPU_v1/README.md)：
CPU实际退出0，523272源帧／494792合法专家中心全部保留，源文件／片段边界／30Hz时序核对。
理想均匀片段→合法中心先验下，当前低根部速度占22.4277%，十帧同时低速／低关节RMS／低角速度占14.0034%。
全中心等权对应20.8158%／12.5499%，不同于真实训练曝光；本地阈值不是站立语义或作者协议。
5项检查／3个subtest、Ruff及保存数组逐值复算均通过，57训练来源／7输入保持。
实际bank已声明child_joint，corpus无旋转转换；未加载模型、启动SDK或重启训练。

2026-10-08复查[论文IV-B](https://arxiv.org/html/2606.26265v1#S4.SS2)：外层奖励明确为AMP＋0.5×任务，当前配比符合此式。
仓库保存的[PACER示例](../third_party/pacer/pacer/data/cfg/train/rlg/amp_humanoid_smpl_cnn_task.yaml)则为0.5／0.5、判别器奖励scale2；
该示例不能直接作为NavIsaacLab作者配置。当前softplus／scale2变换仍是复现假设，不能仅凭外层权重认定奖励尺度完全对齐。
论文未给出完整判别器奖励变换，后续尺度消融须登记独立配方，并以本轮完整终态评价作比较；当前训练继续。

新增[S3-R冻结奖励平衡CPU核对](style_improvement_20261008/shared_social_S3R_frozen_reward_balance_CPU_v1/README.md)：
CPU实际退出0、Ruff通过，完整8192策略／4096专家窗口保持；未加载Torch或SDK。
静止对照平均AMP增幅2.665086，98.9380%策略窗口超过加权轨迹项增益上界0.5，98.2910%超过轨迹＋方向上界1.0。
没有给能量项推导上界，也未计算实际反事实动力学、未来回报或当前候选评分，不能认定总奖励支配或训练停滞因果。

**当前状态：S3-R训练及[独立评价续跑已完成](style_improvement_20261008/shared_social_S3R_navigation_retry_v1/README.md)，尚未达到论文完整效果**。
2026-10-08 01:34:43（Asia/Shanghai）：无GDB导航200s与原CPU数组审计实际退出均0，清理成功。
复用已成功的actor／跟踪风格／导航短测，与新导航组合完成评价；旧post退出1／导航-11保持，SDK根因仍未定位。
[严格独立对比S2](style_improvement_20261008/shared_social_S3R_comparison_v1/README.md)
CPU实际退出0，72份资源／捕获／数组哈希核对。跟踪0/22→14/22，位置.918030→.276747m，旋转1.412125→.749612rad；
导航跌倒9→0、行人接触32→0、路线节点1→4。路线推进仍不足，K50／K100覆盖下降；默认权重保持。
四个90D速度桶样本数／参考索引匹配，等权MMD²=.0684345→.0471716；其余桶unknown，不计入分数。

以下保留首次正式v2执行与故障诊断的事实：
2026-10-08 00:50:15（Asia/Shanghai）：正式owner实际记录pipeline退出1、navigation200s退出-11。
严格CPU actor读取实际256行、N1生命周期、全部validation22、冻结风格及导航smoke均实际退出0。
完整11982帧：14/22成功、零跌倒，原世界坐标位置0.276747m／旋转0.749612rad。
45D MMD²=.0230231，K50/K100覆盖=.58/.52；相对S2距离降低、覆盖缩小，不能声称全部风格指标改善。
首次200秒数组与结果写出后发生段错误；该v2进程的raw-CPU审计未启动，其完整退出准入仍未通过。
[独立失败数据取证](style_improvement_20261008/shared_social_S3R_failed_navigation_forensics_v1/result.json)
CPU实际退出0：0跌倒／0接触、4路线节点，低速<.1m/s区间占84.81%；最后150秒全部低速。
这些是失败进程保存数组的核对结果，没有改写-11退出或认证SDK成功。
[完整跟踪分解](style_improvement_20261008/shared_social_S3R_tracking_diagnostic_v1/README.md)
CPU实际退出0：根部XY=.209620m、根部旋转=.523158rad；原始聚合逐值保持，诊断不替代论文评分。
同一6000步的[GDB诊断owner](style_improvement_20261008/shared_social_S3R_navigation_gdb_v1/completion.json)
于01:12:47实际退出0，日志明确正常退出，未复现故障，根因未定位。
01:20:09启动无GDB的单次续跑owner360200／SDK360245；01:34:43两阶段均实际退出0，完整预算通过。
54份原v2输出只读保持，显式组合凭据保留原post退出1／导航-11，不改写旧失败。
固定2848、全部11982验证控制步、冻结45D／90D参考、同一Diffusion40k与N4／200秒预算保持；
新增显式恢复来源准入及私有入口路由，原跟踪／物理／评分代码不变，14项CPU检查与Ruff通过。
首次导航冷预检的输出目录不符合原准入，失败保留；v2两项实际预检均退出0，21来源／5输入固定。
26项恢复评价／比较／续跑CPU检查、34个subtest及Ruff通过；默认权重尚未变更。

[S3-R优化器恢复训练](style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/README.md)已完成。
2026-10-07 23:46:12（Asia/Shanghai）终态：训练／CPU审计两个子进程均实际退出0，
恢复段500/500更新／4096000转换；保留原12000更新，有效总计12500／102400000转换。
最终权重`shared_social_budget12500_recovery_training_v2/step_012500.pt`及完整rollout已生成。
[外部冷核验](style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/owner_capacity_v3/external_verification.json)
实际退出0：16来源／4输入保持，28份训练输出只读，202张量审计通过，原权重／日志／AMASS资源哈希保持。
原S3[会话不可用核查](style_improvement_20261007/shared_social_S3_session_unavailable.json)仍成立：原日志12454，最后保存12000，退出码未知。
恢复模型／两Adam／四RMS／回放／训练RNG；物理World／课程私有RNG／历史重建。
未保存的454更新回退，重算12001–12500；不能宣称原S3连续执行或完整物理恢复。
真实256人单更新smoke实际退出0、CPU终态审计通过；8项检查及5个subtest、Ruff通过。
每25更新保存权重、125更新保存rollout；20权重／4完整rollout已保留，固定真实2848／seed7／256人保持。
[独立OS owner终态](style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/owner_capacity_v3/completion.json)
记录`Popen.wait()`实际子进程退出，训练耗时6925.39秒（约1小时55分25秒），终态清理成功。
外部IsaacSim进程保持、非GPU独占。恢复训练完成，跟踪／风格新结果与导航退出问题见顶部。
此前S2独立评价与新鲜S1 Diffusion/SFM200s配对已完成，模型质量均未通过；尚无达到论文效果的证据。

新增[episode候选终态来源准入／独立评价入口](style_improvement_20261006/shared_social_episode_evaluation_preparation_v1/README.md)：
新候选需匹配实际训练结果、注册来源、AMASS31配置、完整预算与保存配置，再调用原Pop-Art独立评价流程。
修正重复来源检查遗留私有helper别名的问题，原loader的严格拒绝条件保持；初始源码／登记及失败保留。
15项检查与11个subtest、Ruff及来源预检通过；真实S2在原退出／CPU审计准入后被候选终态选择正确拒绝，预期退出1。
未导入Torch或启动SDK；10份证据只读封存，[独立冷核验](style_improvement_20261006/shared_social_episode_evaluation_external_verification.json)
实际退出0，57项来源绑定／3项真实S2元数据输入／23项S3guard保持。新候选训练、终态接受及真实评价尚未执行。

新增[episode固定参考课程／World入口CPU准备](style_improvement_20261006/shared_social_episode_reference_runtime_preparation_v1/README.md)：
程序路径只在episode重置生成，后续重规划按episode年龄查询同一参考；AMASS仍future31，与future81候选分开。
真实2848的8人／256人私有入口实际CPU退出0，原AMASS轨迹／reset来源与每种规模4096专家来源窗口逐值保持。
66项检查（本候选27／参考9／原混合30）及Ruff通过，World四处准入替换可逆恢复原实现字节。
参考覆盖物理上限加5秒未来，同时改变末端保持和程序私有RNG曝光；不能称单一标量或作者配方精确复制。
31份证据只读封存，[独立冷核验](style_improvement_20261006/shared_social_episode_reference_external_verification.json)
实际退出0，41项实现来源／4项输入／23项S3guard保持。三份配置预检退出0，启动准入因缺少完整S3证书退出1。
World类型／来源／配置接线已完成CPU验证；物理World、SDK检查、2000更新独立训练及质量评价均未执行。

新增[S1/S2完整根部／身体误差诊断](style_improvement_20261006/shared_social_tracking_root_body_CPU_v1/README.md)：
两模型各22条／11982评价帧，实际CPU退出0，原始世界坐标评分逐值复现，跌倒和失败后帧全部保留。
S2根部XY误差0.853973m、根部旋转1.252676rad；移除各自根部平移后0.218184m，
再转入各自完整根部坐标系后0.132644m。这两项是诊断值，不能等同论文0.218m的原始跟踪分数；
原始全身误差仍0.918030m、成功0/22。后续需同时审查轨迹与朝向跟随，尚无训练失败因果结论。
22项CPU检查及Ruff通过，闭合条件反转触发6项失败并精确恢复；18份文件只读封存，74项原始输入复核。
没有新增SDK模拟或改变S3配方、固定2848清单及默认权重。

新增[完整跟踪命令／相位审计](style_improvement_20261006/shared_social_tracking_commands_CPU_v1/README.md)：
S1/S2各119820目标点，PRE相位／物理时钟／根部／episode／20维前缀一致，实际CPU退出0；
float64重算差约2.37e-7m，符合逐元素float32舍入界限。两者末尾保持目标均17858点（14.9040%）。
同相位参考根部公式离线均值0.296733／0.343848，不是训练奖励trace，也不定位因果。
38项CPU检查（新16／原22）及Ruff通过，相位条件反转触发3项失败并精确恢复；17份证据只读核验，83项输入保持。
本批未发现孤立捕获错帧；下一步审查训练目标剩余时长、真实episode长度与共享组重置造成的任务分布差异。

新增[实际训练episode时长与AMASS重置先验](style_improvement_20261006/shared_social_episode_horizon_CPU_v1/README.md)：
真实S1/S2终态与全部2848 bank的来源/帧数/30Hz验证通过，CPU会话14028实际退出0。
扣除末尾未完成组年龄后，完整整组episode平均1.699144／1.933537秒；独立validation22平均18.154545秒。
50个canonical未来规划节点的累计来源外比例48.6179%／49.6418%，不能混用10个目标点或评价末尾保持的分母。
全部2848保留时最多可要求81个真实未来控制步（2.7秒）；候选先验计算完成，现有训练配方保持。
32项检查及Ruff通过，条件反转触发3项失败并精确恢复；没有完整重置ID/帧或时长直方图，来源与物理终止可能重叠。
该任务分布差异尚未证明失败因果、未来81训练曝光或论文效果。

新增[future81独立课程CPU准备](style_improvement_20261006/shared_social_minimum_future81_CPU_v1/README.md)：
独立类型/状态schema保留原课程；37项检查（新7／原30）及真实2848 bank接口验证通过。
实际CPU会话78833退出0，来源抽取ID/procedural帧/每次RNG消耗保持；同合法绑定的原生轨迹和10目标逐值相同，
4096个专家来源窗口逐值相同。边界条件反转触发3项失败并精确恢复；初版测试ABI失败保留。
真实训练曝光会随物理episode/重置频率变化，不能由单次CPU抽样推定。
该历史CPU课程批次尚未迁移World；下方独立入口批次已完成CPU接线，SDK smoke和训练仍待执行。
两批各10份文件现已只读封存，独立冷核验通过；时长报告83项实际输入/16项实现来源和候选接口3项输入/6项来源保持，
S3的23项来源guard保持。原cell581会话丢失，当前优化器恢复任务见顶部；原完整完成证书未补造。

新增[future81独立World／训练入口CPU接线](style_improvement_20261006/shared_social_minimum_future81_runtime_preparation_v1/README.md)：
四处来源／类型／future81准入替换可逆恢复原World字节，物理／PD／控制实现保持。
真实2848／256人私有包入口检查实际退出0，40项实现来源／3项输入保持，最小剩余81帧；未启动SDK或训练。
23项检查（新入口16／课程7）及Ruff通过；反转条件触发5失败，恢复后全通过。
三份原配置逐字复制并CPU预检退出0，固定8人／256人短检查及2000更新候选顺序；
串行启动准入因尚无S3完整完成证书而实际退出1。World与entry源码接线已完成CPU验证，
实际SDK、完整训练和风格／路线／物理改善仍待实测，默认权重保持。

新增[全量人体前向轴／重置姿态CPU核对](style_improvement_20261006/shared_social_heading_geometry_CPU_v1/README.md)：
实际会话64029退出0，完整train2848／523272帧及S1/S2各22条／11982评价帧保留。
髋横向在自身根部坐标系中与+Y平均相差1.109473°，本批未观察到固定90°前向轴偏移或左右轴反转；
22个重置帧最大全身位置／旋转平均误差1.446798e-6m／1.120105e-6rad，原世界坐标指标保持。
34项检查（新12／原22）及Ruff通过；叉乘顺序反转触发5失败，精确恢复后全通过。
75项实际输入／8项来源保持；投影诊断、微小reset误差和通过测试均不能证明动作控制或模型效果合格。
本批没有启动SDK或新训练；继续针对任务分布与控制学习排查。S3当前会话状态见顶部核查。
15份证据现已只读封存，[独立冷核验](style_improvement_20261006/shared_social_heading_geometry_external_verification.json)
实际退出0，75项原始输入／8项实现来源及23项S3guard再次核对保持。

新增[实际终态路线条件响应](style_improvement_20261006/shared_social_trajectory_response_CPU_v1/README.md)：
CPU会话45955实际退出0，S1/S2终态使用相同的S2完整11982评价／8192训练观测；六条件共242088策略输入行实际前向。
仅更换20个局部路线值，身体／地图／RMS保持；前进条件有效动作RMS变化，validation为.019135／.020091，training为.019459／.017160。
所有六条件的57有效动作均无严格限幅，五个替换路线均未触及RMS clip5；本批不支持路线输入完全断开或响应均被限幅消除。
S2原条件CPU对GPU捕获的最大归一化动作差5.429983e-5如实保留，未放宽准入或声称逐位复现。
27项检查（新11／原16）及Ruff通过，KL求和错误突变被解析检查发现，恢复后全通过。
路线响应仍未证明动作方向／步态／闭环成功；原模型0/22与默认部署保持。
后续程序轨迹重锚定CPU对照见下方；公共PACER的episode固定参考机制与作者未公开训练分布分开说明。
15份证据已只读封存，[独立冷核验](style_improvement_20261006/shared_social_trajectory_response_external_verification.json)
实际退出0；81项实际输入／10项直接实现来源、逐行人口与23项S3guard保持。

新增[程序轨迹重锚定全量CPU对照](style_improvement_20261006/shared_social_procedural_anchoring_CPU_v1/README.md)：
绑定实际S2配置／退出，真实2848各取最早合法帧9的位置／姿态，声明静止及初始XY速度匀速运动，
三种原生成器／同首路径重启／episode时钟共2563200个奖励行，CPU会话15431实际退出0。
静止平均轨迹项为.964443／.964326／.234614；该机制保留高静止奖励，尚未证明实际策略失败因果。
原奖励／51节点插值保持，没有物理积分或动作质量结论；7项输入／10项直接实现来源与23项S3guard保持。
新增独立EpisodeProceduralReference模块只在重置生成路径，查询不依赖当前根部；10秒episode加5秒未来的路径设计明确披露。
真实2848候选接口实际退出0，首51节点和前150个POST目标逐值保持，455680个目标查询及参考状态恢复通过。
49项检查（本批20／原29）与Ruff通过；该历史模块批次未接World，上方新入口批次已完成CPU接线。
SDK模拟和独立训练尚未执行，future81与此候选分开登记。
21份证据只读封存，[独立冷核验](style_improvement_20261006/shared_social_procedural_anchoring_external_verification.json)
实际退出0；原环境下所有已存均值／逐行人口和候选目标逐值保持，23项S3来源guard再次核对。

新增[原生共享AMP接入](SHARED_SOCIAL_NATIVE_EVALUATION_ZH.md)：四个真实地图各N3–15共52个固定案例，
48个资源准入、2058段A*通过；Hospital N12–15仍因间距不足拒绝。48个合成快照SFM预测实际CPU退出0。
原生DP/SFM入口已接入Pop-Art准入与N+1接触视图释放，30项CPU检查及两臂资源预检通过；
真实原生SDK smoke、准确初态配对及完整物理/风格审计仍待完成，不能计作论文效果。
原生初态版本现已接入同AMP的DP/SFM共用采集点，新增四个原地图通道，显式扩展N3–15并默认拒绝CPU夹具。
[最终初态CPU证据](style_improvement_20261005/shared_social_native_initial_preparation_v3/README.md)：
79项相关检查通过，两臂严格资源预检实际退出0；准入条件突变触发3项失败，恢复源码后全通过。
正常/异常路径恢复临时回调，避免保留World别名；尚无真实原生初态档案或新增物理/风格效果。
新增[原生raw审计CPU准备](style_improvement_20261005/shared_social_native_raw_audit_preparation_v2/README.md)：
显式N3–15、30/6000控制步；最终62项检查及四项来源预检通过，实际外部退出0。
原有DP/SFM真实N4捕获的全部指标字段保持；退出条件突变触发21项失败并精确恢复。
已修正driver没有`native_dynbench`字段的ABI假设，改用实际native scene/provider/live stage准入；
v1失败、旧源码和88份文件只读封存。尚无新原生SDK结果；新seed版本CPU接口见下方新增记录。
新增[原生独立风格CPU准备](style_improvement_20261005/shared_social_native_style_preparation_v2/README.md)：
N3–15完整6000步、两臂实际raw-CPU退出、原生初态/资源和只读数组准入；53项相关检查通过。
旧DP/SFM真实N4的全部样本、45D/90D字段及共同桶数值保持，固定validation参考与K50/K100中心不变。
v1的tuple/list JSON表示失败保留；v2只修正runner的表示比较，没有指标改动或数值容差。
准入突变实际21项失败并恢复；92份文件只读封存。原生风格reader已完成CPU准备，
真实原生SDK/初态/200秒风格和多seed仍未验证，不能计作论文效果。
新增[显式seed入口及本地矩阵](style_improvement_20261006/shared_social_native_seeded_preparation_v1/README.md)：
48项相关CPU检查、12项冷源码检查及两项S1兼容资源预检通过，17个子进程均实际退出0。
driver逐字保持；相对原seed7版本，实际转发参数仅seed和output不同。类型校验突变触发26项失败并恢复。
四地图N3–15按本地seed7/19/42登记156组合，其中144资源准入、12保留拒绝，计划288次两臂200秒运行；
这些运行均未执行。88份证据只读核验，S3的23项来源guard保持。只选S3预登记终态，不选中途模型。
该阶段的接口缺口现已由下方独立seed版本补齐；旧seed7入口和reader保持原来源约束。
本地三seed矩阵未匹配作者八场景协议，不能计作论文效果。
新增[显式seed原始数据／风格CPU准入](style_improvement_20261006/shared_social_native_seeded_admission_preparation_v1/README.md)：
195项相关检查通过，33个子进程实际退出0；seed相等条件突变触发13项失败，精确恢复后全通过。
六个测量内核及原风格抽取函数保持；两份历史真实N4的完整原始指标、45D/90D与共同桶数值保持。
退出记录／driver／初态／入口注册的seed必须相同，风格两臂还须匹配raw报告／内核／底层准入seed。
119份文件只读核验；S3的23项guard保持，没有新增SDK捕获或论文效果。
[实际分项计数修正](style_improvement_20261006/shared_social_native_seeded_admission_preparation_v1/test_inventory_correction.json)：
新接口95项、原回归100项，总195项不变；封存v1 README的97/98误写保留并显式纠正。
真实World/RNG配对、SDK smoke和每cohort的完整200秒raw/风格准入仍待实际执行。
新增[原生工具退出桥接CPU记录](style_improvement_20261006/shared_social_native_execution_bridge_preparation_v1/CPU_preparation_execution.json)：
30项契约检查、四地图冷预检及Ruff通过，6个子进程实际退出0；条件突变触发5项失败并精确恢复。
桥接要求实际工具终态chunk/整数退出码，并拒绝用S1权重填入S3登记位置；23项S3来源guard保持。
56份文件现已只读封存并独立核验；真实SDK退出凭据发布仍未验证。
新增[串行工具owner修正后CPU准备](style_improvement_20261006/shared_social_native_tool_owner_preparation_v2/README.md)：
42项pytest（新12项/桥接30项）、四地图登记/冷读取及Ruff通过，10个子进程退出0；另13项V8契约通过。
运行时命令/路径必须与只读登记逐字段匹配，条件反转被检查捕获并精确恢复；失败退出先发布，再停止后续步骤。
71份文件只读核验，S3的23项guard保持；实际工具链因尚无S3完整证书而退出1，没有启动SDK。
[冷加载启动模板修订](style_improvement_20261006/shared_social_native_tool_owner_preparation_v2/launch_wrapper_revision_v2.json)
及其来源独立绑定在原索引之外；原初版/失败和74份只读证据保留。冷加载与SDK观察均不自动重启会话。
完整原生21步串行流程及成功凭据仍需S3结束后实测，本批没有新动作或物理效果。

## 已实际通过

| 检查 | 实际结果 |
| --- | --- |
| 原生 SDK 接触检查 | 8 人／2 组、64 个物理子步，真实进程退出 0 |
| 无过滤接触视图结构 | SDK 返回 `filter_count=0` 和 24 个空过滤列表；映射与路径顺序通过 |
| 共享 World | 8 人、9 个控制步／36 个物理子步，整组重置和跌倒探针通过，真实退出 0 |
| 真实 AMASS 短训练 | 8 人、32 条物理转换、1 次 PPO／AMP 更新，真实退出 0 |
| 短训练权重审计 | 模型、两个 Adam、replay、四组 RMS 及最终 rollout 核查通过；202 个张量有限，CPU 未初始化 CUDA |
| 256 人规模检查 | 64 组、2 次更新／16,384 条物理转换，replay 实际参与判别器更新；真实退出 0，CPU 审计通过 |

当前短训练不是具备成熟步态的性能模型。256 人规模检查已通过，保留了
随机初始策略的 126 次真实跌倒标志和 40 次非脚部净法向力终止；这些数量
不是同行人 pair 接触指标。两个更新都触发目标 KL 提前停止，各实际施加
1 个 PPO minibatch；这一机制和配方保持，后续检查实际学习效果。

2,000 次更新的共享 AMP pilot 已从全新初始化完成，实际 16,384,000 条
物理转换和 8,192,000 个专家窗口；不续训短检查权重。2026-10-05 主机进程
实际退出 0，256,000 个物理子步，原生接触视图和 backend view 均释放。
终态 CPU 审计通过：模型、两个 Adam、replay、四组 RMS 及最终 rollout 一致，
202 个张量有限；终态权重 SHA256 为 `8dcedc67e8b0cafb8fb9d53ca4ca34a09a712ff90e88b8572996892e85d516f8`。
完整 validation22 和固定参考风格评价已实际退出 0，但模型质量失败：
0/22 跟踪成功、22/22 摔倒，平均位置误差 1.865725 m、旋转误差 2.138779 rad。
45D MMD² 为 0.863276；90D 同速度桶 MMD² 为 0.499120。
四人 200 s 独立闭环已实际退出 0；独立 CPU 数组审计确认 956 次摔倒重置、
四人路线点进度全为 0，24,000 actor-controls 的静态及同行人接触均为 0。
大量摔倒和零路线进度说明模型不合格，原部署默认权重保持。
Pop-Art 的真实 8 人/256 人训练、终态审计、CPU 权重加载及四人导航 30 步均退出 0。
S1 同数据/seed/预算的全新 2,000 更新配对已训练并完成所有独立评价，实际退出0。
跟踪仍为0/22；跌倒动作降为7/22，位置/旋转误差1.008892m/1.332191rad。
固定45D MMD²改善到.035621，但K50/K100覆盖率降为.64/.49；
共同4个速度区间的90D MMD²为S0 .509862→S1 .068384。
四人200s为5次跌倒、37个同行人接触pair-controls、3个路线点，仍不合格。
新增[原始运动与支撑诊断](style_improvement_20261005/shared_social_locomotion_CPU_v1/README.md)
确认S1有效位置间隔80.46%低速、90.44%双支撑，四人最长连续双支撑149–176s。
执行参考本身也以低速为主；条件停滞统计单独保留各自命令分母，不能据此定位因果。
[已保存计划的CPU对照](style_improvement_20261005/shared_social_planner_response_CPU_v1/README.md)
进一步确认S1 raw前缀均速.328m/s、执行参考.253m/s；后100秒raw也以低速为主。
原始计划／适配／反馈的关联需通过后续配对消融定位，不能仅凭此诊断调整默认模型。
新增[严格补充准入](style_improvement_20261005/shared_social_admission_v2/S0_S1_supplemental_admission_v2.json)
绑定实际child、完整跟踪协议、底层风格实现及各方captures；只有样本数／参考索引相同的90D桶可比较方向。
当前S0/S1四个共同桶均通过，原结果保持。
详见[S1 完整独立对照与下一轮](SHARED_SOCIAL_POPART_RESULTS_ZH.md)及
[S0 原始结果](SHARED_SOCIAL_PILOT2000_RESULTS_ZH.md)。

## 入口与配置

- 规范训练入口：[train_shared_social.py](../scripts/train_shared_social.py)，显式 legacy/popart
- 8 人短检查：[smoke.json](style_improvement_20261004/shared_social_training_entry_staging_v1/smoke.json)
- 256 人规模检查：[smoke256.json](style_improvement_20261004/shared_social_scaled_training_smoke_v1/smoke256.json)
- 256 人／2,000 更新本地 pilot：[training.json](style_improvement_20261004/shared_social_training_entry_staging_v1/training.json)
- 完成后的权重与 rollout 审计：[audit_shared_social_training.py](../scripts/audit_shared_social_training.py)
- 只读训练进度：[shared_social_training_status.py](../scripts/shared_social_training_status.py)
- 独立评价协议：[SHARED_SOCIAL_EVALUATION_PROTOCOL_ZH.md](SHARED_SOCIAL_EVALUATION_PROTOCOL_ZH.md)
- 新评价入口：[evaluate_shared_social_tracking.py](../scripts/evaluate_shared_social_tracking.py)
- 固定参考风格评价：[evaluate_shared_social_style.py](../scripts/evaluate_shared_social_style.py)
- 串行评价执行器：[run_shared_social_evaluation.py](../scripts/run_shared_social_evaluation.py)
- 四人闭环入口：[evaluate_shared_social_navigation.py](../scripts/evaluate_shared_social_navigation.py)
- 终态 CPU 稳定性诊断：[diagnose_shared_social_training.py](../scripts/diagnose_shared_social_training.py)
- 规范训练入口，显式 legacy/popart：[train_shared_social.py](../scripts/train_shared_social.py)
- 新权重的实际原始观测 CPU 加载：[audit_shared_social_inference.py](../scripts/audit_shared_social_inference.py)
- 完整导航原始数组审计：[audit_shared_social_navigation.py](../scripts/audit_shared_social_navigation.py)
- 训练结束后的串行评价：[run_shared_social_post_training.py](../scripts/run_shared_social_post_training.py)
- 独立退出记录与终态封存：[seal_shared_social_training.py](../scripts/seal_shared_social_training.py)
- 独立效果对照：[compare_shared_social_evidence.py](../scripts/compare_shared_social_evidence.py)
- 独立补充准入：[verify_shared_social_evidence_admission.py](../scripts/verify_shared_social_evidence_admission.py)
- 完成记录的运动／支撑诊断：[diagnose_shared_social_locomotion.py](../scripts/diagnose_shared_social_locomotion.py)
- 终态D离线敏感性：[diagnose_shared_social_discriminator.py](../scripts/diagnose_shared_social_discriminator.py)
- 原训练窗口先验／实际专家曝光：[diagnose_shared_social_expert_coverage.py](../scripts/diagnose_shared_social_expert_coverage.py)
- 实际动作前任务／身体／专家速度：[diagnose_shared_social_task_context.py](../scripts/diagnose_shared_social_task_context.py)
- 完整跟踪根部／身体误差诊断：[diagnose_shared_social_tracking_errors.py](../scripts/diagnose_shared_social_tracking_errors.py)
- 完整跟踪PRE命令／相位审计：[audit_shared_social_tracking_commands.py](../scripts/audit_shared_social_tracking_commands.py)
- 实际训练episode与AMASS重置时长：[diagnose_shared_social_episode_horizons.py](../scripts/diagnose_shared_social_episode_horizons.py)
- 独立future81课程CPU候选：[minimum_horizon_curriculum.py](../CrowdSim/humanoid/training/minimum_horizon_curriculum.py)
- 独立BABEL dense序列跨度约定：[extend_babel_dense_sequence_semantics.py](../scripts/extend_babel_dense_sequence_semantics.py)
- 同一AMP的SFM高层CPU准备：[social_force.py](../CrowdSim/humanoid/trajectory/social_force.py)
- 同一AMP的SFM fixture接入：[evaluate_shared_social_sfm.py](../scripts/evaluate_shared_social_sfm.py)
- 新鲜高层配对与初态捕获：[evaluate_shared_social_high_level.py](../scripts/evaluate_shared_social_high_level.py)
- 初态精确接纳：[compare_shared_social_initial_conditions.py](../scripts/compare_shared_social_initial_conditions.py)
- smoke／完整原生控制与全身来源审计：[audit_shared_social_high_level.py](../scripts/audit_shared_social_high_level.py)
- S2 完整 PPO 轮次候选：[training.json](style_improvement_20261005/shared_social_full_epochs_v1/training.json)
- 原生N3–15 DP/SFM准备入口：[evaluate_shared_social_native.py](../scripts/evaluate_shared_social_native.py)
- 原生N3–15初态配对入口：[evaluate_shared_social_native_paired.py](../scripts/evaluate_shared_social_native_paired.py)
- 原生N3–15独立raw审计：[audit_shared_social_native.py](../scripts/audit_shared_social_native.py)
- 原生N3–15独立风格评价：[evaluate_shared_social_native_style.py](../scripts/evaluate_shared_social_native_style.py)
- 原生N3–15显式seed入口：[evaluate_shared_social_native_seeded.py](../scripts/evaluate_shared_social_native_seeded.py)
- 本地三seed矩阵登记：[register_shared_social_native_seeds.py](../scripts/register_shared_social_native_seeds.py)
- 显式seed原生raw审计：[audit_shared_social_native_seeded.py](../scripts/audit_shared_social_native_seeded.py)
- 同seed原生两臂独立风格：[evaluate_shared_social_native_seeded_style.py](../scripts/evaluate_shared_social_native_seeded_style.py)
- 原生工具退出桥接与案例命令准备：[prepare_shared_social_native_execution.py](../scripts/prepare_shared_social_native_execution.py)
- 串行案例登记/冷准入/捕获封存：[prepare_shared_social_native_owner.py](../scripts/prepare_shared_social_native_owner.py)
- 实际工具API串行owner：[shared_social_native_tool_owner.js](../scripts/shared_social_native_tool_owner.js)

## S1／S2 已实际完成；两者质量仍未通过

S1 工具会话16374已实际退出0；终态CPU审计与封存通过。
完整后续串行流程也实际退出0（`c57118`），五个子进程全部正常退出。
上述性能指标来自独立物理测量，未使用训练奖励或测试fixture替代。

实际训练启动与配方见[登记](style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_registration.json)，
已验证的评价源码与测试见[准备记录](style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_preparation.json)。
外部工具owner已完成会话16374的终态接管（exec cell223已完成）：
[实时阶段](style_improvement_20261005/shared_social_evaluation_v1/popart_terminal_owner_state_v2.json)、
[封存owner代码](style_improvement_20261005/shared_social_evaluation_v1/popart_terminal_tool_owner_v2.js)。
最终阶段为`completed_measurements_quality_unclassified`；
[实际退出及测量哈希](style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_execution.json)
已独立保留。不从PID或结果文件推测退出。
初版owner因元数据输出截断而停止，记录独立保留；v2增大元数据读取额度，训练没有重新启动。
S2 根据PACER公开循环，仅将`training.target_kl`从.02改为null，保持Pop-Art和其余配方。
CPU完整6轮/12minibatch fixture通过；真实8人1更新/32转换和256人2更新/16,384转换、
CPU审计及冻结推理均退出0；四人Diffusion30控制步也实际退出0。
S2全新256人/2000更新训练及全部串行独立评价已完成，外部owner cell278已结束。
真实训练、CPU终态审计、封存、五个评估子流程和CPU对照均实际退出0。
实际2000更新／16,384,000物理转换／8,192,000专家窗口，全部24,000 PPO minibatch施加。
严格补充准入及全量原生数组复算通过，但质量未通过：跟踪仍0/22，跌倒动作7→5，
位置1.008892→.918030m、旋转1.332191→1.412125rad；K50/K100覆盖.64/.49→.72/.63。
共同四桶90D MMD² .06838440→.06843448，导航跌倒5→9、路线点3→1、peer pair-controls37→32。
完整结果与独立证据见[S2结果](SHARED_SOCIAL_FULL_EPOCH_RESULTS_ZH.md)。
见[S2实际入口与预登记](style_improvement_20261005/shared_social_full_epochs_v1/README.md)、
[S2实时阶段](style_improvement_20261005/shared_social_full_epochs_v1/terminal_owner_state.json)。
当前保持所有部署默认权重。

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/train_shared_social.py --value-normalization popart \
  --config docs/style_improvement_20261004/shared_social_training_entry_staging_v1/training.json \
  --device cuda:0 --output output/diffusion_amp/shared_social_popart_pilot2000_v1
```

输出目录必须为空或不存在。每次失败及其部分数据保留。
只启动一个 GPU 训练／模拟进程，避免与正在运行的检查或 pilot 重叠。

读取历史 S1 训练指标；指标不证明模型效果达标：

```bash
python scripts/shared_social_training_status.py \
  output/diffusion_amp/shared_social_popart_pilot2000_v1
```

本次 PID 58934 已实际退出；未来训练的 `--pid` 必须使用对应的新 PID，并在
该进程所在的命名空间中读取。指标本身不证明进程存活或正常退出。

该入口验证并加载仓库内独立 simulation-manager 扩展副本，安装 SDK 文件
未修改。修复包含 AMASS 启动前内容校验、接触视图空过滤结构兼容及已验证的
原生终态释放。构造中途发生其他异常时的完整清理仍有未验证路径。

## 与 Diffusion 闭环的剩余工作

1. S2已完成所有独立轴及严格准入，质量失败；下一候选需检验实际学习效果，不能据训练退出或覆盖率单项晋升。
2. [新鲜完整两臂结果](style_improvement_20261005/shared_social_S1_paired_actual_v1/README.md)：固定同S1／seed7／formalDP40k，两臂smoke和6000步、native CPU审计、完整32通道初态匹配、闭环风格均实际退出0。DP/SFM跌倒5/20、静态0/21、peer37/31、路线点3/6；各有取舍，两臂均不合格。
3. [闭环风格入口](../scripts/evaluate_shared_social_navigation_style.py)直接读取完成后的native全身姿态，8项CPU回归检查及实际两臂测量通过。固定validation／K50／K100、不过滤跌倒；共同五桶固定平衡样本数与参考索引，90D MMD²为DP .112125／SFM .131224，但DP覆盖和路线进度更低。继续训练可用共享控制器后再验证native场景。
4. 原生N3–15 DP/SFM、显式seed入口及raw/风格CPU接口、48个资源配置和156组合登记已通过检查；退出桥接及串行工具owner完成CPU准备/封存，实际S3门控拒绝通过。待S3及其后续评价实际结束，再验证SDK smoke、准确初态、多seed和200s配对及原始指标。Hospital N12–15间距检查未通过。作者八场景名单及计数分母仍缺少，不补造场景。
5. 获得可用共享 critic 后，接入和消融候选轨迹的 value guidance。
6. 在相同评估协议下分别验证交互课程、AMASS 窗口分层和真实脚部支撑测量。

2,000 次更新是本地 pilot，未证明等于作者完整训练预算。
当前没有达到／超越论文完整效果的证据，也没有替换部署默认权重。

S0/S1完整200s测量及生命周期释放均已正常退出，完整跟踪／风格也已执行；
当前模型未通过质量要求。Pop-Art已完成真实AMASS/SDK训练及独立评价。
[终态D输入检查](style_improvement_20261005/shared_social_D_input_diagnostic_v1/README.md)
仅加载D/RMS，完整8192 policy与4096 expert行的离线动作归零／交换几乎不改变评分；
该负向诊断降低“动作代理主导失败”假设优先级，没有作训练消融或质量因果结论。

[原窗口先验与实际专家末批](style_improvement_20261005/shared_social_expert_coverage_CPU_v1/README.md)
已实际CPU退出0：完整2848／494792合法中心，原clip-uniform的fast短窗概率1.5801%，
与window-uniform的1.2254%分母不同；末批4096行中fast46，覆盖2149个clip。
约72%的fast先验来自非running_like整段，不能按17个run整段推断高速支持。
保留1713行没有BABEL train exact match，1218行显式时间并集覆盖；后者不等于单动作真值。
全程8,192,000专家draw计数已证明，所有ID的全程ledger尚缺；未改采样或证明失败原因。

[真实动作前任务上下文](style_improvement_20261005/shared_social_task_context_CPU_v1/README.md)
新增8项CPU契约与完整8192行／4096专家实测，实际退出0；PRE／POST历史、episode、
时钟及native速度来源核对通过。程序／AMASS分支未来9段中位位移率≥1.8m/s分别为
1628／20行（各4096），当前身体均速1.148517／.542247m/s；位移率不是即时速度命令。
专家history10高速46行，与原分类一致；任务时间尺度与专家短窗没有逐行配对。
所有跌倒／reset行保留，未改先验或据此证明导航低速原因。

[BABEL dense跨度约定补充](style_improvement_20261005/babel_dense_sequence_convention_CPU_v1/README.md)
6项CPU测试、实际2848全量构建退出0：304个明确单动作dense父动作对应399个clip，
与原844个显式timed union全clip覆盖分开记录；不从extra回退或拉伸BABEL时长。
原标签／未知／动作ID保留，未改训练或恢复作者私有清单。

训练日志出现后期稳定性变化：更新 1,401–1,500 有 523 个跌倒标志，而
1,771–1,870 有 9,976 个；两个窗口各 819,200 个 actor-controls。后者
平均训练奖励更高，说明不能仅用奖励判断稳定性。这些是训练 reason flags，
不是独立导航跌倒率或同行人 pair 接触指标；先以完整独立评价核查效果。

方法继续以[NavIsaacLab IV-B／IV-C](https://arxiv.org/html/2606.26265v1#S4.SS2)
及[TRACE／PACER 作者资料](https://research.nvidia.com/labs/toronto-ai/trace-pace/)
为依据。原文的 2,848 条导航动作数量不提供完整作者动作清单；现有共享 CNN
和方向惩罚含明确的本地实现选择，训练完成后仍需同协议比较。

## 实际记录

- [共享 World 成功记录](style_improvement_20261004/social_actual_runtime_v1/shared_social_world_probe_contact_v6_actual_root_owner.json)
- [真实短训练成功记录](style_improvement_20261004/social_actual_runtime_v1/shared_social_training_smoke_contact_v6_actual_root_owner.json)
- [256 人规模检查成功记录](style_improvement_20261004/social_actual_runtime_v1/shared_social_training_scaled256_smoke_v1_actual_root_owner.json)
- [2,000 更新 pilot 登记](style_improvement_20261004/shared_social_pilot2000_actual_launch_v1/registration.json)
- [2,000 更新实际退出与终态接纳](style_improvement_20261005/shared_social_evaluation_v1/training_execution.json)
- [终态 CPU 审计](style_improvement_20261005/shared_social_evaluation_v1/training_CPU_audit.json)
- [独立评估准备检查](style_improvement_20261005/shared_social_evaluation_v1/source_preparation.json)
- [TorchScript 来源修正与保留的失败](style_improvement_20261005/shared_social_evaluation_v1/source_preparation_v2.json)
- [实际跟踪、风格与 N4 短检查退出](style_improvement_20261005/shared_social_evaluation_v1/tracking_style_execution.json)
- [真实 GAE 与 critic 稳定性诊断](style_improvement_20261005/shared_social_evaluation_v1/training_stability_CPU.json)
- [短训练 CPU 审计](style_improvement_20261004/social_actual_runtime_v1/shared_social_training_smoke_contact_v6_CPU_audit.json)
- [实际接触视图结构](style_improvement_20261004/social_actual_runtime_v1/social_net_normal_layout_actual_root_owner.json)
- [v4 数据清单错误](style_improvement_20261004/social_actual_runtime_v1/shared_social_world_probe_overlay_v4_failure_owner.json)
- [v5 接触视图构造错误](style_improvement_20261004/social_actual_runtime_v1/shared_social_world_probe_preflight_v5_failure_owner.json)
