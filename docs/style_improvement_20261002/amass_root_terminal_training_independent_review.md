# AMASS根轨迹1000更新：独立终点审计

## 结论与工件

真实训练已完成1000更新、8,192,000 transitions、32,000控制步；CPU审计通过完整checkpoint/source145项门禁及额外normalizer/Adam/replay/RNG/日志计数检查。

- [完整机器报告](amass_root_terminal_training_independent_audit.json)及[完整payload核心审计](amass_root_terminal_training_independent_audit_checkpoint_core.json)。
- [终点训练结果](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/amp_style_amass_root_curriculum_v1/result.json)。
- 不可变snapshot `step_1000.pt`：652,092,533B，SHA `dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e`。

训练elapsed为1,688.677秒。checkpoint使用`weights_only=True`及当前严格CPU trainer loader接受；整bundle含模型、两个optimizer、四个normalizer、replay、RNG及runner物理/逻辑状态，共133份tensor均finite。所有输入文件SHA在审计前后不变，124份预启动archive逐文件校验。

审计只读CPU，没有启动Kit/GPU或修改业务/数据/权重，没有打开封存34条test输出。通过表示执行、工件完整与恢复合同有效；风格和tracking效果需独立实际验证。

## 物理与课程计数

256个actor，120Hz物理/30Hz控制，地面192m、出生offset48m。训练使用真实AMASS根XY、固定reset平移与明确线性尾部；Diffusion没有进入训练rollout。训练数据保持v7 child_joint，2,848 segments，公开PACER actor/critic初始化，state-action discriminator。

| 累计项 | 实测 |
|---|---:|
| 物理fall reset | 157 |
| 非fall截断 | 79,923 |
| clip真实末帧截断 | 79,876 |
| 仅episode cap截断 | 47 |
| clip末帧与cap同时发生 | 2，已包含在79,876中 |
| 初始化+episode reset | 80,336 |
| 课程累计actor控制步 | 8,192,000 |

每100,000 transitions出现1.9165次fall reset；这个分母不同于episode失败比例。clip结束/cap不代表tracking成功，课程大量短源episode也不等于200秒连续导航稳定。

累计621,164份计划、31,058,200个未来10Hz knots，其中17,277,090个(55.6281%)超出真实clip范围，并且全部由声明的linear_velocity任务尾部产生，hold计数为0。没有在真实clip末帧之后继续物理episode；这部分不是观测到的AMASS未来。

## 更新、optimizer与normalizers

1000条metrics的update IDs及每轮8192新增transitions连续；最终fall/timeout与结果一致。每个Adam参数state的step与实际接受minibatches累计值精确一致：

| 项目 | 参数state数量 | Adam step累计 | learning rate |
|---|---:|---:|---:|
| actor/task encoder/critic optimizer | 16 | 9,864 | 2e−5 |
| discriminator optimizer | 6 | 2,000 | 1e−5 |

四份normalizer的count均从实际warm-start cap200,000和真实更新batch精确推导：

| Normalizer | dimensions | 最终count | std median |
|---|---:|---:|---:|
| self | 357 | 8,392,000 | 0.257922 |
| task | 1044 | 8,392,000 | 0.241913 |
| AMP | 1950 | 12,488,000 | 0.157058 |
| value | 1 | 8,392,000 | 11.741907 |

task最后1024个height channels的std median为0.239669，min0.0257265/max0.371195。这是平地训练后的分布记录，不能据此确定导航障碍反应原因。value mean为101.323331。所有moments variance非负、mean/variance/count finite；正常化计数符合每轮更新规则。

Replay容量/size均65,536、accepted146,978、cursor15,906，环形计数一致，history形状`[65536,10,195]`及action形状`[65536,69]`匹配；12个冻结applied action坐标严格为零。

Python/NumPy/CPU Torch RNG均由局部validator接受，共享corpus与课程RNG逐位相同。保存1份16-byte CUDA RNG状态；CPU审计没有调用CUDA恢复，因此不冒充GPU RNG功能实测。runner全部256个实际绑定phase/offset/age/episode/native151计划满足恢复门禁。PhysX contact/solver缓存没有序列化，不能宣称物理逐位恢复。

## 四段训练统计

下表每段均2,048,000 transitions。reward为update算术平均，用于检查日志趋势，不作为论文一致性或风格收敛阈值。

| 更新 | falls | clip/cap截断 | trajectory reward | AMP reward | backwards fraction | expert/live D概率 |
|---|---:|---:|---:|---:|---:|---:|
| 1–250 | 32 | 19,969 | 0.887786 | 0.349122 | 0.224607 | 0.834811 / 0.153271 |
| 251–500 | 52 | 19,934 | 0.957743 | 0.367485 | 0.134403 | 0.844667 / 0.160680 |
| 501–750 | 45 | 20,057 | 0.966978 | 0.362183 | 0.123870 | 0.845909 / 0.158214 |
| 751–1000 | 28 | 19,963 | 0.971308 | 0.358289 | 0.126954 | 0.845640 / 0.156548 |

任务跟踪reward增加，D仍明显区分expert/live。训练fall计数或reward趋势不能单独证明风格已经收敛。

## 源码与watcher身份

[独立watcher身份审计](amass_root_terminal_watcher_identity_independent_audit.json)通过：

- GPU预启动124-source archive SHA为`ed16c4cdbfd533a9fe2bfeb50c3f717cdbb0f033462dadd8b4acf62a0adaf56a`，其中watcher保存的是历史准备版本SHA `97182cfeb9ad2116cfe9b0ef41281fd7fea72e39753fd1157469bf86db058dae`。
- 实际CPU watcher为单独冻结v2，SHA `db8d243aedc5f43cd0f920b3888c8e35e207929826be9f26e850c8dad7713eb8`，不是与旧准备版本逐位相同的工具。GPU业务源码没有因此改变，训练不导入该watcher。
- 实际ledger终态passed、missing为空，200/500/1000三个不可变snapshot的bytes与SHA均重新核对。watcher只读训练目录、发布到独立snapshot目录；审计未启动第二实例。

新完整22条tracking的协议/参考输入验收见[独立tracking报告](amass_root_step1000_tracking_independent_review.md)。误差分解和冻结风格/MMD由各自负责代理报告，未在本审计重复计算。
