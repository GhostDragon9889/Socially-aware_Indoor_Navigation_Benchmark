# 地形统计保护：真实 fresh2000 训练终态独立核验

**2046/2046 项独立 CPU 门禁通过，实际终态准入成立。** 读取真正封存step2000、actual result与精确2000行metrics，未使用milestone替代。root报告三个会话exit0；CPU读者本身不验证SDK进程退出。

| 项目 | 实际值 |
| --- | --- |
| fresh更新 / transitions | 2000 / 16384000 |
| real-AMASS / procedural transitions | 8192000 / 8192000 |
| 256环境共享control steps | 64000 |
| policy / D Adam steps | 17510 / 4000 |
| self/task/value / AMP RMS count | 16584000 / 24576000 |
| protected hook calls | 2000 |
| replay size / accepted / cursor | 65536 / 228415 / 31807 |
| 累计训练跌倒 / 时间限制 | 510 / 106771 |
| 最后metrics用时 / 最终result用时 | 3813.4069095509767s / 3815.4821180829895s |

checkpoint 653044761B，SHA `e11a9a47d5e1147c4f894a519827d50539faf449a892c714d690957f1b267663`；result SHA `21130f3eb889a4ddcb793bb59702984528388d551a5377ba2b51eb18fdca9392`。watcher封存500/1000/2000且无missing；先前两份封存metrics前缀与真正终态日志原始字节完全一致。两个用时字段分开保留，约2.075秒差值来自记录时点，不能把最后训练日志时间当result总时间。

## 统计与完整恢复

- 公开参考count61931520001、暖启动count200000、实际增长globalcount分开。task[20:1044]的1024维mean/variance逐位等于固定公开fixture；500/1000/2000参考一致，数组SHA `5ebd61c10871dfad790d1dbbf6e9d1e05e6e6f0642522900dc0412601c176137`。前20维与globalcount继续更新。
- D2019未转移public state-only D/AMP统计。完整actor/critic/taskencoder/log_std、两Adam和实际优化步数、四normalizer、57有效动作replay、256个COMMON XYZW/69DOF physical state、same-episode history10、real151/procedural51计划相位、各课程/原corpus RNG均严格加载。
- 下一次subset reset/expert/plan的CPU逻辑抽样重复一致。CUDA RNG仅保存字节，没有CPU执行CUDA或物理重放，未序列化PhysX solver/contact cache，不声称物理轨迹逐位恢复。
- strict推理只读此checkpoint，不需外部统计NPZ/公开checkpoint；3个明确synthetic CPU输入raw/applied与trainer逐位一致、moments/caller RNG不变。旧RMS-only隔离epoch保留完整PaperAMPNetwork，未夹带现代actor-only裁剪。

## 来源与准入

931只读代码/资产与940-member归档逐项SHA互证；旧926中921文件保持原样，仅5旧路径RMSdelta+5新增。source manifest4227b3d8…；实际GPU ledger前后90events/87owned全来自隔离code。SDK快速退出缺少final事件，finalization=false按实际保留；实际结束确认来自result及root退出通知，不造ledger事件。

冻结f58评估executor在纯stdlib CPU中实际接收本终态admission，source931/2000updates/16.384M/D2019/保护reference均匹配。本读者未执行GPU评估。

510训练跌倒/106771时间限制保留完整。此结果是技术来源与恢复验收，尚无导航、步态自然性或论文一致性效果结论；下一步才是预注册完整22与同协议导航/风格评估。

[完整2046项核验](audit.json) · [真正终态闭合](terminal_closure_actual_proof.json) · [实际consumer准入](consumer_admission_actual_proof.json)
