# 地形 RMS 保护：真实 4 + 恢复 4 的独立核验

**2094 项独立 CPU 门禁通过。** 读取的是 root 实际 GPU 作业关闭后封存的两个 checkpoint、result、metrics 与同步 import ledger；本审计没有启动GPU/SDK，没有改训练源码。该结果仅证明来源/完整保存与恢复合同，不是导航、步态或论文质量验收。

| 累计更新 | 总 transitions / 各来源 | task 共享计数 | AMP计数 | hook calls | policy Adam / D Adam step | replay size/cursor |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 32768 / 16384 | 232768 | 49152 | 4 | 12 / 8 | 32768 / 32768 |
| 8 | 65536 / 32768 | 265536 | 98304 | 8 | 33 / 16 | 65536 / 0 |

公开参考计数始终为 **61931520001**；独立暖启动 task count 为200000，与上述实际增长共享count分开。suffix mean/variance与公开权重导出的固定17KB fixture逐位相等，4→8亦逐位不变，固定数组SHA `5ebd61c10871dfad790d1dbbf6e9d1e05e6e6f0642522900dc0412601c176137`。轨迹前20维与共享count继续使用原数学；本策略仅保护地形1024维mean/variance。

## 真实来源与完整状态

- 新931文件与只读复制集逐项SHA/bytes匹配；完整940-member归档包含931个代码/资产成员与9个准备元数据成员，全部代码逐项互证。旧926中921个文件逐位保留，旧5个文件仅声明RMSdelta，另新增5个路径。实际source manifest SHA4227b3d8…，launcher6d56a3ef…。
- 真实GPU同步执行ledger：首段88个owned模块、恢复段87个，全部来自隔离code。缺少final事件按实际保留，不伪造SDK shutdown finally。CPU import清单不被当作GPU清单。
- 两checkpoint SHA分别934f7137… / d3c7b193…；结果和metrics均与root seal逐项一致。恢复metrics前4行与封存首段逐值相同；恢复argv明确引用sealed step4。
- 完整模型、两Adam与metrics累计优化step、四normalizer、replay ring/57有效动作、physical XYZW/69DOF、sameepisode10帧history、actual151/51计划与phase、256起点/192m地面协议均核验。
- 三私有/原corpus RNG与嵌套课程state严格加载，下一次subset reset/expert/plan逻辑抽样重复一致；Python/NumPy/CPU RNG合法。CUDA RNG字节原样保留，CPU审计没有重放CUDA，也不声称PhysX接触/solver cache逐位恢复。

## 推理与质量边界

新隔离实验按单策略要求仍用旧 **完整 PaperAMPNetwork（actor/task_encoder/critic/discriminator/log_std）**，没有夹带现代actor-only裁剪。strict推理实际只读取该checkpoint，不读外部NPZ或公开checkpoint；3个明确synthetic CPU输入与加载trainer的raw/applied actions逐位一致、moments及调用者CPU RNG不变。它们不是实际保存的pre-action GPU观察。

首4更新报告1次训练跌倒/124次时间限制；恢复4报告3次训练跌倒/153次时间限制。两段相加观测4次跌倒/277次时间限制，但counter语义仍按各段保留，不能写成验证导航结果。模型与D在恢复段真实更新，统计保护本身不构成效果改善证据。

[完整2094项核验](actual4to8_complete_v2/audit.json) · [先前必需1150项核验](actual4to8_initial/audit.json) · [先前首4项1108核验](actual4_initial/audit.json)
