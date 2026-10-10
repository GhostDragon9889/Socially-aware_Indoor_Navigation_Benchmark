# 地形统计保护：真实 step500 阶段核验

**2042/2042 项独立 CPU 检查通过。** 仅准入实际封存 step500；训练仍在运行，没有读取或构造 terminal result。

| 项目 | 实际值 |
| --- | --- |
| 更新 / 总 transitions | 500 / 4096000 |
| 各来源 transitions | 2048000 / 2048000 |
| policy hook calls / task 共享 count | 500 / 4296000 |
| 公开参考原计数 / 暖启动计数 | 61931520001 / 200000 |
| AMP count | 6144000 |
| policy / discriminator Adam step | 3730 / 1000 |
| replay size / accepted / cursor | 65536 / 105669 / 40133 |
| 累计训练跌倒 / 时间限制 | 368 / 26515 |

checkpoint SHA `4c9c964a760914c90191cb0227cd327c3144a2a373f307a3f9bf82f8baca3d7d`，653044761B，与实际独占封存 ledger 一致。完整模型、两 Adam、四 normalizer、57 有效动作 replay、common XYZW/69DOF 状态、same-episode 10 帧 history、real151/procedural51 计划相位与三私有/共享 RNG 均严格恢复。公开 suffix mean/variance 与冻结17KB公开统计 fixture逐位一致，数组 SHA `5ebd61c10871dfad790d1dbbf6e9d1e05e6e6f0642522900dc0412601c176137`；task前20维和共享count没有被保护策略冻结。

隔离931源码与940-member完整归档逐项互证。GPU import ledger审计前后实际 committed sequence均90、87个owned模块，全来自隔离副本；runtime finalization=false按实际保留。只解析已提交的500行metrics前缀，不把之后仍在增长的日志当作本阶段证据。下一次CPU逻辑reset/expert/plan抽样恢复重复一致；未执行CUDA RNG或物理求解器重放，不能声称PhysX cache逐位恢复。

推理检查只读取此checkpoint，不读外部统计或公开权重，三个明确synthetic CPU输入与加载trainer输出逐位一致。该旧隔离epoch保留完整PaperAMPNetwork，不能描述为现代actor-only裁剪。以上训练跌倒计数不是导航验收；没有风格、导航质量或论文效果结论。

[完整实际检查](audit.json)
