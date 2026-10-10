# Fresh mixed Step500：独立中间快照资格审核

**161项 CPU payload检查通过；训练仍在运行，不是终点执行或效果验收。** [完整JSON](mixed_curriculum_step500_milestone_admission_v1.json)、[日志](mixed_curriculum_step500_milestone_admission_v1.log)。

- 独占 `step_500.pt`：652,179,357 B，SHA `42939bf6a120e478541eb7ea7209d72158acb075fec30f86931341c8044d084b`。
- 已提交500次更新、4,096,000次转换、16,000控制步，前后128 actor各2,048,000次转换。源135 manifest SHA `8cfb987d4dbd569b2d205c4ec015f29b49c8cb4cb8a6e7a03ed4a92becd18f8e`逐成员绑定，当前 CPU loader使用相同冻结业务源码。
- 完整145个 Tensor finite，严格CPU模型/优化器加载接受；原corpus/private AMASS/procedural三套 RNG及real151/proc51有效计划、episode、source phase、刚reset必须replan等合同通过。
- self/task/value RMS count4,296,000，AMP6,344,000，均等于初始化cap加已提交batch数。
- Adam4792步、D1000步；16/6 parameter states完整。未提供封存metrics前缀，尚未将这两个步数与实际log逐项核算。
- Replay容量/大小65,536，accepted105,973，cursor40,437；动作冻结坐标保持0。
- checkpoint累计课程状态记录真实组44次fall、19,838次clip-end、14次cap-only；程序组57次fall、6746次cap-only。这些是**已存状态计数**，没有终点result对账，不能冒充独立观察的跌倒事件或最终质量结论。

实际训练preflight为immutable `argv/resources` 文档。审计仅使用被冻结的CLI注册代码读取参数/defaults，不重跑针对已有输出目录的overwrite guard，不构造`status=passed`的GPU运行result。第一次工具用错误的`arguments`字段读取pilot preflight而失败，原失败log/JSON保留；修正文档审计器后才生成本次Step500报告，没有业务改动。

1.155 GB train bank的完整当前SHA重新核算为 `46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`，见[独立绑定](mixed_curriculum_step500_actual_bank_binding_v1.json)。实际CPU审计代码与依赖另封存 [v3](mixed_curriculum_CPU_auditor_actual_source_v3.json)，保留v2与原135 GPU源快照。

没有读取 mutable `last.pt`、增长中的metrics或封存34 test输出。终点将继续用真实result、sealedmetrics及500/1000/2000权重逐项对账；本次没有恢复CUDA RNG或PhysX solver/contact cache。
