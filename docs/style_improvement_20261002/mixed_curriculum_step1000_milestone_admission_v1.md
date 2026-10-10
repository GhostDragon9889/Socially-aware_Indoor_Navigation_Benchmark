# Fresh mixed Step1000：独立中间快照资格审核

**161项 CPU payload检查通过；训练仍在运行，不是终点执行或效果验收。** [完整JSON](mixed_curriculum_step1000_milestone_admission_v1.json)、[日志](mixed_curriculum_step1000_milestone_admission_v1.log)。

- 独占 `step_1000.pt`：652,179,357 B，SHA `76708bfa8cc2a55d621f8c3b39a864b66a3b2522eb15d2ca7f451583db165d7d`。
- 已提交1000次更新、8,192,000次转换、32,000控制步，前后128 actor各4,096,000次转换。源135 manifest SHA `8cfb987d4dbd569b2d205c4ec015f29b49c8cb4cb8a6e7a03ed4a92becd18f8e`，archive SHA `5692f2cdc4aed9052a9f3c363b37ba3d8b13167a27c034f78c6486ae56b2ce4f`，逐成员绑定。
- 完整145个 Tensor finite，严格CPU模型/优化器加载接受；原corpus/private AMASS/procedural三套 RNG及real151/proc51有效计划、episode、source phase、刚reset必须replan等合同通过。
- self/task/value RMS count8,392,000，AMP12,488,000，等于初始化cap加已提交batch数。task height1024 std median0.2396686是normalizer状态，不能单凭该值推断导航质量。
- Adam10,063步、D2,000步；16/6 parameter states完整。没有封存metrics前缀，尚未将步数与实际log逐项核算。
- Replay容量/大小65,536，accepted147,057，cursor15,985；动作冻结坐标保持0。
- checkpoint累计课程状态记录真实组70次fall、39,801次clip-end、26次cap-only、1次clip/cap overlap；程序组64次fall、13,543次cap-only。仅为已存状态计数，尚未与终点result对账，不能冒充独立观察事件或最终质量结论。

实际preflight是immutable `argv/resources` 文档。只从冻结CLI注册代码读取参数/defaults，不重跑已有输出目录overwrite guard，不构造GPU `passed` result。实际CPU工具另封存于 [v3](mixed_curriculum_CPU_auditor_actual_source_v3.json)，原135 GPU源快照不变。

没有读取 mutable `last.pt`、增长中的metrics或封存34 test输出。终点仍须真实result、sealedmetrics和500/1000/2000权重对账；未恢复CUDA RNG或PhysX solver/contact cache。
