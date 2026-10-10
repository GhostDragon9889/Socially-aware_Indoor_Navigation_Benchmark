# 独立保护地形统计训练核验

工具只读实际不可变 checkpoint，CPU严格校验来源、完整恢复、受保护统计和实际同步 import ledger。业务源码与训练预算保持冻结。

- [真实 4+4 核验](actual_pilot_report.md)：2094项通过；旧各工具版本/报告保留。
- [真实 step500 阶段核验](step500_milestone_admission_v1/report.md)：2042项通过，仅 milestone admission，没有 terminal result。
- [可执行命令](commands_v4.json)：实际1000/2000封存通知到达后才运行对应命令。terminal2000还必须有真正结束的result及精确2000行metrics。
- [工具 v4 源码封存](source_freeze_v4.json)：931隔离source身份与文档审计工具身份分开。

`--phase milestone` 禁止 `--result`；只读取对应更新数的完整已提交日志行，保留审计前后atomic GPUledger和初次已提交events前缀。`pilot4/pilot8/terminal2000` 必须提供真正closed result，闭合metrics不允许超预算行。传入文件必须位于已登记sealed目录，拒绝mutable last。

未来terminal schema保留 `protected_terrain_state_action.actual_training_checkpoint_audit.v1`。必须同时满足全检查、实际931来源与owned-import执行证据、完整payload/resume/normalizers/replay/RNG/physical计划合同、公开61931520001参考与200000暖计数分离、保护suffix逐位/数组SHA互证。只有真正fresh2000终态才设置 `actual_terminal_fresh2000_confirmed=true`；未完成阶段不构造passed训练result。

推理探针明确使用三synthetic CPU输入；旧隔离模型保留完整PaperAMPNetwork。CUDA RNG字节保存不等于实际CPU重放，求解器cache未序列化，不做bitwise PhysX恢复或论文质量宣称。9项SDK-free接口测试通过；禁用闭合日志超预算拒绝的临时副本mutation被测试捕获，未改训练/生产文件。
