# 合并后的初始化代码验证

实际生产源码通过44项新增集成测试与224项回归，无deselection，Ruff与生产资源预检通过。原两个运行器AST fixture已修复并纳入回归。新方案将公共PACER判别器的1950维历史权重零扩展至2019维，同时复制其AMP历史归一化统计；初始化的动作权重为零，后续参与训练。默认开关为false，旧模型兼容。

[准入记录](admission.json) · [实际测试证据](cpu_evidence.json) · [生产V11](../native_runtime_epoch_v11/native_pedestrian_source_v11.json) · [试训封存](../state_prior_actual_runs_v1/pilot4_seal.json)

真实pilot4已关闭并封存，完整状态审计与续训仍待完成；未评价风格。首次生产预检使用了错误CLI参数并退出2，原记录保留；使用真实preflight模块的v2退出0，未改代码来绕过检查。
