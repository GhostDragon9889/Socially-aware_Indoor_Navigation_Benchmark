# Post-step AMP：固定协议评价接线

本目录仅准备未来评价。新AMP权重、终态审计、运行凭据和物理结果尚不存在；计划中的未知SHA保持null。

[命令与来源](commands_and_source_plan.json)直接采用已经实际退出0的q936跟踪与四人场景命令。完整22动作命令只改变checkpoint和输出目录，运行前追加真实封存SHA；四人200秒命令只改变AMP checkpoint、对应profile与输出目录。根工具a883b5已检查反向恢复的完整参数数组相等，没有运行评价、导入模型或启动SDK。

生产185中的原benchmark和simulate入口已经支持这些参数，不需要新增控制循环或推理适配。保留完整validation22、seed7、0.5m失败阈值、原位置与旋转统计；四人场景保留旧Diffusion63e、6000控制步、alpha0和原规划／物理配置。失败、接触、idle及reset继续计入各自既有分母。

## 执行顺序

当前40,000次Diffusion训练使用GPU。之后先完成该模型的终态检查与固定AMP对照，再串行执行post-step AMP的fresh4试训、完整payload审计和fresh2000训练。真实2000终态通过完整审计并封存后，才能运行这里的完整22和四人200秒命令。AMP训练与此对照仍使用原Diffusion63e，避免同时改变两层模型。

SDK退出与checkpoint封存顺序见[训练审计合同v3](../state_only_q_poststep_training_audit_preparation_v1/root_tool_owner_finalization_contract_v3.json)。此处不提前创建训练watcher的snapshot目录。

## 原始物理审计

历史q936的raw adapter固定了旧936来源、48ec终态、22b9计划及旧preflight／owner身份，不能直接作为新post-step凭据。待真实新终态和SDK-free preflight发布后，另登记新937来源、实际root seal及capture。仅调整metadata准入，继续调用原f851的`Audit`和`audit_capture`函数，接触、COM、跌倒与时钟数学保持原函数，并在读完后恢复来源回调。

这是后续必要接线，当前raw准入与评价均未执行。MMD不作为启动过滤，封存34条测试输出未用于优化，默认模型与论文效果结论保持现有实际证据状态。
