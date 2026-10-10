# 真实根工具会话对应的新审计入口

本补充采用 `state_only_q.poststep_actual_training_root_owner.v1` 真实工具会话凭据，取代历史source-plan中可能被理解成必须存在可执行owner的假设。没有Popen启动器，没有伪造child PID或owner源代码。

`audit_poststep_training.py` 复用原q936完整reader和原q sampler验证函数，仅派生来源/报告/归档身份及唯一post-step字符串guard；root会话与pilot seal的退出字段改为真正的 `actual_exit_code`。字段及CPU命令前缀见 [root_tool_owner_contract.json](root_tool_owner_contract.json)。

原主reader/helper的11/18项声明替换已经标准库语法检查，并在内存中反向还原整文件bytes/AST。工具3d0c05实际exit0。未导入候选、未执行数值函数、NN/SDK/GPU或未来检查点；这不是模型payload已经通过。

模板 `root_owner_binding_TEMPLATE.json` 明确未登记、未来SHA为null，因此执行入口会在数值reader运行前拒绝它。Root实际执行完成后发布独立readonly binding，绑定真实receipt以及真实GPU背景观察helper。该helper只须提供 `validate_observation(evidence)`，按root已有安全观察语义验证真实NVIDIA记录；这里不硬编码ToDesk PID，也不沿用历史owner背景binding。

持续训练已有用户授权；根代理串行硬件调度及真实pilot验证安排后续执行。旧训练源、审计、数据、生产185及默认模型均未改动。
