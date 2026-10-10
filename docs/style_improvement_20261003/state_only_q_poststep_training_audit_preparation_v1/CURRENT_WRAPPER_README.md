# 当前post-step reader：环境合同修正

当前入口 `audit_poststep_training.py` SHA `ae502054ae34fa77f48cb6dbb881cab4b7ed1d47ded37dc04ee3e6c41d4bab9a`；当前模板 SHA `79efb695804b7f1f0858f2ad1e610e9995f1232ca53ea8cc49058eea9f62b20d`。

独立审查和root审查发现，原登记计划的6个环境字段不包含CUDA选择，真实训练额外设置 `CUDA_VISIBLE_DEVICES=0`。当前binding显式记录7字段 `actual_training_env`，入口先精确核它等于原计划环境加CUDA0，再精确比较实际root receipt环境，不允许其它字段。见 [环境合同v2](root_tool_owner_env_contract_v2.json)。历史444合同及README保存；其中旧env==plan.env描述由此补充替代。

标准库检查工具1c0c90实际exit0，wrapper及派生源码语法通过，原18/11项声明替换整bytes/AST反向还原原源码。未导入候选、NN/SDK/GPU或未来检查点。数值检查、时间戳及真实工具会话凭据合同均未再改变。

仍需root发布真实会话receipt及背景观察来源，模板当前不可执行。持续训练已有用户授权，等待root串行硬件调度；不新增用户确认步骤。
