# 新937 post-step AMP完整检查点审计：最小接线计划

状态：只完成源码检查与接线计划，未创建可执行新审计器，未读取未来检查点或启动NN/GPU。持续训练已有用户授权，执行时间由根代理串行调度。

## 必要修改

1. 原 `validate_state_prior` 唯一语义修改：保存的 `amp_reward_timestamp` 必须等于 `post-step h[t+1..] before resets`，不再接受q936的pre-step声明。D1950、公开D/RMS直接初始化和计数公式全部保留。
2. 完整来源改绑实际937/e57清单、941成员/5f2归档、父936/5c2、唯一runner修改和新增a06配置。原每个归档成员及运行导入逐字节检查保留；新derivation的键为 `inverse_fullbytes_equal`/`inverse_AST_equal`，不能误读旧字段。
3. 原主审计器的bundle/profile/pilot/train/snapshot路径与报告schema改绑新实验。原两Adam、replay、全RMS、q采样器、curriculum和RNG数值验证原样复用；`expert_sampling_checks.py`无需改动。
4. 实际owner与pilot seal需由root登记后绑定确切schema、source、argv和进程关闭记录。已有background-v2修正应保留：无并行训练GPU和完全无GPU进程必须区分。历史helper硬编码ToDesk PID135883及旧owner binding不能充当新实验的真实背景证据。

完整来源身份、函数位置和未确定producer字段见 [source_plan.json](source_plan.json)。未来checkpoint/owner SHA均未编造。

## 执行范围

新独立目录中派生main/helper并保留原q callback即可；根代理登记真实owner后一次窄源码审查，再在真实pilot4和fresh2000终态执行已有完整payload检查。无需重跑旧951项synthetic边界、旧风格/物理指标或新增通用框架。完整payload通过仍不代表post-step模型达到论文效果。
