# 地形统计保护候选（未集成）

最小范围、恢复与推理合同见 [集成提案](integration_proposal.md)。当前仅 docs 模块/fixture/config/test，不改生产174或任何旧模型。root/deps完成读审、运行中的15人任务结束后再决定合入并实际4+4 pilot。

- [原型模块](statistics_policy.py)：legacy不写统计；fresh显式来源保护task[20:1044]1024维mean/variance，保留原prefix20/globalcount更新。resume必须匹配调用数与actual update_count。
- [真实公开统计CPU逻辑证据](actual_public_statistics_logic_proof.json)：safe CPU/mmap载入actual public moments，原reference_count61931520001与暖cap200000分离；两次人工零地形moment更新保护正确。不是实际物理rollout或改善证明。
- [CPU专项测试](tests.xml)、[source冻结](source_freeze.json)：具体计数与hash以最终冻结记录为准。真实Torch moments/安全序列化/恢复/篡改/legacy/RNG；没有Kit/GPU。
- [staged profile](amp_style_mixed_state_action_terrain_protected.yaml)与[当前拒绝门禁](staged_profile_gate.json)：仅新增 `task_terrain_statistics_mode`；当前runner不支持，不能静默当legacy执行。旧字段命名门禁保留为历史 [before_field_rename](staged_profile_gate_before_field_rename.json)。

正式实现会从已验证 `warm_start_pacer`对象绑定内部统计并保存完整政策state，部署不依赖这个17KB NPZ。当前原型尚不承担训练器/推理集成证明；未来sharedvalidator/caller职责在提案中逐项说明。没有额外冻结参考moments到旧模型，也没有训练或默认推广。
