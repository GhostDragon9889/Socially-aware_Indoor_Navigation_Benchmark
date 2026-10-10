# 混合 AMASS／程序轨迹课程：终点独立完整性审查

## 结论与范围

**220 项 CPU 检查全部通过**。审查读取独占封存的 step500／1000／2000 checkpoint 与训练退出后复制的完整 result、2000 行 metrics 和 snapshot ledger；未读取可变 last.pt，未启动 SDK／GPU，未读取封存的 34 条测试结果。

完整机器证据：[终点及三个里程碑完整审查](mixed_curriculum_terminal500_1000_2000_independent_audit_v1.json)、[执行日志](mixed_curriculum_terminal500_1000_2000_independent_audit_v1.log)、[原始终点证据复制身份](mixed_curriculum_terminal_inputs_v1/identity.json)。这里验证序列化状态、来源与逻辑恢复合同；跟踪、风格和导航效果应由各自实际评估验收。

## 实际训练与三个封存点

256 个隔离环境固定分配前 128 个 AMASS、后 128 个程序轨迹。fresh 训练完成 2000 updates／16,384,000 transitions／64,000 controls，两组各 8,192,000 transitions，耗时 3827.667372 秒。AMASS 使用 30Hz 真实源前缀、固定 reset 位移及显式线性尾；程序组保存 51 点 10Hz 命令并保留既有插值。课程不调用 Diffusion；Diffusion 属于单独部署阶段。

|封存 update|总 transitions|每组 transitions|control 计数|Actor／critic Adam step|D Adam step|self／task／value RMS count|AMP RMS count|
|---:|---:|---:|---:|---:|---:|---:|---:|
|500|4,096,000|2,048,000|16,000|4,792|1,000|4,296,000|6,344,000|
|1000|8,192,000|4,096,000|32,000|10,063|2,000|8,392,000|12,488,000|
|2000|16,384,000|8,192,000|64,000|20,796|4,000|16,584,000|24,776,000|

每份 checkpoint 为 652,179,357 字节，完整重新计算 SHA256：

- step500：`42939bf6a120e478541eb7ea7209d72158acb075fec30f86931341c8044d084b`。
- step1000：`76708bfa8cc2a55d621f8c3b39a864b66a3b2522eb15d2ca7f451583db165d7d`。
- step2000：`28aed874464379f0b0628719290d36af0a8b0c53691c0dd51f56102c5f428f72`。

实际配置为 `discriminator_input=state`、公共 PACER actor／critic／state D 暖启动、`direction_mode=backwards_penalty`、均匀专家采样和 child_joint 参考速度。它保留明确的工程重建标记，不能当作论文 state_action 公式或作者私有训练配置的精确复现。

## 完整 payload／日志相互核验

- 三份 checkpoint 各含 145 个 Tensor；全 Tensor 有限，严格 CPU weights_only 读取及模型／优化器加载通过。Actor／critic 的 16 个 Adam 状态、D 的 6 个 Adam 状态与实际 metrics 相应前缀的 minibatch 累计数逐项一致；学习率分别为 2e−5／1e−5。
- 归一化器 count、维度和有限方差，replay 65536 容量／大小、cursor／accepted，Python／NumPy／Torch CPU 与课程私有 RNG，256 actor episode／age／plan_age／reset 门禁通过。
- 固定 128／128 分配、所有配置及 runner 合同在三个点相同；真实组 native151 和程序组 canonical51 的源帧／相位／保存计划均通过实际状态恢复验证。三个点 replay accepted／cursor 分别为 105973／40437、147057／15985、228582／31974。
- metrics 正好 2000 行，update 连续 1…2000。原始复制前后及目标文件 SHA 相同；result SHA 为 `39210723bca8707b229e2acd06be8625f597199aa040f59673b16aeeb5e12143`，metrics SHA 为 `13bf0ea5c7ee697dd0d9c6b59b936e1fb4ff88dad9689a65370b82553c287eaf`。

## Reset 与截断的实际分母

终点 **182 次跌倒**＝AMASS109＋程序73；**106,999 次 time-limit／源尾截断**＝真实源尾79,737＋真实 cap-only43＋程序 cap-only27,219。源尾与 cap 同时成立的 1 次已计入源尾，不能再加一次。107,181 次非初始化 reset 恰好为跌倒与截断之和；加初始256后，两组保存 reset 计数80,017＋27,420＝107,437。

真实组保存 620,657 次计划，程序组555,634次，总计1,176,291。真实未来节点31,032,850个，其中17,233,397个超出真实源并按显式线性尾推算，held节点0。它们是工程命令尾部，不是 AMASS 额外真实帧。step500／1000 的计数只采用相应 checkpoint／metrics 前缀，未将终点182次跌倒倒灌至里程碑。

## 来源与边界

GPU 预启动135路径来源清单 SHA 为 `8cfb987d4dbd569b2d205c4ec015f29b49c8cb4cb8a6e7a03ed4a92becd18f8e`，归档 SHA 为 `5692f2cdc4aed9052a9f3c363b37ba3d8b13167a27c034f78c6486ae56b2ce4f`。精确成员、路径安全、size／SHA 与 checkpoint 来源绑定通过。实际 CPU 审查工具另行记录在 [v3 工具归档](mixed_curriculum_CPU_auditor_actual_source_v3.json)，不冒充 GPU 启动时的工具字节，也未更改冻结的业务源码。

AMASS v7 manifest SHA `68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`，bank SHA `46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`，公共 PACER SHA `bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615` 均明确绑定。2848个片段来自1853个父源，作者原始 clip manifest 未提供。

Snapshot watcher 的 transport ledger 仍明确 `semantic_resume_validation_performed=false`；本独立完整审查补足 CPU 逻辑准入。CUDA RNG 仅检查序列化字节，未在 CUDA 恢复；PhysX solver／contact cache 的逐位重放、后续导航质量与作者实验复现均未由本报告证明。
