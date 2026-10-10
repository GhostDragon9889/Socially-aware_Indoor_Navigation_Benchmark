# 隔离 state-action：真实step500完整CPU恢复审计

2026-10-02。**1077项检查全部通过，Ruff通过**。此报告核验已封存的真实step500，不声称完整2000训练已完成，也不声称导航效果或逐位PhysX恢复。

- checkpoint：653027293 bytes，SHA `3b385a34777ac55d5ee1be7e46abf4b388cf0c26c5d9454a179fd68e7e76e1b0`。唯一transport watcher的size/SHA一致；未写训练或快照目录。
- 实际500更新 / 4,096,000 transitions / 16,000控制步；两组各2,048,000 actor-transitions。完整native151/原proc51计划与sourcephase/episode/reference/192米地形origin的CPU恢复通过。
- 2019维D为history+action，公开state-only D和AMP RMS均未迁入。PPO Adam step 4058、D Adam step 1000与真实metrics前500行的实际minibatch累计精确一致；非零optimizer moments与全部训练tensor有限。
- self/task/value RMS各4,296,000；AMP RMS6,144,000。replay capacity/size65536，accepted=106034 / cursor=40498，原full后keep_probability=.01使accepted不等于全部transitions，冻结12坐标为0、有效57，完整replay恢复逐tensor一致。
- 实际Python/NumPy/Torch CPU RNG和课程三RNG有效；恢复→乱序subset reset→expert→plan→再次恢复逐tensor/三RNG完全一致。CUDA RNG字节保留，不在CPU执行；PhysX solver/contact缓存未保存，因此不称物理轨迹逐位恢复。
- 同步v3 ledger实际观察到86个owned执行模块，逐个与只读CODE登记SHA匹配；926复制文件全部hash。已独立复制当时的JSON与commit events前缀。`runtime_finalization_observed=false`原样保留（审查时长train尚运行）；没有虚构后续imports或终态。
- 大共享资源只核对link/size/mtime，不重复1.15GBbank完整hash；manifest实际SHA与checkpoint数据provenance一致。未读取sealed34、启动SDK/GPU或改业务/数据/模型。

训练截至该milestone的累计fall=292、time-limit=26573，是训练事件，不是完整validation/导航成绩。原始metrics/完整失败数保留。

证据：[1077检查](audit.json)、[CPU日志](audit.log)、[实际GPU observed ledger](GPU_import_ledger_as_observed.json)、[commit events](GPU_import_events_as_observed.jsonl)、[真实metrics前500行](training_metrics_committed_prefix.jsonl)、[Ruff](ruff.log)、[CPU审计工具](../audit_state_action_training_checkpoints_v1.py)。

工具初始对自动actionmask与replay full后随机保留机制的错误假设产生失败，三份失败报告/当时工具/日志保存在相邻`*_failed_tool_*`目录；修复只限docs审计器，原模型/配置/训练未改。最终以原实现的实际规则验收，不用错误全量accepted假设。
