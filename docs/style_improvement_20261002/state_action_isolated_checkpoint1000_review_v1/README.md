# 隔离state-action：真实step1000独立恢复审计

2026-10-02。**沿用step500逐字节冻结的审计工具，1077项CPU检查全部通过**；没有修改step500或正在训练的目录。step1000已由唯一watcher封存：653027293 bytes，SHA `66e03fadd0aabf02210d7427cb3f4d86d6ecadc639395c294a70072f0b1e00e2`。

|项目|实际step1000|
|---|---:|
|累计更新 / transitions|1000 / 8192000|
|累计控制步|32000|
|AMASS / procedural actor-transitions|4096000 / 4096000|
|PPO Adam累计step|8930|
|D Adam累计step|2000|
|self/task/value RMS count|8392000|
|AMP RMS count|12288000|
|replay size / capacity|65536 / 65536|
|replay accepted / cursor|146839 / 15767|
|累计训练fall / time-limit|330 / 53397|

2019D为history+action，公开state-only D和AMP RMS均未迁入。原full后keep_probability=.01使replay accepted不等于transitions；12冻结动作坐标为0，57有效。两个真实Adam的step与metrics前1000行实际minibatch累计精确一致，完整model/optimizers/RMS/replay/RNG在旧隔离类CPU加载后逐tensor相同。

实际v7、192米floor/48米origin、256actors的native151/原proc51计划、sourcephase/episode/reference/clocks完整恢复通过。恢复后乱序subset reset→expert→plan再恢复返回张量和三个课程/语料RNG字节相同。CUDA RNG仅保留序列化bytes，未在CPU执行；没有求解器/接触缓存，因此不声称逐位PhysX恢复等价。

926复制文件全SHA通过；v3同步ledger实际观察86个owned执行模块逐SHA匹配。保存审查时已提交ledger/events，`runtime_finalization_observed=false`原样保留，不虚构后续模块或训练终态。大共享bank仅核对声明link/size/mtime，没有重复完整hash；manifest真实SHA和checkpoint数据provenance一致。

这不是完整2000训练、validation/导航成绩或收敛判断。训练fall/time-limit是累计训练事件，不能当模型闭环质量。全checks/metrics/RNG及来源见[审计JSON](audit.json)、[CPU日志](audit.log)、[实际observed ledger](GPU_import_ledger_as_observed.json)、[提交events](GPU_import_events_as_observed.jsonl)、[真实metrics前1000行](training_metrics_committed_prefix.jsonl)、[证据冻结](source_freeze.json)。审计工具[与step500同字节](../audit_state_action_training_checkpoints_v1.py)，其既有Ruff gate保留。

CPU-only：未启动Kit/SDK/CUDA、未读取sealed34或修改业务/模型/快照。待实际step2000与终态result完成后单独审查，不提前填充不存在的最终效果。
