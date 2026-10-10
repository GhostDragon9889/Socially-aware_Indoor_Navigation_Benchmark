# episode候选第二次物理预检：CUDA时钟边界失败

2026-10-08 02:26:20（Asia/Shanghai），[owner终态](completion.json)记录8人SDK子进程实际退出1。
App、World、网络、专家采样器与收集器均已构建；在首次参考查询处停止，完成0次更新，清理成功。
后续256人／2000更新／评价均未启动。

实际独立GPU诊断复现：float32 `150 / 30` 在CUDA为`5.000000476837158`，CPU为`5.0`。
原参考查询严格拒绝大于五秒的偏移；此前纯CPU验证没有覆盖此差异。
后续显式私有派生对30Hz除法先计算float64，再舍入为float32；边界检查保持，未加入偏移clamp。

[失败训练输出及哈希](failed_training_artifacts.json)只读保留，原smoke8_v2目录不复用。
55份[执行源码快照](source_snapshot_index.json)、原计划、冷预检、日志及终态保留。
后续实际CPU／CUDA资源验证与物理预检见[v4](../shared_social_episode_after_recovery_v4/README.md)。
