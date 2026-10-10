# episode候选首次物理预检：失败保留

2026-10-08 02:02:59（Asia/Shanghai），[owner终态](completion.json)记录8人SDK子进程实际退出1。
App、物理World和训练网络已构建；完成0次更新，清理成功，后续256人／2000更新／评价均未启动。

错误是专家采样器仍要求旧World的文件来源，拒绝了已登记的episode World。
收集器也具有相同的旧来源要求，需要一起修正。修复使用调用方私有模块中显式绑定的新World，
原消费者文件、类型来源检查、采样与控制公式均保留。

[失败训练输出及哈希](failed_training_artifacts.json)只读保留，原smoke8_v1目录不复用。
54份[执行源码快照](source_snapshot_index.json)、原计划、冷预检、日志及终态保留。
后续修复见[v4](../shared_social_episode_after_recovery_v4/README.md)，该修复不改变本次退出1。
