# S3-R导航退出段错误的原生诊断

正式6000步／200秒运行写出完整数据后实际退出-11，原OS owner记录pipeline退出1。
`SimulationApp.close()`与现有弱引用清理检查返回成功，退出阶段仍需原生调用栈定位。
当前不能声称SDK清理修复或整条评价完成。

本次使用相同权重、输入和6000步物理／评分代码，增加GDB原生诊断和Python faulthandler。
没有修改控制、物理、模型、评分或原成功退出准入；属于诊断运行，不先行认证质量。

2026-10-08 01:12:47（Asia/Shanghai）独立owner记录`gdb -return-child-result`实际退出0。
日志末尾明确为`Inferior 1 ... exited normally`，6000步完成，未复现段错误，因而没有SIGSEGV调用栈。
终态已核对，owner330620／debugger330621均不再存活。这不能证明故障根因已定位或SDK已修复。
正式无GDB的[一次独立续跑](../shared_social_S3R_navigation_retry_v1/README.md)另行记录实际SDK退出与原始CPU审计。

- [当前OS owner状态](owner_state.json)
- [实际调试终态](completion.json)
- [原生调试日志](gdb.log)
- [启动与执行源码](owner.py)

输出：`output/evaluation/shared_social_S3R_navigation_gdb6000_v1/`。
此前跟踪／风格的成功数据以及导航-11失败数据均已保留。

## 上游资料核对

2026-10-08核查[NVIDIA IsaacSim6.0.1已知问题](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/known_issues.html)：
扩展卸载早于相关physics.tensors对象回收可能导致崩溃，建议先释放这些对象。
现有本地生命周期已经执行对象解绑／GC及弱引用归零；本次失败的清理字段均成功。
这条上游资料提供排查方向，不能据此认定本次故障相同，也不能用“可忽略”描述替代实际成功退出。
没有证据支持调整模型、物理参数或跳过原退出准入。
