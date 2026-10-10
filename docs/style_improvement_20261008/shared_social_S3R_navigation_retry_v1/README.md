# S3-R独立评价续跑

训练已完成12500有效更新，恢复段500更新、4096000转换；本批不再训练。
原v2评价的actor CPU、全部22条跟踪／冻结风格、30步导航短测实际退出0。
200秒导航保存完整数据后实际退出-11，原post进程实际退出1；54份输出保持只读。
GDB同预算复查实际退出0，未复现故障，没有定位根因。

本批仅重跑无GDB的200秒／6000步导航及原CPU数组审计，保留成功前缀与失败记录。
使用相同S3-R终态、Diffusion40k、N4、物理／接触／路线／评分实现；不放宽原退出或清理条件。
显式续跑凭据记录两次执行的来源，不声称原post进程成功，不认证论文效果或修改默认权重。

2026-10-08 01:20:09（Asia/Shanghai）owner360200／SDK子进程360245启动。
启动工具`196596`退出0仅表示owner启动；真实存活由`e62465`确认。
2026-10-08 01:34:43（Asia/Shanghai）[组合评价终态](completion.json)已生成：
导航200s和原CPU数组审计的子进程实际退出均0，清理成功；新SDK／owner进程均已结束。
CPU核对完整6000控制步／24000 actor-controls：0跌倒、0静态接触、0行人接触，路线节点[1,1,1,1]。
同episode根部速度均值0.126079m/s；路线质量仍不足，执行成功没有认证论文效果。
原失败保持，根因仍未知；不能据一次正常重跑宣称SDK故障已修复。
已有IsaacSim6.1外部进程358046保持，开始时空闲11022MiB，非GPU独占。

## 准入与检查

- [readonly计划](owner_plan.json)：26份来源及54份保留输出哈希绑定。
- [执行状态](owner_state.json)：独立OS owner以`Popen.wait()`记录实际子进程退出。
- [准备入口](prepare.py)实际工具`61d31b`退出0，未导入Torch／SDK。
- 新增续跑6项CPU测试、13个subtest实际`4dfe6f`退出0；Ruff `087219`通过。
- 合并恢复评价／比较／续跑26项测试、34个subtest实际`1da564`退出0；Ruff `dbab9e`通过。
- 24份执行Python源码已保存为只读快照，见[source index](source_snapshot_index.json)。
- [独立对比S2](../shared_social_S3R_comparison_v1/README.md)实际CPU退出0，72份来源／捕获／数组严格核对。

新输出：`output/evaluation/shared_social_S3R_navigation_retry_v1/`。
生产入口：[resume_shared_social_recovery_evaluation.py](../../../scripts/resume_shared_social_recovery_evaluation.py)。
只执行一次已登记的重跑；失败保留，不自动反复重试。
