# 固定槽位混合课程：独立 CPU 审核

状态：**源码及 CPU 语义门禁通过；本报告未运行物理模拟，未验证实际训练终点。**

## 已核对

- 冻结源码及测试的全部 SHA/尺寸与 `mixed_curriculum_source_freeze_v1.json` 一致。
- 独立 [8 项真实 AMASS 测试](../../tests/test_mixed_curriculum_independent.py) 全通过，未跳过。检查任意 actor 请求顺序、最新帧在前的十帧历史、真实151帧/程序51点计划、线性源外尾部、非实时重锚、异步源结束、恢复后下一次采样和三套 RNG 的 CPU 转运。
- 完整 `for step in range(length)` AST 与修改前相同，SHA `1adbfe51b09a2311b3e450925cface7b4f81a1fbb122dd945838d16a9272a74f`。原 PPO、功率测量、动作投影、终止及保存时序未在控制循环内改变。
- 全部 `TrainingConfig` 字段和值与原 AMASS-root profile 相同。旧124文件快照中的运行源码仅新增整合所需的 world、configuration、runner 三处改动；模型、优化器、奖励与专家采样代码保持原冻结字节。

机器证据：[独立审核 JSON](mixed_curriculum_independent_review_v1.json)、[冻结版测试日志](mixed_curriculum_independent_cpu_frozen_v2.log)、[JUnit](mixed_curriculum_independent_cpu_frozen_v2.xml)。模块 owner 另有30项测试和256 actor/15控制步的128项真实数据验证，分别记录在 `mixed_curriculum_component_v1`；这些是 CPU 证据。

## 课程与恢复合同

固定前128个槽位使用 AMASS 根部 XY，后128个槽位使用原 smooth 程序命令，速度上限2 m/s、加速度上限1.5 m/s²。每控制步恰好各128次实际转换；重置或规划次数不要求相等。

AMASS 组保持 `f0 + age` 真实相位、固定 reset 平移和151点30 Hz参考。仅源外未来命令用 `(root[-1]-root[-11])/(10/30)` 线性延长，源结束仍真实截断。程序组保留原51点10 Hz曲线和旧插值；151点仅用于统一保存与查询。actor 不接收组别标识或未来3D姿态。

三套 RNG 的职责为：原 corpus 负责程序组 reset 和原专家采样，独立 `seed+1` corpus 负责 AMASS reset，独立 `seed` 负责程序计划。top RNG 与程序子状态是同一 RNG 的重复序列化。实际 CUDA map-location 会搬迁嵌套 RNG；driver 已明确只把这三处 RNG 转回 CPU，再执行严格恢复。物理接触和 solver cache 未序列化，不能声称恢复后 PhysX 字节相同。

原 AMASS 全2848动作、409352合法起点和全部256个 spawn 的根命令边界证明适用于其前半子集。程序组10秒 episode 加5秒未来命令的保守坐标范围 `[18,138]` m 位于192 m地面内。这是命令边界，未约束真实跟踪误差、全身碰撞体或接触稳定性。

## 封存与后续验收

`watch_mixed_amp_snapshots.py` 的独占发布和复制双 SHA 机制可用于500/1000/2000步快照；其 ledger 明确没有语义恢复验证。跳过目标必须记录 missing，不能把邻近步数冒充目标步数。

[完整 CPU 审计器](audit_mixed_actual_checkpoint.py) 已准备，待实际4/8及终点工件出现后验证严格模型加载、全 Tensor finite、Adam、RMS、回放、三个 RNG、两组有效计划、episode/clock、source 与分段 reset/clip-end/cap。早期快照不会继承终点跌倒计数。

计划 fresh2000 为16,384,000次转换，各组8,192,000；旧 fresh1000 总预算为8,192,000。后续比较需同时披露混合分布和预算变化，不作单因归因。执行完成与恢复门禁不代表风格、追踪或导航质量通过，也不代表作者完整实验复现。本审查没有读取封存34 test 输出。
