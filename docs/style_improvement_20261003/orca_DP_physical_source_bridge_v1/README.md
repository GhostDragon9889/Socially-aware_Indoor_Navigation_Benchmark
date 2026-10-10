# 旧 native 运行与当前 185 源码的有限比较

实际只读比较了旧 native4 的完整 168-source archive、旧 native15 的完整 174-source archive，以及当前 V12 的完整 185-source archive。每个 archive 成员重新校验 SHA/bytes，再比较完整文件及 changed Python functions 的 AST。没有启动 SDK/GPU或读取模型 tensor。

| 比较 | common 文件完全相同 | changed existing | added | removed |
| --- | ---: | ---: | ---: | ---: |
| old native4 /168 → current185 | 152 | 16 | 17 | 0 |
| old native15 /174 → current185 | 165 | 9 | 11 | 0 |

[完整来源清单](source_inventory.json) 给出了 **20 个全部文件字节相同** 的直接相关 core：AMP controller/network/observations/world、Diffusion planner/paper、EvaluationActor、deployment/native_deployment、physics views、真实 contact/foot/support/reset，以及 IsaacLab simulator/config/scene/rotations。旧 native15 的 DynBench/native grid/visual provider 与当前版本也完整字节相同。

剩余差异不能概括成所有 runtime 源码一致：

- 新 `ReconstructionAMPInference.from_checkpoint` 增加 protected terrain 的条件验证及 JSON metadata；`validate_bundle_identity` 增加 state-prior/terrain 保存状态 admission。
- TrainingConfig 增加默认 `False` 的 action-state-prior 开关和默认 `legacy` 的 terrain mode，以及对应验证。资源 preflight 增加 from-scratch admission；runner 新语句仅在非 evaluation resume 中比较 reward 配置。
- PaperAMPTrainer 的初始化/加载/观察/保存/更新/warm start 增加 state-prior 和 terrain policy 接口；其方法 AST 确实不同。不能仅根据相同 actor checkpoint 宣称整个路径相等。
- 相比旧 native4，当前 map admission 还增加 V8/V9 支持，原 recorded-visual replay 的 source composition/Render namespace admission 也不同。这里记录具体函数差异，没有穷举或证明所有运行分支等价。

因此，旧完整 200 秒记录仍是历史基线，无法作为新 current-source 运行的严格单变量配对。下一轮应在**同一登记来源、默认 62c4 controller、profile、场景、route/seed/guidance/counts** 上分别执行旧 63e 与新 b563 planner，保留全 4 actors ×6000 controls/200 秒及每个失败。fixture 完整 zero fall/static/peer 才进入 native4，native4 完整物理 zero 才进入 native15；不依据风格分数选择是否运行。

若 fixture 与 native 分别采用 isolated935 和 production185，登记应明确两个 epoch，并保证每一对 old/new 共用各自的同一 epoch。这里的 source inventory 没有产生新的物理准入或 planner 质量结论。
