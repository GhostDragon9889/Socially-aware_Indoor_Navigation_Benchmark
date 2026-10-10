# Pacer 外部 AMP 权重 0.5 完整状态 CPU 审计器

只适配已冻结 935 v2 副本、profile、root plan 和审计 metadata。训练实验的唯一数学变量是外部 AMP 权重 2 → 0.5；内部 scale 2 保持，公共 state-prior D/RMS coupled 初始化保持。两个训练 resume guard 修改仅拒绝五个奖励配置字段混用。

`metadata_adaptation_CPU_proof.json` 是实际 SDK-free 源码证明：main 去掉三条新增奖励字段报告/准入语句并逆向路径、schema、935计数等 metadata 后，全文原字节及 AST 完全等于原 2c60 审计器。helper 除 `_verify_launch` 与 `verify_source_archive` 两个 metadata 函数外的全部六函数 AST 精确相同；前者仅新增真实 source artifact 和 child closure 身份，后者验证新 archive 与 932 原字节 + 2 resume guard + 1 profile 推导。原 policy/stateprior/metrics/result 检查保持。新独立 archive 包含 935 code + 4 prelaunch metadata，共 939 regular members。

没有在准备证明里加载 checkpoint、模型或未来数据。953 真实 source/closed pilot launch metadata 检查通过，旧934来源和缺少一个guard的derivation两个反例被拒绝。CPU proof不等于训练结果审计。

实际执行命令见 `commands_CPU_audit_v1.json`。pilot4 必须 root 真正 closed0 后只读封存 step4/result/metrics/GPU ledger；milestone 只消费 watcher 已复制且 OS readonly 的 500/1000；terminal 要 actual closed2000结果、sealed metrics、step2000及 root closed launch。终态报告固定本目录 `terminal2000/audit.json`，现在不预占。

报告 schema `pacer_reward.actual_training_checkpoint_audit.v1`，顶层 `reward_contract` 精确五字段：amp_reward_scale=2.0、amp_reward_weight=0.5、task_reward_weight=0.5、direction_mode=backwards_penalty、amp_probability_epsilon=0.0001。各 checkpoint row保留全部模型/Adam/replay/RMS/RNG/terrain/stateprior/curriculum完整状态检查。pilot不是终态，milestone不是终态；物理求解器未回放，CUDA RNG bytes保留而不CPU执行。原 inference 为完整网络保留，不声称最新 production actor-only pruning。

只读代码冻结后由独立 peer 审核，只有 root 启动 GPU；默认不 promotion。
