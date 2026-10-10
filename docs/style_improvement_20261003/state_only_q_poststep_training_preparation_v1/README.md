# Post-step state-only AMP：实际独立训练副本

2026-10-04。**已经实际复制文件并通过配置/资源/隔离导入 CPU gate，尚未初始化模型、训练、仿真或启动GPU。** [已审源码/机制计划](../state_only_q_poststep_timing_preparation_v1/source_freeze_v1.json)是依据；其独立peer Required0和5个真实AST timing spy正例/1个必要反例不冒认物理或模型效果。

## 实际副本

- bundle：`output/isolated_training/mixed_state_terrain_state_prior_stratified_poststep_v1`。
- 实际937 regular代码文件、941 regular archive成员；[manifest及archive登记](registration_prepared.json)。所有936父文件copy时对照原字节SHA、写入后验证SHA且独立inode/nlink1。
- [实际叶差异](actual_source_leaf_derivation.json)：935个共同路径SHA行保持，唯一共同路径变化为runner的原3片段恢复；另新增 `isolated_state_only_q_poststep_absolute.yaml`，旧matched profile仍保留且未被命令选用。
- 原launcher/watch全文字节不变；launcher对当前BUNDLE动态读取manifest并记录实际ownedimports及该manifestSHA，无旧936count/schema硬编码。shared `code/output`只指向新runtime_scratch，其余数据链接和六个输入身份保持；不写原数据/父bundle/生产185。

[实际GPU命令](root_GPU_launch_commands.json)固定newbundle/newprofile/cwd/env：fresh4与另一个fresh2000输出，没有resume。5s/30Hz、256actors×32steps×2000=16.384M transitions、q sampler、1950 stateD、公用actor/critic/D/RMS、奖励scale2/outer2/task.5、PD、terrain、PPO/GAE/replay/Adam均与q936一致。唯一机制变量是post-step AMP history/reward/replay采集时间。未来checkpointSHA/bytes仍null，实际GPUowner和complete-payload审计尚未登记。

## 真实 CPU 闭合

[工具退出记录](actual_CPU_process_closures.json)：

1. 实际copy/inventory/archive：`80763 / 7492a9→390dc3`，exit0。
2. 首gate：`86225 / 0d843d→e7cb2e`，exit1。检查wrapper直接比较resource Path与配置字符串；[失败源、partial import ledger及真实退出](first_actual_CPU_failure.json)保留，不当成完成/通过，不是原模型或资源错误。
3. 将两者按原q936 gate规则 `Path.resolve()` 比较，并使用独立final ledger路径；`67503 / efa748→2585a1`，exit0。[实际readiness](actual_CPU_readiness.json)验证完整937source/link集合、六个opaque输入流hash、机器配置与q936相等、两条fresh命令真实parser/resource gate、实际ownedimports均来自newbundle，且无Torch/SDK模块。

没有torch.load、NN构造、optimizer、Kit、GPU或q bank重建；opaque输入hash不等同模型/动作读取。未读取sealed34输出。严格导入保护与完整新实际manifest绑定仍保留，没有为源证明增加旧suite或新的owner框架。

下一步root审阅实际源/命令后，按GPU串行规则另安排运行并适配既有1950完整payload reader；当前正在运行的Diffusion40k不受影响。完整22/同旧Diffusion的6000control fixture须保留全部失败/idle/reset，对照q936 pre-state、q935 state-action上下文。相同seed不保证随机轨迹/物理bitwise一致；不按MMD筛选启动，不更改默认，不宣称达到论文效果。

持续训练已获用户授权。后续GPU等待root串行调度、当前formal40k可用时间以及实际pilot完整payload验证，不需再次征求用户训练许可。
