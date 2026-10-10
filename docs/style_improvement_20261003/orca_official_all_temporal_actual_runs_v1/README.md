# ORCA 两域完整时间窗口：实际运行

本页记录真实执行；目前完整训练运行中，尚无最终权重或效果结论。

## 已完成

- [四次更新](pilot4_actual_root_owner.json)：真实 GPU 会话71112，退出0，1,600窗口；Maps／Interact各16个采样窗口，raw与EMA均完成。实际抽到的均为多人场景，单人指标分支另有六项CPU解析检查。
- [实际恢复至第五次更新](resume5_actual_root_owner.json)：真实 GPU 会话17984，退出0；从封存step4恢复raw／EMA／Adam及随机状态。该恢复门槛省略最终采样，未声称逐位配对恢复。
- [数据构建与检查](../orca_all_temporal_view_staging_v3/README.md)：训练233,046、验证29,952、测试29,466窗口；恢复4,458个零近邻训练窗口。原states／maps及历史筛选视图保留。

短训练证明接口可运行，没有证明模型质量。单行人／多人／混合样本的指标分母与空邻居处理见[冻结实现与CPU检查](../orca_official_all_temporal_formal_preparation_v1/README.md)。

## 40,000 次更新：运行中

[实际运行记录](formal40000_actual_root_owner.json)的GPU会话50502、启动块86eb41。最新实际日志观察910173（2026-10-04 03:39:45UTC）为第39,000次更新：78,000次microstep、15,600,000训练窗口曝光、累计43,147.50秒，EMA验证loss Maps=.10581205／Interact=.08072650；真实工具50502轮询7b2efa仍运行，尚无终态退出结论。完整命令来自[已冻结计划](../orca_official_all_temporal_formal_preparation_v1/commands_and_audit_plan.json)，计划SHA256为`2beed1bd10777658eb1c6ec8e1a2aa3bbc7c8571646ddf8f5fa9f3fbec9ab945`。

训练从原始`4b6f41…/step_1000.ckpt`重新初始化，重置Adam和随机状态；不续训pilot。microbatch200、累积2次、有效批量400、学习率0.0002，EMA系数0.995。批量归一化实际看到200样本，与作者配置的单批400存在差异。运行日志：[formal40000_GPU.log](formal40000_GPU.log)。

[启动前GPU观察](formal40000_GPU_prelaunch_observation.json)记录了真实显存和ToDesk视频会话804926；它保留运行。没有并行训练，但有远程桌面GPU进程。该root进程的exe链接不可读，NVIDIA进程名称、argv0、视频标志与稳定进程标识已核对；未记录端口值。

源码`evaluate_loss`每次重设domain采样seed与Torch noise seed，并恢复原状态；这些日志loss使用相同的四批验证抽样（每批200），不是完整验证集。Maps EMA从第9,000次.03849113升至第39,000次.10581205，当前未呈单调改善；不会以训练loss下降认定收敛或替代终态独立ADE／FDE及物理评价。

已保留[实际14,000步中间字节副本](nonterminal14000_copy_actual_tool_receipt.json)，372,121,656字节、SHA4efe37de…47613；输入before/after及独立副本SHA相同，scalar header由stdlib zipfile/pickletools读取optimizer_updates=14000，未执行pickle或加载Tensor。副本在正式输出目录之外、readonly/nlink1，仅供未来训练结束后的诊断；没有恢复或模型效果验证，不改变40000训练／终态评价，也不替换默认。早先用不存在trained_steps键的metadata探查失败被记录，未重启训练。

[后续中间权重归档器](../../style_improvement_20261004/nonterminal_checkpoint_archive_v1/README.md)已完成六项CPU测试、Ruff及独立源码审阅；真实15k副本已封存，CPU watcher39421已自动捕获16k至39k；继续观察40k header以结束归档，但该header不证明正式训练退出。未变化文件代次跳过重复读取，仅保存独立稳定字节，不加载模型或占用GPU。历史仅用于后续验证集诊断，不改40k主结果，也未按test选权重。

[配对评价归一化修订v2](../../style_improvement_20261004/paired_validation_EMA_normalization_v2/README.md)及[全量验证损失诊断](../../style_improvement_20261004/full_validation_loss_diagnostic_v1/README.md)已完成源码审阅并封存；全量清单实际确认Maps10,566／Interact19,386共29,952窗口。原WholeStateEMA常量FP32舍入通过仅允许配方支持值的新门接纳，保持实际buffers不变；实际checkpoint常量和模型损失尚未读取。正式退出、77025终态审计之后先完成固定配对及物理比较，再运行全量历史诊断。

训练退出后才封存终态、运行CPU终态审计和独立效果评价。默认部署权重保持原选择，封存34条行人测试输出未用于优化。论文作者2848动作ID、八场景清单及指标完整协议尚未匹配，不能将本地结果直接认定为达到或超越论文。

## Actual closed40k terminal

Formal50502 actually closed0 in f10f35, observed2026-10-04 03:59:40UTC. Completed40000 optimizer updates/80000 microsteps/16M exposures/44298.592s. Independent readonly snapshot372121656B SHA `4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff`; sealed owner8603B SHA `b0502a2cceb51fa9ec641ff611071a30cbc7a68af38048286de634f3654030a5`. Initialauditor673299→6770fb exit1 is preserved: JSON config state_mean/std list versus checkpoint tuple, values otherwise exact. [Explicit decoder adapter](../../style_improvement_20261004/formal_terminal_configuration_adapter_v1/README.md) restores only those two types, then calls unchanged77025 strict reader. Actual e87839/session31365→999fed exit0; sealed report6478B SHA `fbe1880fae2b4687165d7caeffd9bf6fd8c3cdedf57a44b26d0030185e5c1e44`. Original validate_resume/raw+EMA+Adam and production EMA equality passed; no forward/optimizerstep/SDK or full GPU resume was executed. Paired and200s physical effects remain pending.
