# Actual state-action isolated fresh2000 terminal admission

**实际 CPU 执行：1079 checks PASS；旧 v1 evaluation adapter admission 已实调用通过。**

- Checkpoint：`output/diffusion_amp/amp_style_mixed_state_action_isolated_snapshots_v1/step_2000.pt`，653,027,293 bytes，SHA256 `eeac037798d2244e0cd4f0a81b40f51020c2aa3320106c7f10620e8c2ddde526`。
- Actual update2000 / 16,384,000 transitions / 64,000 control steps；AMASS/procedural 各8,192,000 actual actor transitions。组曝光不等于 reset/plan 各半。
- 实际终态 result：passed，elapsed4,886.308517707秒，累计训练跌倒429、time-limit106,839；`convergence_claim=false` 保留。训练完成不是 gait/nav/论文质量通过。
- D 输入2019 = 10×195 history +69 effective normalized action；public actor/critic/moments 转移，public state-only D/AMP RMS **未转移**；expert action 是 next-frame normalized PD position proxy，非测量专家控制。
- PPO Adam18,218 / discriminator Adam4,000；self/task/value RMS counts16,584,000，AMP RMS24,576,000。Replay size/capacity65,536，accepted229,144/cursor32,536；accepted≠physical transitions。
- 完整 model/Adam/RMS/replay/3global RNG/2curriculum child RNG/physical runner state 的 CPU restricted-mmap 恢复通过；后续逻辑 reset/expert/plans 精确复现。**未执行 PhysX/CUDA，不声称逐位物理恢复等价。**
- 真实已观察 GPU owned imports86、隔离复制source926逐项 SHA通过；同步ledger已有实际记录，SDK退出后的 finalization仍false，未伪造。终态由真实 result与checkpoint证明，未用 finally 事件代替。
- 共享大bank只复核link/size/mtime，不重复1.15GB整库SHA。未读sealed34、未改checkpoint/训练目录/业务源码。

## 评估 admission

可传给 `run_isolated_state_action_evaluation_v1.py --checkpoint-admission` 的文件是本目录 [audit.json](audit.json)。Schema仍 `state_action.actual_training_checkpoint_audit.v1`，sole2000行及实际来源完整，另有 `actual_terminal_fresh2000_confirmed=true`。

[全部payload审计](full_payload/audit.json)、[真实终态副本](actual_terminal_result.json)、[执行日志](execution.log)、[来源封存](source_freeze.json)。

下一步具体 full22/200s CPU/GPU分工命令见 [准备说明](../state_action_post2000_evaluation_preparation_v1/README.md)。尚未执行新模型跟踪/导航效果；不会从checkpoint训练计数推断效果。
