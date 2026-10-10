# 部署ABI补充与已完成的GPU恢复检查

这是017cd reader的窄补充，精确delta见[production_deployment_ABI_reader.diff](production_deployment_ABI_reader.diff)。
不改变40k命令、训练代码、预算或NN数学，未执行新NN／CPU审计。

未来同一CPU fork边界先调用真实生产185的
`CrowdSim.humanoid.diffusion.PaperDiffusionPlanner(checkpoint_path=snapshot, device='cpu')`：
只strict load原`model`字段，逐tensor确认其与payload EMA相同且step40000，然后释放实例。
后续原staged raw／Adam strict load保持。没有forward／optimizer step／CUDA恢复。
生产paper.py、package入口与四个直接NN/helper文件的身份单独匹配185 epoch，不重核全185。
来源记录和reader新SHA登记在[commands_and_audit_plan.json](commands_and_audit_plan.json)。

原准备的GPU resume1已由root实际运行完成，**不得重复启动**：

- session75232，bd04d8→7f61e5，真实exit0；step4恢复后达到累计step5／10microsteps／2000窗口。
- [实际root owner](../orca_official_formal_actual_runs_v1/resume5_micro200_accum2_actual_root_owner.json)：
  4709 B，SHA `ae5eab4be2f3f686c750fe48e9d5e4a17fad95fee2deb57e853b93d95afd73c1`。
- root实际调用原restore_training，包括saved CUDA topology／raw／EMA／Adam／samplers；
  未进行两次GPU下一更新的bitwise对照，不能把API成功称bitwise恢复证明。
- step5 SHA `3c788546e1ce7669a48fd589dbf6b5b26b314b1c4a6658396df80691cdba73b8`仅据root实际记录引用，
  此准备没有读step5或旧4b6权重。

正式40k仍从原4b6 fresh开始，不继承pilot／resume5。
JSON保留原resume argv仅为历史准备，status标注already_actually_completed、future_launch_authorized=False；
同文件绑定实际成功200×2 pilot owner71c532与resume ownerae5eab。没有新launcher或重复检查。
