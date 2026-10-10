# AMASS root → Diffusion导航：只读迁移诊断

这是CPU checkpoint/source诊断，不是新训练或normalizer干预。实际四人200s仍在运行；完整失败数据将保留。

## 已确认的差异

- 两个1000模型的terrain1024 RMS mean/variance/count逐元素完全一致、差值0（task count8392000）。因此不能将新部署退步直接归因于“terrain归一化被AMASS训练改坏”。
- 其他self/trajectory20/AMP/value RMS和actor/taskencoder/critic/D权重都变化；单靠静态差异不能分离因果。
- heading-relative5s forward target均值2.773729→1.332044m，表达不同命令分布；这是训练统计，不是当前闭环指令。
- 完成episode平均 298.411052→102.085002控制步（9.947035→3.402833s），中位300→87，根课程source-end reset明显频繁。
- 55.628111%未来XY knot位于原动作结束后，按明确线性命令尾部外推；episode仍在最后真实动作帧截断，无假3D运动延长。
- full22任务消费真实未来rootXY；部署消费学得Diffusion、causal reference、障碍/同伴与经常低近时域指令，不能由full22成功推导闭环稳定。

## 下一实验建议（尚未接入）

以公开PACER同一warmstart重新开始，保留negative-backward、PPO/D/RMS/rig/PD/v7专家采样；第一轮只改变为明确的程序化+AMASS-root混合命令课程。先目标80%程序化/20%AMASS**控制transition**，不自动五类均分，不同时启用新的专家采样或损失。

episode长度不均：如果只按reset设置20%AMASS，实际AMASS transition约8%；按当前均长估计，需要约{episode_p:.3%}AMASS episode概率才能接近20%transition，实际长度受falls变化，必须记录实际actor-step占比，不把该估计当保证。

每个episode固定mode，AMASS保持合法frame/fixedoffset/realclipend，程序化保持原速度/朝向生成与300步。共享actor task[B,10,2]接口；禁止输入未来3Dposes。完整保存mode、两类私有RNG、source-phase/procedural状态、每类transition/reset/truncation计数。真实sourceend仍truncation，不伪延长或用物理teleport保稳定。

评估继续固定full22+c274风格，和同guide20/eta0四人200s真闭环/原生ground-contact门槛；若仍falls，不替换现有0fall基线。34测试不参与调参。

[详细机器证据](amass_root_transfer_static_review_v1.json)包含实际checkpoint SHA、RMS数组/指纹、source与episode统计及未来接口；没有更改任何业务或配置。
