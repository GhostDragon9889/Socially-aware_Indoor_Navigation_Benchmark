# 关节功率修正：CPU 独立复核

**状态：通过，33 项回归，Ruff PASS。** JUnit：[joint_power_tests.xml](joint_power_tests.xml)。测试文件：[test_paper_joint_power.py](/project/EXP/SangforAndSaut/ProtoMotions/tests/test_paper_joint_power.py)。

## 验证对象与结果

从真实 `amp_world.py` 原样提取 `PaperPowerSimulator` 类 AST，以记录顺序的 CPU 响应源执行；同时执行基类真实 `get_dof_forces()` 和真实 `RobotState` 排列转换，没有导入 Isaac SDK 或启动 Kit／GPU。

4×2×69 输入的独立解析期望为 **[-6037.5, 9660] W**；子步顺序严格为 apply → write → step → 可选末步render → update → measuredForce → 对应postVelocity。四步mean恰好等于期望，保留功率单位与 `−0.0005` 系数；不是最后一子步，也不是四项未归一的sum。第二控制步不会累加旧控制步。

使用非平凡69DOF排列验证native→common、env子集和相同排列下功率不变；getter结果clone可抵抗SDK刷新同一内存。legacy模式两方法原样委托，不读新测量；getter异常不会自动切回proxy。

## 恢复门禁／评估语义

`power_stage` 拒绝非dict、额外未知字段、未知source、被修改合同和非boolean选项。旧无标注包被解释为原legacy；训练更换测量必须显式 `--new-power-stage`，且只能legacy→projected，重复、逆向或evaluation宣称trainingstage均拒绝。

真实runner的provenance分支AST验证：新训练stage写sourcehash、sourceupdate1000与奖励变更声明；evaluation允许显式运行当前测量，但不重写旧训练provenance，并单独报告变化。业务修正由root完成，此审查仅新增测试／文档。

## 实证边界

这些CPU测试证明实现调用链、索引、平均与门禁；**CPU响应源自身不证明真实PhysX读数、符号、实际碰撞行为或吞吐**。投影是多种关节内部贡献的generalized effort，不是独立motor torque；substep平均力配步末速度属于已声明的离散quadrature。保留原1000-update成果不变，新rewardstage单独记录。

原始语义问题记录见 [torque_semantics_audit.md](torque_semantics_audit.md)；该文档最初源码hash是修正前快照。修正后精确代码hash、完整检查清单见 [joint_power_patch_verification.json](joint_power_patch_verification.json)。

## 后续独立真实 PhysX 门禁

root 随后完成了 [projected_power_physics_probe.json](projected_power_physics_probe.json)：64 个环境、64 个控制步、4096 次物理交互、2 次真实 PPO／AMP 更新，状态通过、跌倒／reset 均为 0、最低 pelvis 高度 0.6731m。报告明确调用真实 `get_dof_projected_joint_forces`，没有 proxy fallback，物理 120Hz／控制 30Hz、四个等时子步求平均。

此后续结果由 root 实际运行，区别于本文件的 CPU 响应源测试；它确认 SDK 读取与训练兼容性，不宣称策略已收敛、导航稳定或测到了分离的 motor torque。后续长训练应单独引用其最终 checkpoint、指标和 provenance，不使用这个两次更新门禁作为最终训练结论。
