# 混合课程真实物理探针：独立 CPU 保存恢复审核

## Step4

实际 GPU 执行已结束，独占快照 `step_4.pt` 为387,544,989 B，SHA `45f0dbfd9d71c69d1583e69a399209ffbcd34bdfcf1958ac62ce08e7a0c7cb93`。独立 CPU 审计 **170项通过**，见 [完整 JSON](mixed_curriculum_pilot4_independent_audit_v1.json)。审计本身没有启动 Kit/GPU。

- 4次更新、32,768次实际转换、128控制步；前后128个槽位各16,384次转换。
- AMASS125次实际源结束截断、0次跌倒；程序组1次跌倒、0次超时。原始 result 的1次跌倒及125次 time-limit 与课程累计状态逐项对应。初始化256次 reset 被明确排除，未冒充运行中重置。
- 145个模型、Adam、RMS、回放、物理及 RNG Tensor 全 finite；严格 CPU checkpoint loader 完整接受。
- 原 corpus/private AMASS/procedural 三套 RNG 可独立加载；top 与程序子 RNG 副本一致。AMASS151点真实相位、程序51点原插值、固定 actor 分组、episode、source offset 与刚重置必须重规划的门禁通过。
- self/task/value RMS count232,768，AMP249,152；Adam9步、D8步，与实际 metrics 的已接受更新相同。

135-file预 GPU 源快照逐成员 SHA/尺寸/路径完全验证。实际 CPU 审计器按新 manifest 的 `archive` 字段读取；其运行源码与两个工具依赖另封存为 [CPU auditor v2](mixed_curriculum_CPU_auditor_actual_source_v2.json)。原 GPU source manifest、tar及其内部工具快照不变；CPU admission 使用的业务源码与135-file冻结源一致。

## 实际恢复 Step4→8

恢复后 `step_8.pt` 为652,181,213 B，SHA `8613884fbe962ead945a8a544ba50b1adb9c9a811b2b64f77063cfc4d96557d6`。[两段完整审核](mixed_curriculum_pilot4_resume8_independent_audit_v1.json) **198项通过**。

累计8次更新、65,536次转换、256控制步；两组各32,768次。新增段为32,768转换、0次跌倒、159次真实 AMASS 源结束截断、0次 cap。累计程序跌倒仍1次，真实源结束累计284次，未误计为恢复段的新增数量。两组分配、TrainingConfig、runner合同及有效计划保持相同。

Adam从9→39、D8→16；self/task/value RMS232,768→265,536，AMP249,152→298,304。严格模型、完整回放及145个 Tensor 全 finite；源相位与原始 reset offsets 恢复门禁通过。

额外 [15项 RNG 续流检查](mixed_curriculum_pilot4_resume8_rng_progression_v2.json) 直接从 step4 的已存 CPU RNG 状态推进已记录的调用数量：AMASS159个整数/均匀数对、原 corpus16,384个专家采样对、程序1152个计划共173,952个均匀数。三个最终 generator state 均与真实 step8 **逐字节相等**；从 fresh seed 仅推进本段的状态全部不相等。这为实际续流提供证据，未声称重建 reset 批次、动作或 PhysX 轨迹。runner合同额外键被明确拒绝，CLI仅显式 resume 参数变化。

额外检查首次运行曾因审计脚本误用 `source_motion_ids` 字段失败；修正为实际 `motion_ids` 后生成v2报告，失败v1日志保留。这不是课程运行失败，也没有改动已冻结的业务或主审计器。

以上验证实际探针完整执行、工件完整性和逻辑恢复资格。尚未验证 PhysX solver/contact cache 字节恢复，也不据此认定步态、风格、导航质量通过。未读取封存34 test输出。
