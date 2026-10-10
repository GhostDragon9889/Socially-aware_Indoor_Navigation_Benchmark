# AMASS根课程step500：完整22条独立协议审计

实际固定step500 SHA `8db53ffad6b60782a7e73dd924de675f0caaa941d951efb6806fad22cd092314` 完成22条validation、11,982个poststep metric帧。所有原始capture SHA、完整source IDs、30Hz phase、初始状态、reference和future XY输入通过与旧negative baseline的逐位配对；现有冻结primary指标从raw重算一致，没有改metric或删fall帧。

| 相同完整22条协议 | negative baseline | AMASS root step500 | AMASS root step1000 |
|---|---:|---:|---:|
| 成功序列 | 11/22 | 12/22 | 17/22 |
| 有跌倒的序列 | 0 | 1 | 0 |
| 位置误差(m) | 0.3007080082 | 0.3578618902 | 0.2292981841 |
| 旋转误差(rad) | 0.8614544462 | 0.8757440268 | 0.8318819187 |

step500和step1000属于同一次fresh训练不同预算，不能当成等预算方法对照。step500的12/22结果未超过step1000的17/22；候选选择还应结合资源代理的唯一冻结风格评估。本报告没有计算风格/MMD或误差分解。

- [完整独立机器报告](amass_root_step500_tracking_independent_protocol_audit.json)、[执行日志](amass_root_step500_tracking_independent_protocol_audit.log)。
- [raw clock/action补充检查](amass_root_step500_tracking_clock_sidecar.json)：22个初始reset、0个time-limit flags，12个冻结action坐标恰好为零，全部applied action在既有normalized clamp范围内。
- [原始实际结果](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/motion_tracking_amass_root_curriculum_step500_full_validation_v1/result.json)。

每条从source frame9初始化，所有后续帧保留；有fall的sequence不进行中途reset。初始reset帧不计primary。SDK物理counter没有在此benchmark保存，本报告仅验证公开capture的reference时间及执行协议。此项为true-future-XY oracle tracking，不是Diffusion导航或严格作者评估。公开PACER初始化预训练数据的潜在heldout重叠未知。封存34条test结果未读取。
