# AMASS root sealed500→750：完整22条独立raw协议验收

## 验证结果

**实际完整capture CPU审计通过。** 新续训750控制器SHA `1e87b82b643a1d94bf2117de5a57a27ebb1e0e9811a4cabf66ab1588fcaf9592` 完成全部22条validation序列、11,982个poststep metric帧；16条成功、2条跌倒。全部raw position/SO(3) primary、每条summary及frame-weighted整体聚合与原始result在1e−12绝对容差内相同，没有删fall帧、重置失败片段或优化yaw/time对齐。

| 相同完整22条协议 | 完整序列成功 | 有跌倒的序列 | 位置误差(m) | 旋转误差(rad) |
|---|---:|---:|---:|---:|
| negative baseline | 11/22 | 0 | 0.3007080082 | 0.8614544462 |
| AMASS root fresh500 | 12/22 | 1 | 0.3578618902 | 0.8757440268 |
| AMASS root sealed500→750续训 | 16/22 | 2 | 0.4012078918 | 0.8053352740 |
| AMASS root fresh1000 | 17/22 | 0 | 0.2292981841 | 0.8318819187 |

750失效IDs为1、14、20、245、257、261；跌倒IDs245和261。这些跌倒后的姿态保留在inclusive指标中。其16/22成功高于fresh500，低于fresh1000；包含跌倒的750平均位置误差较高，平均旋转误差较低。不同预算与新物理恢复会改变优化轨迹，此表不提供单因归因或自动候选推荐；还须结合资源代理唯一冻结风格评估。

## 配对与来源

与negative baseline全22条逐位相等的输入：全部10类reference root/body/DOF状态、XYZW旋转、child_joint速度，future10XY命令，source frame/time及query phase，tail-held/metric/reset/action-valid标记；每条初始实际10类完整状态也逐位相同。schema与所有capture SHA有效，全部source IDs及合法完整window相同，所有frame9初始化后frame10…真实末帧按30Hz连续。

- 22个初始reset帧明确不计primary，序列内没有reset，0个time-limit flags；每条独立导数合法间隔总数11,960。
- normalized applied action范围及12个冻结坐标零值通过；fall frames不因action/metric处理被删。
- 固定训练manifest及validation manifest SHA有效，显式AMASS train parent与validation parent无交叉。公开PACER初始化预训练名单未知，潜在跨预训练heldout重叠未确认。
- 新旧选定source仅world源码身份不同；primary metric源码未改。此world差异是已经冻结的可选AMASS训练接线，evaluation参数一致。

[机器报告](amass_root_from500_to750_tracking_independent_protocol_audit.json)、[执行日志](amass_root_from500_to750_tracking_independent_protocol_audit.log)、[raw action/clock sidecar](amass_root_from500_to750_tracking_clock_sidecar.json)，以及[原始实际结果](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/motion_tracking_amass_root_from500_to750_full_validation_v1/result.json)。

## 解释范围

本协议是true-future-XY oracle完整tracking，没有Diffusion闭环导航或ground-support采集。作者精确clip清单/指标细节未公开，本地固定22条不等同作者完整评估。SDK物理counter未保存在tracking capture，因此不虚构原生counter数组核对。reference/初态配对不代表后续物理状态相同。

750来自[sealed500→750实际续训](amass_root_from500_to750_independent_terminal_review.md)，不是原fresh训练在750时的快照；PhysX caches不序列化，续训没有逐位物理重放承诺。本报告不重复风格/MMD或误差分解。封存34条test输出未读取。
