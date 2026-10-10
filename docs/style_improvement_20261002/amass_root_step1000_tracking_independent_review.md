# AMASS根课程step1000：完整22条tracking独立协议审计

## 实际结果

新固定controller完成全部22条validation locomotion序列、11,982个poststep metric帧；没有clip cap或序列内reset。独立CPU审计从每份原始NPZ重算现有primary指标，与报告逐项一致：

| 同一22条协议 | 旧negative baseline | AMASS root step1000 |
|---|---:|---:|
| 完整序列成功 | 11/22 (50.00%) | 17/22 (77.27%) |
| 有fall的序列 | 0 | 0 |
| frame-weighted mean 24-body位置误差(m) | 0.3007080082 | 0.2292981841 |
| frame-weighted mean 24-body旋转误差(rad) | 0.8614544462 | 0.8318819187 |

成功仍采用冻结规则：任何测量帧24个world body-origin平均位置误差超过0.5m即整条失败。初始reset帧不计入主指标，全部后续帧保留。该协议是true future XY oracle tracking，不是Diffusion导航场景；作者精确clip/metric协议未公开，不能将此表当作严格论文复现结果。

## 独立证据

- [完整机器报告](amass_root_step1000_tracking_independent_protocol_audit.json)、[日志](amass_root_step1000_tracking_independent_protocol_audit.log)、[审计脚本](audit_amass_root_tracking_protocol.py)。
- [原始clock/action补充检查](amass_root_step1000_tracking_clock_sidecar.json)：新旧time-limit flags均为0，恰好22个初始reset，12个冻结applied action坐标为零且全部normalized applied actions满足原有clamp范围。
- [实际新tracking结果](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/motion_tracking_amass_root_curriculum_full_validation_v1/result.json)绑定step1000 SHA `dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e`。

每份capture SHA与结果声明相同，checkpoint SHA与preflight/CLI相同。实际controller训练manifest及validation manifest哈希有效，训练parent sources与validation parent sources不交叉；完整validation的22个source IDs全部一致，没有glob补缺、替换sequence或挑选成功片段。

parent gate覆盖本轮显式AMASS训练数据。公开PACER初始化权重的完整预训练motion名单未知，其潜在heldout重叠仍未确认。

## 配对输入与时间

新旧22条逐位相等的输入包括：全部reference root/body位置、XYZW旋转、root/body线速度和角速度、69D DOF位置及child_joint速度；十点future XY命令、reference frame/time、query phase、clip-tail hold flags、metric/native-reset/action-valid标记；初始实际10种完整物理状态字段。

每条从source frame9初始化，随后frame10…真实末帧按30Hz连续测量，一次初始reset，没有序列内fall/time-limit reset。11,982个metric帧与22个初始reset帧均清晰区分。每条独立pose-style导数的合法间隔总计11,960；此处只核对边界数量，没有计算MMD或风格指标。

primary errors使用原冻结Euclidean及SO(3) geodesic函数，从raw positions/quaternions重算，存储的body/per-frame errors、每条成功/fall及whole frame-weighted聚合全部在1e−12绝对容差内相等。没有root/yaw/phase最优对齐，没有替换主指标或删除fall帧。

benchmark没有保存SDK物理counter，因此本审计验证reference/source30Hz时间及执行协议，不凭空宣称原生counter数组已逐项检查。独立GPU执行真实性由原始结果及源码契约支持。

## 解释边界

输入/初始状态逐位配对不代表后续物理轨迹相同。训练课程同时改变了XY命令分布、合法reset phase及源末帧截断，并使用更大训练地面；即使core TrainingConfig保持baseline，效果不能归因于一个未隔离变量。

本报告仅确认协议公平性与冻结primary指标。heading/limb/速度bin误差分解由CrowdSim代理负责，官方冻结风格/MMD由资源代理负责。step500完整验证与22条风格结果用于后续明确候选选择；封存34条test捕获尚未读取。
