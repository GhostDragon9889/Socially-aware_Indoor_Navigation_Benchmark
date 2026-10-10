# AMASS root1000：实际四行人200秒原生测量独立审查

## 执行与效果分开

**原生测量/来源完整性502项检查全部通过；本轮导航效果未通过。** 实际GPU执行完整6000 controls、四行人200秒，13次跌倒、0 time-limit resets；路线顶点[2,2,1,0]合计5，最低Pelvis高度0.3905838m，elapsed862.265983s。`result.status=passed`仅表示执行完整，不是行人稳定/风格/导航效果验收。

固定AMP `dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e`、实际Diff5000 `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`全部重算SHA与127路径prelaunch快照匹配，没有把训练时继承的Diffusion provenance误认成部署planner。

## 完整capture及失败边界

保留全部24,000 actor-control frames、24,000个实际连续120Hz提交步、96,000 actor-substeps及384,000足部子步观察。每control恰好四子步，无partial、补零或丢步。ground的episode/reset/terminated/truncated和末子步body position/Pelvis quaternion与独立physical_states逐位一致。

所有13次终止及其下一个episode首个poststep捕获帧都有精确对应；17个reset markers包括最初四人初始化。下表使用实际poststep capture index及明示dt时间，不声称SDK native engine time。全部事件上下文及reset清单在[事件/source sidecar](amass_root_step1000_navigation_event_source_binding_v2.json)保留，失败帧不从任何inclusive结果删去。

| actor | episode | 终止capture index | 完成control时刻(s) | 提交physics step | Pelvis z(m) | root up z | 同control static contact | 同control peer actor |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 3 | 1 | 592 | 19.766667 | 2372 | 0.3906 | 0.2628 | false | 无 |
| 0 | 1 | 652 | 21.766667 | 2612 | 0.4397 | 0.7026 | true | 无 |
| 2 | 1 | 935 | 31.200000 | 3744 | 0.4203 | 0.7608 | false | 无 |
| 3 | 2 | 1107 | 36.933333 | 4432 | 0.4428 | 0.7796 | false | 无 |
| 3 | 3 | 1412 | 47.100000 | 5652 | 0.3910 | 0.2628 | false | 无 |
| 1 | 1 | 1630 | 54.366667 | 6524 | 0.4092 | 0.6536 | true | 无 |
| 3 | 4 | 1797 | 59.933333 | 7192 | 0.4492 | 0.3514 | false | 无 |
| 2 | 2 | 2355 | 78.533333 | 9424 | 0.3940 | 0.4648 | false | 无 |
| 1 | 2 | 2544 | 84.833333 | 10180 | 0.4049 | 0.2535 | false | 无 |
| 3 | 5 | 2768 | 92.300000 | 11076 | 0.4492 | 0.8026 | false | 无 |
| 2 | 3 | 4053 | 135.133333 | 16216 | 0.4979 | -0.0165 | true | 无 |
| 0 | 2 | 4304 | 143.500000 | 17220 | 0.4471 | 0.7666 | false | 无 |
| 3 | 6 | 5880 | 196.033333 | 23524 | 0.4051 | 0.0680 | false | 无 |

表中同帧接触仅作为已记录上下文，不能据此认定跌倒因果。终止状态先capture再reset，下一行episode的第一个状态已经完成一次新episode控制步；没有把该poststep状态冒称为reset写入前的原始姿势。

## 原生force、COM及proxy

- raw floor矩阵唯一filter `/World/ground/terrain/mesh`，16个foot sensor与common/native身体映射精确对应；floor与障碍/peer通道独立。所有raw normal-force sums、严格>1N支持mask、左右载荷及per-control汇总从四子步重算一致。
- 全部body mass正值，总质量63.31394896kg。独立SO(3) link-local COM→world重构最大分量误差9.97854389e-07m；质量加权trueCOM一致。AMASS reference trueCOM未知，保留null，没有虚构reference COM/ground truth对照。
- 双支持比例57.834375%、无足支持0.726042%，分母为全部96,000 actor-substeps、包含跌倒及reset边界。它们是明示阈值的传感统计，不是论文未公开验收指标。
- 原static contact：2873/24,000 actor-frames；peer touching actor-frames 120，unordered pair-frame union 60。分别最大力3887.3357N和850.4664N。原四contact字段与ground投影一致；没有原生peer raw force矩阵，因此不虚构独立raw peer-force重建。
- contact条件foot **body-origin切向运动代理**：mean0.086304、median0.036871、P950.302989m/s。滚动/旋转也使原点运动，此项不是true contact-point slip；没有摩擦力或contact patch速度，不能与旧AMASS toe运动代理硬当同一滑动指标。

历史`native_capture_probe_passed=false`及neutral-stance `native_physical_probe_passed=false`保持原样；新独立`actual_native_capture_validated=true`仅说明这一真实learned-policy录制通过检查，不追溯改历史QA结论。

## 来源与匹配部署

127文件archive SHA `c5c8c4d26ffe6220405742ec842f683a708c84a8afcbd41489bf5e9ade5d7dd5`，manifest SHA `c9be09f46086c5909b864806bf9242d9f422a64122f1da88888443b760d2b752`。所有path、成员类型、bytes/SHA严格校验；124父清单仅CPU watcher身份变动并加三份CPU provenance docs。GPU core/profile与独立CPU冻结审查一致。整个ground控制For和两CLI项/四保护条件AST保持旧定义。

真实CLI与prelaunch准备完全相同；与旧negative guide20部署只有config、output、resume路径变化，以及旧metadata不存在的`near_horizon_waypoint_mass`显式写0（功能disabled）。所有原TrainingConfig字段精确相等；新metadata多`expert_sampling=None`，明确对应已知legacy uniform，JSON对象并非完全相同。新controller训练课程/合法reset phase/clip-end规则变化，因此不是单一loss参数因果消融。

本次只读取结束后完整产物，不以current tree覆盖历史源身份，不修改业务/原数据。官方冻结风格、停留与路线对比由资源代理负责；正式34 test已由root运行，但本审查未读取34输出或用于训练设计。

## 证据

- [502项机器报告](amass_root_step1000_navigation_ground_actual_verification_v1.json)、[生成的measurement说明](amass_root_step1000_navigation_ground_actual_verification_v1.md)、[日志](amass_root_step1000_navigation_ground_actual_verification_v1.log)。
- [全事件与部署/source绑定](amass_root_step1000_navigation_event_source_binding_v2.json)、[日志](amass_root_step1000_navigation_event_source_binding_v2.log)、[CPU脚本](audit_amass_root_navigation_events.py)。初次v1侧记检查因旧TrainingConfig缺`expert_sampling`metadata而拒绝，日志保留；显式记录这个已知新增disabled字段后才通过，未改native审计或业务。
- [127来源wrapper](audit_amass_root_navigation_ground.py)，复用原f851e7…measurement auditor全部检查；[准备审查](amass_root_step1000_navigation_ground_source_preparation_v1.md)。
