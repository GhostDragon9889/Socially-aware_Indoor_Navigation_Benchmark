# Warehouse 四行人真实 200 秒：独立 CPU 审计

**983 项既有原生审计门禁全部通过。** 本次读取完整实际 GPU 产物，无新 SDK/GPU 启动，未修改生产源码或原 result/NPZ。原审计的 source/raw/COM/force/live-floor/route 公式及门禁使用已封存字节复制，不更换指标。

## 最终原始结果

| 项目 | 独立核对结果 |
|---|---:|
| 控制步 / 人数 / 模拟时长 | 6,000 /4 /200 秒 |
| actor-controls / shared substeps / actor-substeps | 24,000 /24,000 /96,000 |
| 四足子步观测 | 384,000 |
| 提交物理计数 | 0 →24,000，无缺步/额外步 |
| 跌倒 /时间上限重置 | 0 /0 |
| 到达顶点，各人 | **[14,14,14,15]**，合计 **57** |
| 同一 episode 内完整四顶点循环，各人 | [3,3,3,3]，合计 **12** |
| 静态接触 actor-controls，力阈值 >1N | 0 /24,000 |
| 行人接触唯一 pair-controls，力阈值 >1N | 0 /36,000 |
| 保留的 static/peer 最大力 | 0 /0 N |
| 最低 pelvis 相对已声明地板高度 | 0.825075388 m |
| 真 COM 重构最大绝对误差 | 9.53674316×10⁻⁷ m |
| 质量加权真 COM 重构最大误差 | 4.76837158×10⁻⁶ m |
| GPU 执行墙钟耗时 | 1,183.543599336 秒 |

最后一条进度日志为 `[14,14,14,14]`；**最终 raw/result 是 [14,14,14,15]**。所有 57 次到达的实际顶点身份、物理 XY 距离 ≤0.75m 与 episode/route-index 顺序逐项复核通过。没有将跨 episode 的事件拼成完整循环；此运行原始 terminated/truncated/reset 事件均为零。

每人 episode 内实际行走距离分别为 226.7431、225.4774、225.7845、227.5663 m。到执行 reference target 的平均 XY 误差约 0.1014–0.1052m；该量不是全局路线目标到达误差，也不是作者未知汇总口径。

## 真实原生通道

全部 6,000 postcontrol 帧和四原生子步 retained；ground 与 physical 的 episode/reset/terminated/truncated 精确相同，末子步 24 common body origins 与 physical body positions 逐位相同。XYZW 单位旋转、正质量、local COM offset、世界真 COM、质量加权 COM 都由保存通道独立重算。>1N foot-to-selected-floor 力阈值、四语义足部顺序、左右支撑、双支撑与无支撑比例均按原公式复核。各人双支撑比例约15.70–15.77%，无支撑比例为 [0.01667%,0%,0.0125%,0.00417%]；这些是力测量统计，不是 AMASS 接触力 ground truth 或姿态风格验收。

原生 live observation 保留同一 singleton/current stage 身份、源 USD 在 session 中恰一次、一个活动 PhysicsScene、原 PhysicsScene 被 session 禁用、745 exact static colliders、一个实际有限 reference 地板。原 Plane 和11 decals 被禁用；实际地板 geometry/transform/material 与已绑定 recipe 相符。53 份来源层的完整 SHA 与 preflight 相同。

四个实际 FP32 reset 起点及128个确定性 admitted-free 样本的 **132 条 all-hit 射线**全部保留：每条恰一个 selected-floor support，geometry 有限，exact paths 无 foreign/coplanar floor，submitted counter 始终0→0，发生在首个 physics step/reset 之前。它们不是200秒结束时重新查询，也不是重新进行全地图 floor cooking 验证。

## 来源与闭包

- source168/V4 manifest SHA：`07a34630e052a356aefa3755781bd4c5a7d1406d6a6dd83a0f0f3ec0de80b54f`。
- source168 归档 SHA：`310b87c79851ac44ffe02efc282cb412bb31f27236340242e39f21724cadc9f0`。全部168个 regular member 的大小、内容 SHA，以及审计登记/实际读取时 current168，精确一致；差异列表为空。
- actual AMP checkpoint SHA：`62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b`；actual consumed Diffusion checkpoint SHA：`63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`。两份文件实物完整 SHA 独立核对，未用 inherited training hash 代替实际 Diffusion。
- 推理 runtime 记录只保留 task_encoder/actor，critic/D/log_std 仅加载时验证，未构造训练 optimizer/replay。实际执行参数与 prelaunch deployment_entry 精确相同。
- Warehouse 地图沿用真实已绑定历史 v5 resource 和 historical producer snapshot，未将 v8 CPU proposal 标成这次运行的来源。
- result SHA：`5ced8ef73305421a4d3ae1ee0805a29e3981ed44223a46bc28ebc7db9ce41985`。
- physical states SHA：`15dbf98cb483dfefdff63b6ce781af88c04090028927b16330986d4d20f3e616`。
- ground support SHA：`29f72121af965902a75e6f658978a84cdab3342016a67b53fe55c12a95bbed97`。
- independent audit SHA：`882f17cc3c45c5b216d17b0e7bfaf90a9bd41c49bb9e4960b509befdb841b389`。

审计完成并向 root 报告后，root 才应用 separately approved v8 patch。新 current bytes 不追溯替换 source168；本目录封存登记/实际读取时的168份 byte records 和历史归档。此后的 current-exact 重新执行应拒绝合法漂移，不能把漂移回填成原 GPU 用了新源码。

## 可得结论与边界

本次证明这个固定模型/配置/种子在真实 Warehouse 四人200秒完整观察中零跌倒、零已测 static/peer 接触，并完成12个路线循环。`result.status=passed` 仍只代表执行完成，导航观察与原生通道审核另列。不是八场景复现、所有种子稳定、论文 style/机器人导航指标全部通过，也没有建立 Warp/PhysX 普遍逐位确定性保证。

ground report 的历史 `native_capture_probe_passed=false` 保留；本次实际 learned-controller capture 独立有效由审计报告单独说明。App/Context 构造次数、独立 native engine timestamp 仍未测量，保持未知；submitted step counter 不改称 engine time。足部 body-origin tangential speed 仍是代理，不是原生接触点滑移；AMASS reference 真 COM/摩擦未知。没有新拟合 MMD 或更换 frozen style reference。

证据：[完整审计](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_scene_200s_actual_review_v1/actual_audit.json)、[预登记](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_scene_200s_actual_review_v1/registration.json)、[原函数冻结](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_scene_200s_actual_review_v1/frozen_reader_inputs.json)。
