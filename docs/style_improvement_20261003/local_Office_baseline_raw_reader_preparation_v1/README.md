# Office3 / 10 s：独立原始物理读取器

准备的是本地 Office 资源诊断：当前生产 185 文件、默认 `62c4edc1…` controller、3 actors、300 controls、10 秒。Root 已完成并封存 GPU capture；本目录目前仅完成 source-only 登记，尚未读取该 capture 的 foot/contact/COM 数据。CPU 登记成功不能代替实际物理审计。

最终入口为 `audit_Office3.py`，最终登记为 `registration_v2.json`，具体命令在 `commands_CPU_Office3.json`。首次 `registration.json` 和 `audit_Office3_before_metadata_boolean_fix.py` 保留历史。第二次登记只修正了报告字段：`all_failed_actor_controls_retained=True` 位于报告顶层，原 `counts.falls` 保持实际整数计数。

## 来源和数学

三个 `frozen_*.py` 与已封存的原 native reader 逐字节相同。入口只在内存中转换八个登记的元数据表达式，逐项反向转换后要求完整原 AST 一致：archive 的绝对路径转相对索引；Office 静态 collider 数为 3642；reset actor 数 3；ray 行数 131（3 个 reset origins 加 128 固定样本）；原 authored PhysicsScene 清单为空；每条循环路线三个顶点。

原 foot force/contact/COM/fall/raycast/route 数学和全部失败帧保留。reshape 的 4 仍表示四个物理子步或四个 foot bodies。完整分母为 300 controls、900 actor controls、1200 shared 120 Hz substeps、3600 actor substeps、14400 foot observations，force 数组为 `(300,4,3,4,1,3)`。

`office_evidence.py` 在实际审计前检查当前 185 来源、默认模型完整 SHA/bytes、真实 Root GPU closure、原 exact argv、actor checkpoint/provenance、配置和 Office source 请求。生产 `deployment_entry.base_checkpoint` 实际只提供 path/SHA；真实模型 bytes 由 checkpoint record 和 EvaluationActor metadata 验证。此默认模型不适用 `09e` 的 reward-.5 terminal reader。本 CPU reader 不反序列化模型、不打开 SDK、不推进物理。

## 实际执行和解释范围

`source_only_registration_receipt_v2.json` 记录最终 source-only 进程已观察到 exit 0，以及实际全来源和反向 AST 检查。Root 的实际 CPU owner 还需执行登记命令、等待实际 child 关闭，绑定 GPU closure/root seal 与实际 raw report；未来 report 没有预占。

该短 capture 保留所有跌倒、负高度、静态和同行人接触；它们不是读取器完整性准入条件。实际保存的配置重力与 active USD gravity 是配置观察，不是 native engine gravity 测量。没有达到 200 秒导航稳定性验证，也没有匹配作者八个场景或证明论文效果。读取器登记发生在 Root 已封存短 capture 后；source/count 场景计划和原 raw 数学更早已确定，不把登记时间描述成 GPU 之前。
