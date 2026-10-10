# 40k EMA 固定 AMP 评价：最终训练计划接线 v2

只新增此目录；[v1](../formal_EMA40000_same_AMP_fixture_source_plan_v1/README.md) 保留原字节。当前是**源码准备**：40k 训练已经 root 实际启动，终态和物理评价均未完成；本轮未执行 adapter、preflight、NN/SDK/GPU 或读取任何未来 checkpoint/capture。

## 本次精确变化

- [参数方案](commands_and_source_plan.json) 绑定最终训练 plan `2beed1bd10777658eb1c6ec8e1a2aa3bbc7c8571646ddf8f5fa9f3fbec9ab945`、source registration `de9d6d279bb2ba2e3f25fb38b51288280532e55b81caa7fed8ad1121cf50b4ac`，all-temporal Maps/Interact 数据清单 v2、view staging v3。
- 候选路径明确为最终 plan 的 `official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt`，**bytes/SHA 仍 null**。已观察实际训练 owner50502/86eb41 为 running、`actual_exit_code=null`、`child_exit_observed=false`；不固定其 mutable running SHA，不据此产生终态准入。
- future GPU argv 仍使用原185 `simulate_paper_pedestrians.py`。相对已完成的 OLD，只有 DP path/output 两值改变；相对 v1 模板，只把 DP placeholder 换成最终 plan 的已知未来路径。AMP62、scene7971、v7 manifest、seed7、4people/200s/6000、guide20、sample1、replan15、alpha0、reference/physics/routes 参数全部沿用。
- 新增 [audit_fixture.py](audit_fixture.py)：只实现实际 terminal/owner/source 的 I/O 准入，原 f851 `Audit`、`audit_capture` 函数对象直接调用；唯一原模块 callback 替换为 `verify_sources`，使用 `finally` 恢复。不新增 launcher/control loop 或物理公式。

## 准入与真实 producer ABI

训练 owner 已由 root 修正为 canonical `command/env/child_exit_observed`；adapter 只接受这些键，不支持已废弃的别名。未来必须实际闭合0且 observed=True、readonly owner 与终态 audit identity 相同、command/env 与最终 plan exact。

终态报告严格接受冻结77025 auditor 的 `official_orca.all_temporal_actual_formal40000_CPU_checkpoint_audit.v1`，要求三个 strict-load/validate flags True、production load 与完整 EMA state 全等，实际训练40000/80000/16M预算和 snapshot identity 对应、独立 readonly inode/nlink1。auditor 自身没有 forward，仍不能证明物理效果；当前没有读取或生成这个未来报告。

未来 fixture owner 的窄接口沿已实际 stateD root-tool receipt：`formal_EMA40000.actual_fixture_root_owner.v1`，包含 exact command/env、`actual_exit_code`、`child_exit_observed`、真实 session/launch/completion、185 `production_source_before/after` 原始 records list、`production_source_epoch`、`actual_terminal_admission`、`checkpoint`（AMP62）、`planner`（实际 EMA）、`requested_profile` 和 `capture_files_sealed` records。**这是未来 root 发布的明确接口，尚未有实际 fixture owner**，不虚构 Popen/driver_completed 字段。plan 列出了完整所需字段。

entry 的 AMP `base_checkpoint` 仍仅 path/SHA；bytes 从 runtime actor/固定 controller record 校核。整个 AMP 历史 training provenance 与 OLD 封存结果完全相同，实际 EMA 通过 arguments 和真实 `evaluation_planner_change` 三字段单独绑定。header/stat 不充当 full SHA 证明。

## 后续最小执行顺序

1. root 实际完成40k训练并封存 output/owner，运行既有77025终态 CPU auditor，保存真实工具闭合与报告。
2. root 查看这份新 source 后，按 plan 的相同 entry argv 做实际 SDK-free resource preflight、注册/发布上述 fixture owner，再独占跑200s；不以离线或 MMD 数值筛掉此配对。
3. 真正 physical GPU closed0并封存所有捕获后，root 才执行 plan 中 `future_raw_CPU_argv_template`：填两个真实 audit/owner SHA，CUDA hidden。Adapter 在完整 metadata 准入前不会打开数组；实际阶段只对 checkpoint 做 opaque full hash，不构建 NN。
4. 原 raw reader 记录所有fall/static/peer/reset/ground/COM/routes；失败与idle不删除。零事件条件只报告，不能把 audit `status=passed` 解释成质量通过，不 promote 默认。

OLD current185配对已经完整完成：F0/static0/peer0、routes21，保留为对照不重跑。全6000控制步/24000 actor-controls/96000 actor-substeps与原 ground/normal-force/foot-origin proxy口径不变；peer最大力字段不是完整raw peer矩阵，足部原点速度proxy不是接触点滑动。

## 实际完成与限制

本轮唯一执行是 stdlib `compile`/AST 源码检查（工具 completion `d7b365`，exit0）；顶层 imports 仅 stdlib。未执行 adapter、未新增 synthetic suite、未读取运行中训练 payload或任何封存34输出。新 future fixture owner与真实 CPU terminal receipt仍待 root 实际生成；本轮不把它们标为 passed。代码/默认/生产185/旧方案和旧封存证据均未修改。
