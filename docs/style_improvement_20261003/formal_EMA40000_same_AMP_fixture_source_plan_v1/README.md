# 40k 全时轴 EMA：固定 AMP 的 200 s 物理评价源码方案

状态：**仅源码准备，等待真实 40k 终态 CPU 准入及 root 串行授权**。未执行 preflight、加载 NN/权重、启动 SDK/GPU、打开未来捕获或保留任何假成功记录。只有本目录的 [参数与源码方案](commands_and_source_plan.json) 和本文；没有新 launcher、控制循环或生产代码修改。

## 可直接复用的运行路径

使用现有 `scripts/simulate_paper_pedestrians.py` → `application/deployment.py` → `scripts/train_paper_amp.py`。现有 `EvaluationActor` 仍读取 AMP62，alpha=0、无 donor；默认 AMP 文件原 0664 模式保留。所有 scene/seed/routes/目标时钟/reference/PD/action/物理参数沿用已完成的 current185 配对 OLD 基准。

GPU 子进程 argv 模板在 JSON 中：相较原 OLD command 仅索引 7 的 **DP checkpoint path** 和索引 22 的 **output** 改变。`ACTUAL_ROOT_ADMITTED_READONLY_STEP40000_PATH` 必须等终态 CPU 准入后替换，当前模板不可执行。候选 `path/bytes/SHA`、终态 audit、真实 owner 和 root 最终计划均为 null；现有 draft 路径只是参考，不能作为真实候选身份。

主要固定参数：4 人、200 s/6000 控制步、seed7、AMP62、OLD profile/v7 AMASS manifest、固定 scene7971、30 Hz 控制/120 Hz 物理、15 步同步重规划、单候选、trace_clean guide20、alpha0、agent5/radius0.6、scene10/radius0.8、causal_accel reference（1.5 m/s、2 m/s²、overspeed brake）、接触与地面支持阈值1 N。完整 forwarded argv 与解析参数都保留在 JSON。

OLD matched185 基准已实际完成：6000 控制步/200 s、0 fall/0 static actor-contact frames/0 peer pair-contact frames、routes `[2,8,6,5]` 共21。GPU/CPU owner 均 closed0、child exit observed、driver completed；本次仅读取既有封存 JSON，**不重跑 OLD**。

## EMA 与现有 Planner 的 ABI

- formal `training_state.checkpoint_payload` 保持 `schema=crowdsim.paper_five_state_diffusion.v1`、`config` 和 `step`；`model` 是完整 EMA state（含 buffers），`training_raw_model` 和 raw Adam 单独保存。
- current185 `PaperDiffusionPlanner.__init__` 使用 saved config 构造同五状态架构，检查每个 `model` tensor finite，`load_state_dict(..., strict=True)`，读取 `step`，然后 `.eval()`。部署不会选择 `training_raw_model`，额外训练状态字段不影响该选择。
- 这是源码 ABI 兼容，**不是实际未来权重兼容性证明**。已有 `audit_terminal.py` 的真实 CPU gate 必须严格加载 production Planner，并逐 tensor 比较部署 state 与 terminal `model` 全等；另按原恢复接口严格检查 raw/EMA/Adam/recipe/corpus/RNG。它不执行 forward，也不证明物理效果。
- 部署换 DP 时 entry 自动转发 `--allow-evaluation-planner-change`。runner 的真实 producer 保留 AMP 的旧完整 training provenance，另记录 `evaluation_planner_change={training_sha256:OLD63e,evaluation_sha256:actualNEW,checkpoint:actualNEWpath}`。不能把历史 `result.provenance.diffusion_*` 改标成当前 EMA。

源码位置：[部署接缝](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/application/deployment.py:103)、[现有加载器](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/diffusion/paper.py:394)、[formal payload](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/orca_official_mixed_training_staging_v2/training_state.py:147)、[实际终态 gate 源码](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/orca_official_all_temporal_formal_preparation_v1/audit_terminal.py:110)。

## 原 reader 的必要薄 I/O 替换

旧 `default_newDP_physical_preparation_v1/physical_evidence.py.source_binding` 硬要求 localMaps 的935源、5000步 owner/schema/seal，不能直接接此 formal 候选。所需改动仅为新的 actual formal source/training owner/terminal CPU audit/DP 记录与 output 身份准入；禁止借用旧 b563 seal 或旧935血缘。生产部署仍是185。

可保留 v2 `validate_runtime_planner` 的历史/当前 DP 分离逻辑：完整 AMP 历史 provenance exact，实际 arguments 指向 admitted EMA，`evaluation_planner_change` 三字段 exact，资源 header/stat 指向实际 DP；完整 SHA 由 root sealed artifact 与 owner 前后身份提供，header 不是 hash 证明。

未来 metadata 准入至少需要：

1. root 最终 formal plan/source registration、真实闭合训练 owner、封存完整 training result；40000 optimizer updates/80000 microsteps/16000000 windows、micro200×accum2、EMA inference。
2. 真正 closed CPU terminal receipt 与 `official_orca.all_temporal_actual_formal40000_CPU_checkpoint_audit.v1` passed；三个 strict-load/validate flags true、checkpoint path/bytes/SHA exact、独立 readonly inode/nlink1，production load 与 EMA 全等。
3. 实际 fixture preflight/report、root GPU command/env、真实 exit0/observed closure、185 source 前后身份与全部 readonly capture；不根据不同 owner 类型猜 `Popen` 字段。AMP entry `base_checkpoint` 真实只有 path/SHA，bytes 另由 sealed record 和 runtime actor 描述核对。
4. 6000/200 s 完整 captured masks/counters，保留所有 fall/contact/idle/reset；status passed 仅说明来源/完整性，不等于零事件或质量通过。

原 raw `Audit` 和 `audit_capture` 函数对象直接复用 [f851 reader](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/audit_ground_support_capture.py:142)。在完整 metadata 准入后只替 `verify_sources` callback，`finally` 恢复；不改 forces/COM/contact/threshold/clock/mask/分母数学。原 90D/45D reader 如需后续描述仍单独使用，不作为启动或样本筛选条件。

## 全 200 s 输出范围与解释

保留24000 actor-control frames、24000 physics substeps、96000 actor-substeps 和384000 foot-substep observations。原函数检查并报告 fall/reset/route、static actor-contact frames、peer actor 与无序 pair-contact frames、四子步 normal-vector foot support、double/no-foot support、contact-conditioned foot-origin tangent-speed proxy、mass-weighted COM 与 pelvis height；不把 foot-origin proxy 改称 contact-patch slip，原 peer max-force字段也不宣称完整原始 peer force matrix。

不新增质量/MMD launch filter，不因离线排名或物理失败删除数据，不据此 promote default。未来 fixture 的零 fall/static/peer 条件只用于是否继续 native 的后续决定；本 source-plan 不启动 native。

## 当前待决项

读取时现有 `commands_and_audit_plan.json` 仍引用 all-temporal 数据 v1/output v1，而训练源码已指向 view v3、默认数据 v2。**必须使用 root 最终选中的 plan 和实际终态记录**，不把 draft 路径/旧 filtered pilot 当本轮准入。JSON 保存该 draft 的身份和此差异，未来 actual candidate 字段全部保持 null。root 决定后只需替实际路径并注册来源；本轮没有读取未来文件或创建多层适配框架。
