# 分层专家实验：训练 payload 审计薄适配

## 准备状态

仅准备 CPU reader。未来 pilot/full checkpoint、GPU ledger、训练 result 尚未读取，当前没有真实训练 payload 准入或质量结论。Root 计划 fresh pilot4 和独立 fresh2000；不新增 GPU resume4to8。旧准备命令中的 resume 项保留为历史登记，不是本轮计划执行步骤。

原 reader `state_prior_training_audit_v1/audit_state_prior_training.py`（`2c60b638…`）和 helper（`cdb8a07e…`）曾实际通过 19f/934 训练审计。当前 `audit_stratified_training.py` 只替换 bundle/profile/输出/schema/935 身份，并接入七处明确 q hook；逆这些精确替换后全文 bytes/AST 与旧 reader 完全相同。helper 只适配 plan/source/archive 身份及 934→935 的来源推导，逆替换后全文 bytes 与旧 helper 相同。

模型权重、实际 Adam steps/moments、inference probe、RMS、57/12 action mask、replay ring、公共初始化、protected terrain、固定组曝光、全部物理状态、Python/NumPy/Torch/CUDA RNG 字节和原 curriculum 恢复逻辑仍由旧代码检查。CPU 不执行 CUDA RNG、不重演 PhysX、不宣称物理轨迹位级恢复。

## 唯一新增的真实 checkpoint 边界

`expert_sampling_checks.validate_expert_sampling` 使用实际 checkpoint 所对应的真实 train corpus 重建 `ExpertReferenceSampler`，核对：

- 完整 outer/inner saved schema、实际配置、registry/config/manifest/probability 指纹；基线 null 配置和缺失保存状态被拒绝。
- 真实计数必须为 `sampled_batches == update_count`、`sampled_windows == update_count×4096`。原 runner 每个 8192-transition update 恰好调用一次 4096 专家 batch。
- 初始 provenance 必须与实际重建的零计数描述完全一致；真实私有 CPU RNG 通过原 sampler loader 恢复后，全部 saved state 逐项相等。
- 构造及恢复不改变 global Torch 或 shared corpus RNG；closed result 处于同一 update 时，其最终采样描述必须与 saved state 完全一致。

原来的 next subset reset→expert→plan 恢复对比使用新 sampler 取代原 legacy expert call，并加入私有 sampler state；第二次对比前恢复 checkpoint 的私有 sampler state。原共享 corpus 与两组 curriculum RNG、reset payload、native plan 对比都保留。

私有 RNG 只检查真实保存/恢复及下一次逻辑取样一致，没有重放全部历史 draw 来推断训练过程。归档和 root 真实进程闭合仍是实际来源依据。q 的类别是本地根部运动代理，不是作者语义标签；未使用 p/q importance correction，不作论文严格风格采样复现的声明。

## 一次必要 prepared CPU 检查

`check_prepared_boundary_CPU.py` 实际闭合 exit 0（.875 s）：950 个 source/config/边界检查，验证完整 935 copied file set、939 archive members、旧 934 来源不变、唯一 q 配置改变，以及四个明确 synthetic metadata 反例。没有导入 Torch/Kit、读取未来模型、重复旧模型恢复测试或训练库大计算。

反例覆盖 null 配置、缺失状态、错误 registry 和错误 counters。它们只测试 metadata 边界；真实 RNG tensor 和完整 sampler loader 必须在 root 的实际 pilot checkpoint 审计时执行。原真实训练库映射/RNG/resume CPU 证据仍绑定 `state_prior_stratified_training_preparation_v1/actual_CPU_readiness.json`，不重复运行。

首次 prepared 检查因 docs helper 未把 previous-manifest 的 str 路径转换为 Path 而 exit 1，原 helper/source 已保留。实际 auditor 的原 record 接口本来支持 str。只修准备检查接口；synthetic fixture schema 改为读取 manifest-bound 原采样实现的 `_STATE_SCHEMA`（v2），不手写过时版本。

## 后续实际命令

`prepared_audit_commands.json` 给出 pilot4 与 terminal2000 的 CPU argv/env。两者只可在 root 实际训练进程 exit 0、结果/metrics/checkpoint 封存后启动，fresh output，保存所有失败审计。

准入条件继续包括 source `2e14550a…`、profile `bcbc8942…`、plan `6fdaf681…`；root owner 使用 `state_prior_stratified_actual_runs_v1` 和 `stratified_prior.actual_launch.v1`，pilot seal 为 `stratified_prior.actual_pilot_seal.v1`。完整预算 2000/16.384M 和 AMP weight2/state prior/2848 条训练数据保持基线。

本 reader 需要其他 agent 独立审查后冻结作为实际入口。训练质量仍由之后固定全 22 validation、同帧参考检查和原 fixture/native 物理门槛判断；prepared source 审查不等于真实 payload 准入，更不等于达到论文效果。
