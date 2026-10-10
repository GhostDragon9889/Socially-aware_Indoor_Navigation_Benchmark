# 地形统计保护：完整候选与隔离训练来源

**完整候选、CPU准入已完成；root真实4更新pilot已独立启动，尚未在此报告中审核终态或宣称质量改善。** 没有由本CPU任务启动GPU/Kit，未修改旧模型、v7bank或旧sealed训练目录。

## 当前代码证据

- [shared module/profile/unit冻结](owner_source_freeze.json)：`terrain_statistics.py` SHA `7c07e3a54617d40b72475bc40bf0b728516b7d4166fb99795c8a8bb69b0f78f3`；33 CPU专项通过，Ruff通过。
- [训练接线及默认兼容检查](dependencies_integration_review.md)：settings/trainer/frontend候选，以及真实public warmstart、完整Adam/RNG恢复、旧默认更新等门禁。完整生产候选66 CPU、0skip通过；默认before逐位/AST及cold检查13通过。生产合入由root单独记录，此处候选不冒充运行源码。
- [实际隔离CPU准入](isolated_CPU_readiness_v1.json)：guard核验931普通文件与6个资源全SHA，实际60个CPU业务模块只来自copy；冷预检10个业务模块、无Torch冷导入/SDK启动。隔离52测试、0skip、0error，包括实际public暖启动、下次optimizer恢复和严格推理输出；不是实际物理恢复逐位证明。

## 同旧训练的明确边界

[builder](build_isolated_training.py) 先逐SHA读取并复制原926个普通文件，拒绝hardlink与既有目标。**921文件字节不变，5文件仅应用before→proposed的RMS补丁delta**；新增shared module、profile、两个tests及绝对资源profile，共931文件。没有把现代native资源分支或actor裁剪夹入旧训练epoch。旧copied scripts/main/world/diffusion/decoder/PPO/PD/data/RNG接口保持原字节。

旧推理epoch保留完整 `PaperAMPNetwork`，并非现代actor-only保留布局。因此新测试的现代module-set断言用 [独立adapter](isolated_epoch_test_adapter.json) 适配旧四module；只有这一AST断言不同，真实暖启动、统计、恢复、三输出检查仍执行，production冻结测试未改。

隔离bundle：[source_manifest](../../../../output/isolated_training/mixed_state_action_terrain_protected_v1/source_manifest.json)，931文件manifest SHA `4227b3d8eef584ded2348519b4ac8d7809b21c16c9a8c2624a2d8d2b82ccdd07`。完整 [源码归档](isolated_terrain_protected_source_v1.tar.gz)：41371556字节，SHA `1982602b67cf21fac0cf3f1d0ac045a195e5af9f5d732b2a70c2277ede895fe7`。源码0444/目录0555、import/exec guard与同步ledger沿旧v3；资源symlink不等于OS级只读隔离，运行前guard重核资源全SHA。

## 命令与watcher

正确当前模板：[commands_v2.json](../../../../output/isolated_training/mixed_state_action_terrain_protected_v1/commands_v2.json)。原prepared `commands.json` 的三个训练argv正确，但继承了**错误旧watcher metadata路径**，已保存历史，不能按其watch字段运行。v2使用lexicalSDK `.venv/bin/python`、protected新训练/快照目录及exact-copy transport watcher；watcher尚未由本CPU任务启动，不是payload恢复/效果审核器。

fresh4 / sealed4恢复再4先完成实际payload验收，之后才按root预登记考虑256env×32×2000，同原state-action2k预算。training唯一字段变化 `task_terrain_statistics_mode=preserve_declared_suffix`；额外profile/source说明与from-scratch拒绝是该策略的准入边界。原reference_count61931520001与暖cap200000单独保留；prefix20/globalcount继续原数学，suffix1024 mean/var保护。新模式的既有checkpoint或策略未被静默转换。

## 后续实际效果验收

动作前真实1044观测已由独立reader看到地形裁剪：120个actor-frame、122880地形坐标、2061裁剪（1.677246%）、60/120有裁剪；这是旧negative1000实际输入，不是“RMS导致空闲”的因果结论。[动作前实际输入检查](../../pre_action_actual_observation_reader_v1/actual_cuboid30_v1/report.json)。新RMS模式须继续保留对应pre-action证据。未来full22、完整200s导航/失败inclusive风格、跌倒/到达/停留均沿冻结协议，不能由pilot通过或统计std保持替代。没有读取sealed34 captures，也没有默认模型推广。
