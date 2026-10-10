# 新 capture 入口窄审

2026-10-04。结论：当前三份新增源码 **无 remaining Required**。本次仅源码与纯接口检查，不是实际相机、蒙皮、物理或 cleanup 成功证明；原185冻结源及旧报告未修改。

- **导航参数保持：** `capture_driver_arguments` 复用原 `deployment.driver_arguments`，仅删除 `--physics-only`、打开 `--capture` 并替换捕获间隔。原 guide20/peer5/scene10/alpha0、causal speed/accel、共享 replan15、seed7、contacts/ground-support 参数保持。相机参数只传给 setup，不改 actor/controller 输入。
- **资源环境一致：** 初版临时资源环境仅覆盖 preflight，真实 delegate 会恢复旧覆盖值而先在原driver的 resource preflight 失败。root 已将实际调用也置于 `_visual_asset_environment`，两次预检与原driver后续固定 `data/smpl`/`data/smpl/textures` 一致。普通 Python 异常会恢复原环境；不会承诺 SDK fast-exit 时执行该退出逻辑。
- **单 App 与 factories：** 外层不创建 App/world，不写控制循环。原 `invoke_existing_driver` 仅调用 driver 一次，保留原 evaluation_actor/native_scene factory kwargs 和 argv 还原。纯preflight/help不启动SDK；新entry预先写的 `deployment_entry.json` 始终是预检记录，`physical_execution_verified=False`。
- **RGB/depth 接缝：** `capture_helpers` 修改原 `simulate` 模块的两个函数，恰一次原 setup、保留三项tuple ABI。depth annotator 接同一个 render product。每帧先且仅调用一次原 RGB frame helper，再读取 depth；没有新增 app update、物理步或 policy调用。原控制1、1+interval…对应同名 RGB/depth 文件。
- **不筛失败或空深度：** 深度原数组保留0与正无穷；拒绝NaN/负值。有限正值 mask 仅用于统计；`actual_synchronization_verified=False`，不假称有相机标定或实际同步证明。
- **fast-exit 边界：** 原driver在关闭world前写 result；SDK app.close 可能直接结束进程，外层 context/finally、detach及argv/environment还原不一定执行。helper逐帧flush depth index，源码只承诺到达context退出时尝试detach。实际RGB/depth、全帧capture diagnostics、物理step/RNG及真实退出状态须由后续实际产物审计判断；未观测时钟保持null。

作者式视觉效果、RGB-D同步、无额外物理变化均没有因本次source/seam通过而升级为实际结论。

## 必要 CPU 检查

`tests/test_paper_pedestrian_capture_entry.py`：**11 passed / 0.10s**，tool `861304` exit0；Ruff `8efb92` exit0。覆盖完整导航argv差异、相机/间隔拒绝、预检失败环境恢复、delegateonce与factory kwargs、普通异常环境/argv恢复和CLIhelp零Torch/NumPy/SDK导入。测试显式使用fake接口，不声称actualcapture。

helper作者另有9项fake seam检查，paper报告tool `744ac0` exit0、Ruff `f69999` exit0；本review没有重复运行。root报告实际SDK-freepreflight `90aaa7` exit0，亦非runtime相机质量证明。

最终小型源码身份与以上证据范围见 [new_entry_source_bindings.json](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/smpl_live_capture_source_review_v1/new_entry_source_bindings.json)。没有新增owner、launcher、指标数学或实际capture结果。

## 实际产物 auditor 的补充源码审阅

root 新增 `audit_paper_pedestrian_capture.py`；本次仅读源码和其测试，不执行 auditor 或重复测试。初版发现的空/缺失RNG可通过、depth index元数据未绑定数组，以及观察标志/数值类型边界，已在当前源码闭合，**无 remaining Required**：

- `result.status` 必须成功；runtime全部JSON arguments与entry parsed arguments一致。原driver未改写args，双方JSON化方式一致，未发现真实输入ABI误拒。
- 每张RGB均对应原控制1、1+interval…的诊断；六类状态delta必须为0；physics counter三项均为int、before非负、before=after且delta0。
- native clock的observed必须为bool。可观察时三个时间数值有限且不变；不可观察时before/after/delta全部保持None，并单独计数，没有把缺失API伪造成通过的时钟观察。
- RNG键必须恰为Torch CPU/CUDA两路，均实际observed/equal，前后64位hex SHA相同。空字典或缺CUDA不能再通过；结论范围没有扩大到Python/NumPy RNG。
- depth逐帧控制步、file/frame_index、shape/dtype/units及有限正值比例/min/max与同一已读NPY核对；正无穷/0背景保留，`actual_synchronization_verified`必须仍False。没有新增传感器同步或标定证明。

root报告新增必要回归后tool `afc691` **33 passed / 0.21s / exit0**；本review未重复执行。最终auditor/source身份见 [actual_payload_auditor_source_review_v1.json](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/smpl_live_capture_source_review_v1/actual_payload_auditor_source_review_v1.json)。这只批准当前payload一致性检查逻辑；实际capture质量及执行来源仍须真实产物与root真实closed记录。
