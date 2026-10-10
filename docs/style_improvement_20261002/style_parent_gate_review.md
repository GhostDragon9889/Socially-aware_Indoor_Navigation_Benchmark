# 风格评估的父动作隔离门禁

## 修复范围

新训练库允许将同一长动作切成多个不重叠的30Hz片段。`source_path` 因此可能为 `original.motion#frames=250:500`。按完整字符串做训练/参考交集会将同一个原动作的不同片段视为不同数据，漏掉评估泄漏。

本次只修改 `CrowdSim/humanoid/style_evaluation.py`、`scripts/evaluate_pedestrian_style.py` 和专属回归 `tests/test_pedestrian_style_evaluation.py`，没有修改训练器、应用预检、资源筛选或既有评估报告。修改前的三个文件及哈希保存在 `style_parent_gate_before/`。

## 当前契约

- NumPy 模块新增 `parent_source_identity(clip)`：规范路径分隔符与 `.` / `..`，仅剥离合法的 `#frames=start:stop` 后缀。
- 存在 `parent_source_path` 时必须与 source 的真实父路径一致；不能通过声明一个不同父路径隐瞒泄漏。空值、错误后缀、逆序/零长度区间均拒绝。
- `parent_control_frame_start/stop` 若出现，必须同时是整数且与后缀逐值一致；分段 clip 的 `frames` 必须等于半开区间 `stop-start`。
- 原始30/50Hz依赖字段 `parent_source_frame_start/stop` 不被误当成重构30Hz片段长度。新冻结窗口保存父路径、control-frame边界、original_motion_id及原源依赖边界（若提供）。
- `assert_parent_split_disjoint(document)` 核对整个 train/validation/test 清单的父动作交集。允许同一个 train 父动作提供多个片段，但不允许它进入任一 held-out split。
- CLI 还将**完整冻结 reference 的 validation 与 test 清单**同新 run 声明的训练清单交叉比较，不能仅靠当前抽中的256个窗口覆盖泄漏检查。
- 样本加载再次核对冻结窗口与实际 manifest 的 parent identity；实际选中数组的 SHA 校验继续执行。

身份门禁不自行判断同 train 父动作片段是否重叠；最终筛选器及其 manifest 负责这个契约。它也不能证明跨数据集人物身份独立，或公开 PACER 预训练从未见过这些动作。

## CPU 验证与旧结果保持

42项 focused 回归通过，三个文件 Ruff 通过。新增用例覆盖不同片段的同父源泄漏、未抽中验证/测试动作的泄漏、伪造父路径、Windows分隔符与dot别名、非法帧后缀、bool帧边界、长度不一致，以及同train父动作合法片段接受。NumPy模块和CLI顶层导入仍不加载Torch/Kit。

使用同一原319训练声明、同一22clip validation reference，对两个既有真实200秒capture `shared_independent_amp_raw_v1` 与 `shared_independent_amp_continuous_timed_v1` 重新执行CPU评估。以下内容对修改前报告逐值完全一致：generated/reference统计、所有inclusive speed-conditioned分数、upright子集、heldout clip-group calibration和旧disjoint摘要。

冻结 `heldout_reference.json` **字节完全一致**，SHA256保持 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`。实际选中body/rootrot数组SHA也保持不变。因此本次加强隔离检查，没有改变原输入的采样或评分。

最终源的重跑证据在 `style_parent_gate_legacy_final_cpu/`；较早的 `style_parent_gate_legacy_cpu/` 是增加跨完整冻结reference门禁前的中间验证快照。机器报告：[style_parent_gate_review.json](style_parent_gate_review.json)，测试记录：[style_parent_gate_tests.xml](style_parent_gate_tests.xml)。

当时可读取的中间 `manifest_v2.json` 的train2848/validation22/test34父源也与冻结reference相互隔离；这只证明数据身份门禁，不是接受其动作语义、作者确切2848清单或最终分段库。最终分段库冻结后再运行同一门禁。
