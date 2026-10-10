# 原生N3–15初态捕获：最终CPU准备证据

本轮补齐同AMP的DP/SFM原生入口，目标是后续公平比较导航与动作风格。
[NavIsaacLab §IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)使用同一个AMP、8个场景、3–15人、200秒。
这次是CPU源码与资源准备，没有启动Isaac SDK、加载策略或进行人体物理评价。

## 最终实现

- [native_paired_initial.py](../../../CrowdSim/humanoid/analysis/native_paired_initial.py)：
  在独立模块中复用原N4初态工具，9个明确接缝、13次替换可完整逆还原。
  原来的两次同刻COMMON/native读取、完整RNG读取、前后物理计数检查、独占档案写入与逐通道SHA校验保留。
  每个人的24身体、69 DOF、AMP/planar历史、episode/reference ID、路线均完整保留。
  新增原生标签、lower-left origin、bounds和原NativeGridScene SDF四个地图通道。
  合计CPU35通道；已初始化的一块CUDA增加1通道，实际数量按RNG元数据核验。
- [evaluate_shared_social_native_paired.py](../../../scripts/evaluate_shared_social_native_paired.py)：
  原生两臂使用相同采集点：actor准入/设定种子之后、planner构造之前。
  仅添加import和capture两处可逆接缝；原控制、reset、参考适配、测量与终态生命周期保持。
  SFM入口已经派生，不再次应用SFM替换。hook不重设/恢复RNG，不额外推进物理。
  临时模块的三个回调在finally中恢复，避免通过globals引用循环保留World/capture。
- [正式配对CLI](../../../scripts/compare_shared_social_native_initial_conditions.py)：
  默认只接纳原生World声明及完整来源/资源档案；CPU夹具、旧N4 schema、缺字段、地图/路线/RNG漂移被拒绝。
  匹配仅证明已记录初态一致，**不证明SDK实际退出、质量通过或作者实验复现**；这些仍须外部退出与raw审计。

RNG/档案接口沿用[PyTorch2.11 state getters](https://docs.pytorch.org/docs/2.11/random.html)
和[NumPy2.3压缩档案](https://numpy.org/doc/2.3/reference/generated/numpy.savez_compressed.html)，
不恢复或抽取随机数，读取NPZ时禁止pickle；原Torch RNG逻辑未改动。

## 实际检查

[CPU_registration.json](CPU_registration.json)在运行前登记固定Warehouse N3/N15资源、来源与命令。
[CPU_preparation_execution.json](CPU_preparation_execution.json)保存每个子进程的真实returncode及完整日志绑定。
外部工具启动`a016eb`，终态`4df74c`实际退出0；该退出仅对应CPU准备器。

| 检查 | 实际结果 |
| --- | --- |
| 3人DP / 15人SFM源码检查 | 两个子进程退出0；未导入Torch，SDK未启动 |
| 严格S1 Pop-Art / formal DP40k / 原生资源预检 | 两个子进程退出0；没有加载网络 |
| 相关数值、来源、档案、RNG和实例释放检查 | [79通过，7.76秒](tests_final.log)，子进程退出0 |
| 代码质量检查 | [Ruff通过](ruff_final.log)，子进程退出0 |
| 准入条件突变 | [3失败、39通过](mutation_expected_failure.log)；恢复原字节后正常套件全通过 |
| 实际CPU地图通道 | 两份压缩档案与实际准入网格逐字段SHA一致；shape370×251、origin[-12.5,-18.5]，未拟合坐标 |
| 训练隔离 | 23项S3活动来源守卫保持；未改训练数据、参数、权重或SDK文件 |

突变把`require_live`改为`not require_live`，N3/N15的CPU夹具准入及正式CLI拒绝测试确实失败。
恢复SHA写入[mutation_review.json](mutation_review.json)。临时回调在成功/异常两种路径的释放均另有检查。
这两份`map_channels_only.npz`只有真实地图，没有拼接合成人体或RNG，不是live初态档案。
单元测试使用明确的合成CPU状态，未作为真实物理结果发表。

## 失败与中间版本保留

- [v1](../shared_social_native_initial_preparation_v1)：外部`0c7977/06f77f`实际退出1；
  资源预检子进程退出2，因为准备器选择docs作为输出路径，被原入口拒绝。
  日志及两份地图档案保留，未削弱输出限制。
- [v2](../shared_social_native_initial_preparation_v2)：只把输出子路径改到仓库output；
  外部`c8bf83/192a75`实际退出0，77项检查通过。
  此后代码审查补充finally回调释放，v2原完整源码和测试分别保留于
  `native_initial_before_mutation.py`及`test_native_paired_initial_tested_v2.py`。
- v3重新登记、重跑突变与全部检查；[final_review_registration.json](final_review_registration.json)说明差异。

## 后续实测

GPU当前由S3训练owner独占。先完成其终态审计、封存、独立评价与严格补充准入，
再用新进程运行原生N3/N15 30步smoke，核查真实策略推理、native/floor/peer视图和正常退出。
随后两臂在同一场景/人数/种子/权重/配置下运行200秒，并用正式CLI核对初态。
还需扩展N3–15完整physical/ground/plan原始审计、独立全身风格读取和多seed注册。
八场景清单及作者计数分母仍未恢复；当前没有达到/超越论文完整效果的证据。
