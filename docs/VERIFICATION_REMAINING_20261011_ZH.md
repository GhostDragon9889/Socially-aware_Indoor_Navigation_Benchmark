# IsaacLab 3.0 EA / Isaac Sim 6.1 继续修复与完整回归

日期：2026-10-11。本轮已完成下述迁移修复与回归；GitHub 可读取的证据为原始文件的逐字节副本，完整本机日志／轨迹仍保留在 output；逐行、全部功能及模型质量的整体验收仍未通过，边界见末节。

本轮接续 [上一轮报告](VERIFICATION_COMPLETION_20261011_ZH.md)。上一轮 41 个失败和 57 个初始化错误保留在原始记录中；本轮修复后重新执行完整测试，不使用 `-k`、`-m`、`--deselect` 或把失败改为 xfail。原有工作区修改保留。本轮迁移变更和验证证据按用户要求提交、推送到 benchmark/main；没有修改两个参考 SDK。

环境：项目 `.venv` → `.runtime/isaaclab3_ea_sim61/.venv`，基础 SDK `/project/EXP/IsaacLab/.venv`；Python 3.12.13、IsaacLab 包 17.0.2 / 产品 3.0 EA、Isaac Sim 6.1.0.0、Torch 2.11.0+cu128、Warp 1.16.0、NumPy 2.5.1。真实 GPU 为单张 RTX 4080 SUPER 16 GiB。参考 [IsaacLab 3.0 EA 官方文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html)、[Isaac Sim 6.1 官方文档](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/index.html)及本机源码；没有声称读完两个完整文档网站。旧 IsaacLab-Protomotions 用于接口和历史来源对照。

## 修复内容

| 问题 | 修复与验证 |
|---|---|
| 历史产物路径与当前源码混用，导致 S3-R、episode、native 入口的严格校验失效 | 使用完整 SHA／大小固定的只读历史源码；当前执行仍独立验证实时依赖。严格区分旧模型来源、当前运行、未完成的旧运行；历史失败输出保留。增加路径、类型、源码篡改及缺项拒绝测试。 |
| episode／future 的四 articulation 组环境仍直接使用 SDK root-link velocity 写入接口 | 路由到现有 COM 换算 helper，并绑定其完整来源。真实 256 actor 试验对逆序非连续 32 组写入非零角速度及旋转根状态，检查六组物理状态、未选组、输入不变、恢复原状态和零额外物理步。 |
| SDK6.1 在完成运行或构造失败后仍保留 native view／异常帧引用，退出可能段错误 | World、simulator、scene 在构造前建立所有权；释放 facade/core、完成帧、view、timeline/eventdispatcher 引用后完整 App.close。实际部分构造失败、早期缺资源及成功路径均测试，不以快速退出代替释放。 |
| 原生导航在 session layer 内创建资产时未恢复 USD edit target，后续 clone 失败 | `Usd.EditContext` 限定临时编辑范围；成功和失败均恢复原 edit target。实际四地图最小／最大有效 actor 配置通过创建、步进、关闭。 |
| Hospital 的 USD Point3f 坐标量化触发假几何不一致 | 期望坐标先按实际 float32 存储表示比较，保留原 1e-6 容差。原 37.6 坐标的约 1.53e-6 量化误差正确接纳；真实偏移、ULP 篡改、NaN/Inf、错误形状继续拒绝。 |
| 同一场景两次 USD 的匿名 layer ID 不同，导致相同初始状态不能配对 | 只规范化完整拥有的匿名 layer 闭包，保留原 RAM 内容的精确逆证明；未知匿名层和几何篡改继续拒绝。 |
| 原始轨迹 CPU reader 在做地图准入时导入 pxr | 地图 USD 核验移入独立子进程，父 reader 不导入 Torch/SDK；检查子进程真实退出、输入与源码完整 SHA。 |
| 完整风格参考读取仍直接比较旧目录字面路径 | 实际 manifest 143,512 字节及 SHA 与冻结值完全一致；使用已登记目录映射与完整身份比较，给 CPU reader 增加整文件逆变换证明。真实参考数组、全部派生样本和 50/100 聚类中心精确保持，保留改内容、改指标、换路径和缺项的拒绝测试。 |
| 缺少严格的 native 终止／路线统计类型检查 | falls 必须为非负真正整数，路线列表必须与 actor 数匹配且元素类型准确，拒绝 bool 冒充整数。 |
| benchmark 使用已移除接口、缺失 H1 资产及错误构造参数 | 改用项目维护的 H1-2，明确型号变化，支持小规模验证参数。真实 v1 复现退出 -11；v2 复用现有 SDK61Lifecycle，释放部分构造及异常链引用，实际进程退出 0，完整清理通过。 |
| 环境关闭时，组件或 UI 抛错跳过模拟器清理 | 保留原清理顺序，完整退出组件清理栈后，通过 finally 清理 UI 和 simulator；保留原异常链、KeyboardInterrupt 和非零退出码。组件／UI 单独失败、连续失败及真实 helper 链均有测试。 |
| 外部日志保留 traceback，使完成帧继续持有 native view | 先保留完整诊断，再对完成帧调用 traceback.clear_frames，清空局部引用并解除 traceback 链接，保留异常因果链；外部 traceback 对象仍可保留，但不再持有 frame locals。弱引用测试先红后绿。 |
| 普通 mesh terrain 缺少新版 InteractiveScene 所需的 env_origins | 使用新旧两份参考 SDK 共用的公开 grid_transforms 生成场景网格原点，保留共享世界 mesh 坐标；真实 RTX v1 复现缺失属性，4 个 CPU 回归先红后绿，两份 SDK 的实际 grid 函数／scene getter 共 8 组验证通过。 |
| 相机同一秒内重复开启会覆盖先前 frame 0 | 用原子创建会话目录及递增后缀，测试原 RGB／depth／metadata 不被覆盖；恢复原覆盖行为的变异版本被测试捕获。 |
| 深度帧全为 NaN/Inf 时 min/max 崩溃 | 空有效像素集明确显示无命中；保留原浮点 NPY 中的无效值，PNG 只作预览。 |
| 相机暗 uint8 帧被乘 255、CPU 共享缓冲覆盖旧帧、NaN/Inf 帧率被接受 | 按 dtype 处理 RGB，step 前复制深度快照，明确拒绝非有限帧率。真实 SDK ProxyArray CPU 数据验证与 19 个新增边界测试通过；全无效 burst 不再产生空集 mean/std 警告。 |
| ControlConfig 字典覆盖调用不存在的 ControlInfo.from_dict | 使用真实 dataclass 构造，检查所有六个字段和 MJCF 传递；非法字段拒绝，已有实例及 None 语义不变，删除掩盖问题的假方法 mock。 |
| 不完整／错误位置的 isaaclab namespace 导致路径定位 IndexError | SDK 路径发现明确处理不完整安装；冷进程及临时 namespace 回归通过。 |
| 上游维护检查无法表达已登记的本地维护提交 | 以四个明确文件、七条已登记文件维护变更（涉及三个不同提交）限定 HEAD/index 校验；1,017 项映射及 21 项负例验证。该结果不宣称整个当前工作区与上游相同。 |

旧训练器 Pop-Art 数值迁移、CUDA 保存／恢复、PointGoal 超时 bootstrap、规划边界、运动播放器等前轮修复也包含在本轮完整回归中。没有通过放宽历史来源校验或修改冻结文档来使测试变绿。

## 完整测试回归

完整 CPU 命令使用 `CUDA_VISIBLE_DEVICES="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=output/isaaclab61_reaudit_20261011/optional_rvo2:src:. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q --tb=short`。autoload 关闭是为隔离主机 ROS 等外部插件；本轮本地 RVO2 使用真实 CPython3.12 扩展，不是 mock，该可选二进制不随此提交发布。CUDA 补测的精确七个节点见进程 receipt 的 command。

最终 `final_cpu_v11` 在源码不变期间执行完整 pytest 收集：**4,981 passed、9 skipped、130 subtests passed**，实际退出 0，pytest 用时 569.72 秒。独立真实 CUDA 测试覆盖其中 7 个 CPU skip；按完整节点名去重后是 **4,988 passed、2 skipped、0 failed、0 error**，130 个 subtests 另列。剩余两项属于未安装的旧 IsaacGym 后端，不计作 Sim6.1 通过项。上一轮 98 个失败／初始化错误逐项对应当前通过结果，其中 96 个节点保留原名，2 个使用已记录的迁移替代节点。

166 条警告原样保留，包括第三方 MoviePy/ImageIO/NumPy 弃用提示和项目 terrain 的 Matplotlib 弃用提示；当前固定版本下通过不代表未来版本仍兼容。没有将警告计作测试失败，也没有将 skip 计作通过。

[去重测试汇总](verification_remaining_20261011/artifacts/final_test_summary.json)、[CPU 实际退出与源码快照](verification_remaining_20261011/artifacts/final_cpu_v11_execution.json)、[CPU 完整日志](verification_remaining_20261011/artifacts/final_cpu_v11.log)、[GPU 实际退出](verification_remaining_20261011/artifacts/final_gpu_tests_v2_execution.json)。

## 已测覆盖率

本轮十份成功执行的 coverage 记录合并后，`src/` 与 `scripts/` 的 617 个文件共 71,695 条可执行语句，已执行 46,149 条，**语句覆盖 64.37%**；21,690 条分支中覆盖 11,856 条，**分支覆盖 54.66%**。4,639 个函数语法范围中，1,234 个没有函数体语句执行记录。函数范围可嵌套，不能把其数量当成功能数量；执行到代码也不等于行为断言充分。

每个有实际执行行的项目文件均与该进程启动时的 SHA 及当前源文件比对。24 个 `.py:label` 形式的动态编译代码对象核验来源后排除物理行统计，没有把变换后的行号映回原文件。范围不含测试、第三方、SDK、文档及范围外动态源码；静态检查的 384,247 物理行与这里的可执行语句不是同一种统计。失败的原生进程记录没有用于增加成功运行覆盖率。

[覆盖率摘要](verification_remaining_20261011/artifacts/coverage_summary.json)、[逐文件 CSV](verification_remaining_20261011/artifacts/file_coverage.csv)、[函数体已执行／未执行语句](verification_remaining_20261011/artifacts/function_coverage.json)、本机完整 HTML 位于 `output/verification_remaining_20261011/coverage_html/index.html`、[覆盖源码逐运行身份检查](verification_remaining_20261011/artifacts/coverage_source_consistency.json)。

## 真实强化学习并行环境

| 运行 | 实际执行 | 结果与限制 |
|---|---|---|
| episode31：256 actors / 64 groups / 4 assets | 两次更新，16,384 个物理 transitions；64 controls / 256 substeps | 正常退出与完整清理；反序非连续重置的 root_vel 最大误差 2.98e-8，其余五组状态为零；四资产实际 COM 修正约 0.0203–0.02112 m/s。 |
| future81：256 actors / 64 groups / 4 assets | 同上，独立进程与独立 checkpoint | 同上；future 历史和计划状态与 episode 分别检查。 |
| 两种分组训练的独立 CPU 审计 | 每种 202 个有限张量；self/task/value RMS 各 16,384，AMP RMS 245,760，专家窗口 8,192 | 检查模型、两个 Adam、RMS、replay 及最后 rollout 一致性；CUDA 未初始化。 |
| 两种训练的重置隔离归档审计 | 每种最后 rollout 的 8,192 actor-controls；7,936 个具有下一帧的相邻记录 | 11 项检查均通过；episode 300、future 324 个重置 actor-controls。最后一帧重置后的状态未存档，未计入相邻帧证明。 |
| 独立 AMP 路径：128 环境恢复并继续更新 | 六组状态恢复；64 个反序偶数环境重置、64 个保持；更新 12→13，新增 4,096 transitions | 26 timeouts、0 真终止，四个网络均有有限非零权重变化；未恢复 PhysX solver cache，不保证之后逐 bit 相同轨迹。 |
| PointGoal 完整 CLI 链路 | CPU 训练 16,384 步→恢复至 32,768→20 episodes 评估 | 52 个有限 checkpoint 张量，13 个模型张量变化；成功率 0.2、碰撞率 0.8，仅为短训功能验证。 |

分组 episode/future checkpoint **没有实现完整物理环境恢复**，结果中 `resume_implemented=false`；独立 128 环境 AMP 恢复通过不能代替这一能力。短训模型没有被提升为正式默认模型，也不能据此宣称收敛。

[episode256 结果](verification_remaining_20261011/artifacts/final_grouped_episode256_v5/result.json) / [实际退出](verification_remaining_20261011/artifacts/final_grouped_episode256_v5_execution.json)、[future256 结果](verification_remaining_20261011/artifacts/final_grouped_future256_v5/result.json) / [实际退出](verification_remaining_20261011/artifacts/final_grouped_future256_v5_execution.json)、[重置归档隔离审计](verification_remaining_20261011/artifacts/grouped_reset_isolation_audit_v5.json)、[episode CPU 审计](verification_remaining_20261011/artifacts/final_grouped_episode256_v5_audit.json)、[future CPU 审计](verification_remaining_20261011/artifacts/final_grouped_future256_v5_audit.json)、[128 环境恢复](verification_remaining_20261011/artifacts/final_parallel_resume128/result.json)、[PointGoal checkpoint 审计](verification_remaining_20261011/artifacts/pointgoal_checkpoint_audit.json)。

## 原生导航和评估

四地图边界配置已实测：Office N3/N15、Warehouse N3/N15、Full_Warehouse N3/N15、Hospital N3/N11。每个边界运行检查真实 USD 克隆、物理步进和完整关闭。Hospital N12–15 属于登记协议主动拒绝范围。

完整 22 条动作序列的实际评估执行 11,982 controls 并正常退出，但使用两次更新的验证 checkpoint，结果为 **0/22 跟踪成功、22/22 序列观察到跌倒**，位置误差约 1.978 m。这是明确的模型质量未达标结果，不能以脚本 `status=passed` 覆盖。原生 200 秒配对也采用同一个预先选定的验证 checkpoint，保存全部原始帧并记录所有跌倒和重置；速度／加速度只按冻结指标定义使用同回合合法相邻帧，不跨重置求差分，不据此宣称论文效果。

[22 条结果](verification_remaining_20261011/artifacts/full22_episode256_v1/result.json) / [实际退出](verification_remaining_20261011/artifacts/full22_episode256_v1_execution.json)。SDK 本轮接触输出为 normal force；接触计数不是完整切向力测量。支撑脚速度指标是 body-origin 速度代理，不是接触点滑移速度。


Office N3、seed 7 的 SFM 与 diffusion 各运行 6,000 controls / 200 秒 / 18,000 actor-controls / 24,000 physics substeps，实际进程退出 0，完整关闭通过。CPU 原始轨迹审计及配对风格计算均实际退出 0；终止缓存修复后重新执行 diffusion raw_v3 和 style_v3，10,605 个非身份数值字段与 v2 完全一致；36 项初始状态、资源和 RNG 对齐。冻结参考和 50/100 聚类中心未重拟合，指标数学未变。

| 完整 200 秒指标 | diffusion | SFM |
|---|---:|---:|
| 跌倒终止 | 581 | 597 |
| 三个 actor 各自到达路线节点 | [0, 0, 0] | [0, 0, 0] |
| 静态／同伴 normal-force 接触 actor-controls | 0 / 0 | 0 / 0 |
| 同回合根速度均值（m/s） | 0.922020 | 0.950166 |
| 受支撑脚 body-origin 速度代理均值（m/s） | 0.162101 | 0.173337 |
| 合法同回合风格样本 | 17,416 | 17,400 |
| 共同速度分箱等权 MMD² | 0.231474 | 0.210943 |

这些结果表示本次短训验证模型的实际观测，不能证明 diffusion 优于 SFM，更不能作为论文数值复现。原始轨迹父审计进程没有导入 Torch/SDK；风格计算允许 CPU Torch 读取参考，但没有初始化 CUDA。

两次 200 秒物理运行使用先前 b4f85259… 生命周期，本次 terminal-only 改动经整文件逆证明仅增加缓存释放及关闭条件检查；没有重标为新源码的 200 秒运行。新公共关闭代码另以 Office N3 / 30 controls 实测通过。两次长运行期间另有 CPU/相机/分组修复：SFM 外部快照记录 6 个变化文件，diffusion 记录 4 个。两臂运行所固定的 native runtime/producer/lifecycle 来源保持不变，之后的 CPU 审计独立验证对应记录；这里不宣称整个工作区在长运行期间都未变化。

[历史 v2 来源链与科学结果精确等价审查](verification_remaining_20261011/artifacts/final_evidence_assembly_independent_review.json)与[39 项独立复算审查](verification_remaining_20261011/artifacts/final_native_pair_independent_review.json)重新加载冻结参考，按 JSON 序列化规范逐值核对 pose45、全部配对速度分箱／选样索引与 MMD；5/7 共同支持分箱，未重拟合或隐藏失败。

[diffusion 真实运行](verification_remaining_20261011/artifacts/native_final_v4_Office_N03_diffusion_C6000_execution.json)、[SFM 真实运行](verification_remaining_20261011/artifacts/native_final_v4_Office_N03_sfm_C6000_execution.json)、[diffusion 完整原始审计](verification_remaining_20261011/artifacts/native_final_dp200_raw_v3_base.json)、[SFM 完整原始审计](verification_remaining_20261011/artifacts/native_final_sfm200_raw_v1_base.json)、[配对风格结果](verification_remaining_20261011/artifacts/final_native_pair_style_v3.json) / [实际退出](verification_remaining_20261011/artifacts/final_native_pair_style_v3_execution.json)。

## 真实 RTX 相机

`rtx_camera_v1` 首次实测因地形缺少 env_origins 失败，随后在异常关闭时实际退出 -11；失败日志保留。修复地形属性后，验证脚本在完整生命周期关闭前，先停止已拥有的 simulation、通过公开 Replicator stop/wait 完成渲染关闭，并执行 6 次 Kit update，确认恢复协程全部完成。

`rtx_camera_v2` 两环境真实执行 8 controls / 32 physics substeps，实际退出 0，完整清理成功：网格原点分别为 (1,0,0)、(-1,0,0)，RGB 为 uint8 [2,48,64,3]，深度为 float32 [2,48,64,1]，两个中心深度均为 2.5 m。项目 C 键回调保存的数据与原生输出精确一致，headless burst 保存 6 个原始深度帧，RobotCameraRecorder 保存 2 组 RGB/depth。该测试调用了项目的三种保存接口；关闭调度由输出验证脚本负责，不能代替真实主程序 env.close 路径的验收。

`rtx_camera_failure_v2` 在两环境相机初始化后主动触发 RuntimeError，实际退出 1（预期），0 controls，完整清理成功、无清理错误，6 次 Kit update 后无残留恢复任务。成功和异常两次均保留完整父进程退出证据；v1 的 -11 未被删除。

[异常清理结果](verification_remaining_20261011/artifacts/rtx_camera_failure_v2/result.json) / [实际退出](verification_remaining_20261011/artifacts/rtx_camera_failure_v2_execution.json)、[RTX 结果](verification_remaining_20261011/artifacts/rtx_camera_v2/result.json)、[实际退出](verification_remaining_20261011/artifacts/rtx_camera_v2_execution.json)、[RGB 截图](verification_remaining_20261011/artifacts/rtx_camera_v2/output/renderings/camera_check/rgb_env0.png)。

## 生产关闭路径与原生退出崩溃修复

`sdk61_owned_close.py` 将 Sim6.1 终止清理接入真实 `BaseEnv.close → IsaacLabSimulator.close`，覆盖一个或两个项目 articulation、普通 rigid object、相机与拥有的键盘引用。核验并释放 borrowed view，停止渲染、等待 Replicator 清理、执行完整 App.close；部分初始化失败时也触发 STOP。重复关闭同一实例返回原报告。

保留的 `production_camera_close_v1`、`production_camera_bounded_01` 和 `production_camera_late_attach_03` 均实际退出 **-11**。关闭后附加 GDB 抓到 `__run_exit_handlers → omni.physx.bindings._physx → list_dealloc → _PyObject_Free`。进一步反汇编确认：该 PhysX 绑定将 clone 回调返回的 `['/World/template', '/World/envs']` 保存在函数 static 的 `py::list` 中，C++ atexit 析构在 Python finalization 后仍尝试递减引用，导致崩溃。

修复只适用于完整 SHA **8feacb575951aa814dbda988b4e83bf8f3230167b0b4856c1e1e44e3f6a28354** 的 Linux / CPython3.12 绑定。使用已加载 ELF 的实际地址定位偏移 `0x193640`；先验证整个二进制 SHA 和列表内容，再在 GIL 下释放恰好一个 native owner 引用，将 static 指针清空。未知二进制或未知缓存状态在修改内存前拒绝；SimulationContext 未清理时禁止释放缓存。不会修改磁盘上的 SDK 文件。该处理限于终止整个 App，关闭后不能在同进程重新创建 PhysX App。

缓存释放接入公共 `SDK61Lifecycle`，覆盖生产、分组训练、benchmark 及评测入口。历史 manifest 保持原字节；新来源登记对公共生命周期执行整文件逆变换校验，证明其他字节未变，并绑定当前 helper 的完整 SHA。118 项针对性回归通过，包括准确引用计数、未知状态拒绝、清理顺序、异常继续 App.close 和 context 清理失败不触碰缓存。

冻结代码的 `production_physx_fixed_v3` 实际退出 0（48.03 秒），两环境 / 四个 H1-2 实例、rigid cube 和 RTX camera 执行 8 controls / 32 substeps，两个中心深度均为 2.5 m；三种保存接口、native 字段释放、外部引用保留及重复关闭通过。组件关闭故障实际退出 1（预期，48.03 秒），外存 traceback 的完成帧已清空，原 RuntimeError 保留。初始化后故障实际退出 1（预期，48.11 秒），0 controls，完整清理通过。最终 `production_physx_destructor_v3` 在 C++ 析构地址 `0x10b7a0` 设置断点，观察到 static owner 指针 **0**，真实子进程退出 0（52.28 秒）；它是析构行为的直接证据，不能用 debugger 包装进程的返回码代替。

这一探针使用最小 BaseEnv shell 和真实 SDK 资产，并非完整 CrowdSim task、正式模型或 GUI 验收。最新冷执行计划核验 176 项源码、7 项输入及四个真正退出 0 的冷预检；没有启动 2,000 更新训练或 288 臂实验。两个参考 SDK、冻结数学指标和历史失败证据保留。

[终止缓存来源逆证明](verification_remaining_20261011/artifacts/terminal_cache_source_inverse_proof.json)、[原生崩溃栈](verification_remaining_20261011/artifacts/production_camera_late_attach_03.gdb.log)、[当前正常运行](verification_remaining_20261011/artifacts/production_physx_fixed_v3_execution.json)、[组件故障](verification_remaining_20261011/artifacts/production_physx_component_failure_v3_execution.json)、[初始化后故障](verification_remaining_20261011/artifacts/production_physx_initialize_failure_v3_execution.json)、[实际子进程退出](verification_remaining_20261011/artifacts/production_physx_destructor_v3_execution.json)、[C++ 析构断点结果](verification_remaining_20261011/artifacts/physx_cache_destructor_terminal_v3.json)、[当前计划来源复验](verification_remaining_20261011/artifacts/current_episode_plan_v12/final_admission_review.json)。

## 安装包及静态检查

实际构建约 41 MB wheel，隔离安装；逐字节检查 wheel 中 470 份 Python 源码与当前源文件一致。禁用开发环境自动注入源码的 hook 后，确认五个核心包均从 wheel 安装目录导入；六个 console `--help`、info/status、自带 PointGoal 配置的训练及评估共 11 条命令通过。源目录命令继续显式使用 `SOCIAL_NAVIGATION_ROOT`；没有把依赖完整源码／证据目录的研究流程声称为独立 wheel 功能。

当前 Python 清单为 **1,763 个文件、384,247 物理行**，逐文件记录 SHA、AST 和物理行数；AST 失败及 Ruff 致命错误均为 0（F821/F822/F823/F901/E9）；这证明读取与语法检查，不等于逐行人工审查或逐行动态执行。非 Python 检查包括 186 个 JSON/YAML/TOML/shell/JS 文件；两份 JS 的 13 个 V8 CPU 协议用例及一个拒绝假退出码的变异检查通过。仓库已有有说明的 Ruff 豁免保留。另附三份可复现原生关闭的探针／父进程记录器，独立 AST 与致命 Ruff 检查通过。

[源码清单](verification_remaining_20261011/artifacts/inventory.json)、[致命静态结果](verification_remaining_20261011/artifacts/static_findings.json)、[配置与脚本语法](verification_remaining_20261011/artifacts/nonpython_syntax.json)、[最终非 Python 身份复验](verification_remaining_20261011/artifacts/nonpython_final_consistency.json)、[实际安装包运行](verification_remaining_20261011/artifacts/installed_wheel_terminal_validation.json)、[benchmark 真实成功退出](verification_remaining_20261011/artifacts/final_isaaclab_benchmark_v5_execution.json)。

## 验收边界

尚不能声称“每行、每项功能、任何参数组合都通过”或“没有任何 bug”。完整注册矩阵包含 156 个候选、144 个准入、12 个拒绝、288 个计划运行；只读准入检查不等于 288 个原生实验全部执行。多 GPU／多机 DDP、长训练收敛、全部地图／随机种子／模型组合、GUI/VR/真实机器人、旧 IsaacGym 与其他模拟器完整后端不在本机 Sim6.1 运行通过结论中；Docker 构建及全部 C/C++ 后端也未做完整认证。真实 RTX 成像与三种保存接口已验证；生产主入口、GUI 交互与各场景全部相机配置仍需分别验收。

完整 CPU／CUDA 与已选择的功能回归通过，但已定位的生产退出崩溃在固定绑定上修复并完成当前代码验证；覆盖率和模型质量仍未达到全部功能验收要求，整体验收未通过。原失败和中断日志不删除，修复前的 -11 退出和混合源码版本中断不会被计为当前通过证据。


[当前机器汇总](verification_remaining_20261011/artifacts/final_validation_summary.json)、[v3 全部非身份数值字段等价检查](verification_remaining_20261011/artifacts/final_native_pair_v3_numeric_equivalence.json)、[保留全部进程观察](verification_remaining_20261011/artifacts/execution_inventory_current.json)。

## 原生关闭复测命令

GPU 独占运行，先完成仓库所述 SDK / 资源准备。命令从项目根目录执行；探针使用当前 SDK、H1-2 和 RTX，输出目录必须尚不存在：

```bash
PYTHONPATH=src:. .venv/bin/python docs/verification_remaining_20261011/probes/execute_native_probe.py terminal_success \
  .venv/bin/python -X faulthandler docs/verification_remaining_20261011/probes/validate_production_camera_current.py \
  --num-envs 2 --output output/terminal_success
```

父进程将真实子进程退出码、源文件 SHA 和执行中变化写入 `output/terminal_probe_runs/`。故障复测分别追加 `--fail-component-close` 或 `--fail-after-initialize`，使用新运行名称和新输出目录，预期真实子进程退出 **1**；父记录器自身也以非零退出。只检查子进程的 JSON passed 标志不足以判断退出成功。
