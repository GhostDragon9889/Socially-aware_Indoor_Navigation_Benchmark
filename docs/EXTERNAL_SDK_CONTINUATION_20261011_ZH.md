# 外部 SDK 后续核验、配置修复与物理验收缺口

核验延续至 **2026-10-12**。本轮修复了五类 SDK 配置故障及两类机器人 CNN 配置故障，当前收集的 **5,167 个普通测试用例均有执行记录：5,165 通过、2 个旧 IsaacGym 后端跳过**，另有 130 个子测试通过。真实四机器人深度相机 PPO 训练和续训通过；YOHO、SecondFloor 的本轮 60 秒局部物理验收通过。**NewMetro、ThirdFloor 尚未通过物理质量验收，全项目执行覆盖未达到 100%，不能声称全部功能通过或不存在 bug。**

本轮接续[已发布的外部 SDK 核验](EXTERNAL_SDK_VERIFICATION_20261011_ZH.md)，使用外部 `/project/EXP/IsaacLab/.venv/bin/python`。SDK 源提交为 `ae37b028ea415c91ea2bc32609efcd759ed2b974`，Python 3.12.13、Isaac Sim 6.1.0.0、Torch 2.11/cu128；主机为单张 RTX 4080 SUPER 16GB。项目附加包安装在检出目录内，由 SDK Python 执行，项目不替换 SDK 的 Torch、NumPy、Warp 或 USD 提供者。外部绑定、完整项目内安装、中文及空格路径克隆验证、全部模型文件清点和前轮 256 并行环境训练证据见前轮报告及[配置指南](guides/external_sdk.zh-CN.md)。

迁移参考[IsaacLab 3.0-EA 官方文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html)、[3.0 迁移指南](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/migration/migrating_to_isaaclab_3-0.html)及[Isaac Sim 6.1 文档](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/index.html)，并结合本地 IsaacLab 和 IsaacLab-Protomotions 源码核对。本轮场景失败不能仅凭版本差异归因于求解器；尚无这样的因果证据。

## 修复与兼容性

| 已复现问题 | 修复行为 | 原始失败证据 |
| --- | --- | --- |
| 可选 RVO2 输入失败仍发布新解释器绑定 | 发布之前核验原生依赖路径、构建收据及哈希 | [配置故障日志](external_sdk_continuation_20261011/artifacts/transactions_before.log) |
| 附加包发布后校验失败丢失安装日志 | 从实际已发布目录保留日志，再回滚附加包；保留原始异常 | 同上 |
| 中断遗留固定 `.partial` 阻止下一次配置 | 独立临时文件暂存，绑定最后发布，并恢复已替换文件 | 同上 |
| 附加包软链接误用其他检出目录 | 拒绝外部目录、软链接及不匹配绑定 | 同上 |
| 持续 I/O 故障使回滚失败时删掉唯一旧文件快照 | 保留恢复快照，原始异常附注路径，继续尝试其他文件恢复 | [持续故障](external_sdk_continuation_20261011/artifacts/rollback_persistent_before.log)、[发布故障](external_sdk_continuation_20261011/artifacts/publish_recovery_before.log) |
| 机器人 CNN 错误拒绝合法的零 padding | padding 单独按非负整数解析，允许共享或逐层配置 | [CNN 修复前日志](external_sdk_continuation_20261011/artifacts/robot_sensor_before_source.log) |
| CNN 的 `zip` 静默截断不等长逐层列表 | 共享值广播至每一层，不一致列表在构建前拒绝；默认配置随通道数展开 | 同上 |

两份 PPO 实现分别修复在 `CrowdSim.ppo` 和 `crowdsim_ppo`，保留各自模块及序列化身份。两命名空间的 Depth、Map、RGB 六种默认编码器与 Git 基线比较，默认模型参数字节及输出一致；实际已有机器人检查点严格加载通过。[默认模型兼容性记录](external_sdk_continuation_20261011/artifacts/robot_encoder_compatibility_verified.json)。两实现及相关配置、课程、生命周期回归共 [134 项通过](external_sdk_continuation_20261011/artifacts/both_robot_verified.xml)。

配置故障、发布/回滚、锁竞争、软链接、指纹、启动入口、资源损坏和序列化恢复等新增及相关回归，最后一次 SDK 配置修复后 [110 项通过](external_sdk_continuation_20261011/artifacts/runtime_recovery_verified.xml)，已有 SDK 回归另 [11 项通过](external_sdk_continuation_20261011/artifacts/external_existing_verified.xml)。持续操作系统故障下，自动回滚只能尽力执行；恢复快照和异常说明用于人工排除 I/O 故障，不能将失败当作成功配置。

Python 3.10 主机入口在切换 SDK 解释器前关闭用户 site，真实 **3.10→3.12** 启动已核验。[启动日志](external_sdk_continuation_20261011/artifacts/host310_bootstrap_v2.log)。最后一次真实配置、原生兼容准备和冻结输入检查正常完成：原生覆盖文件为零，1,338 份冻结输入保持原样。[配置记录](external_sdk_continuation_20261011/artifacts/configure_recovery_verified.log)、[原生准备](external_sdk_continuation_20261011/artifacts/native_prepare_verified.log)、[冻结输入检查](external_sdk_continuation_20261011/artifacts/frozen_prepare_verified.log)。

## 完整测试与身份核对

新增入口可复现 CPU 与真实 CUDA 的独立进程执行：

```bash
python3 scripts/run_external_python.py scripts/run_project_tests.py --output-dir output/tests_new
```

CPU 合约包含“CUDA 尚未初始化”的断言，因此先在隐藏 CUDA 的进程中执行，再从实际 JUnit 跳过原因发现 CUDA 用例，在新进程中沿用终端 GPU 选择重试。入口保留日志、JUnit 和实际退出码；子进程失败、缺 XML、CUDA 重试数量不匹配均产生失败。硬件或旧后端缺失记为 `passed_with_gaps`，跳过不计通过。

完整隔离入口真实退出 0，CPU **5,156 通过、9 跳过、130 子测试通过**，随后实际 CUDA **7 通过**。这次全量运行收集 5,165 个普通用例；之后仅 `external_runtime.py` 增加持续 I/O 故障修复，并用上述 110+11 项相关回归复测，同时新增两个故障用例。最终当前收集清单与执行身份并集完全一致：**5,167 个普通用例、5,165 通过、2 跳过、0 失败**。重复执行与 130 个子测试单独记账。[入口结果](external_sdk_continuation_20261011/artifacts/full_isolated_current/result.json)、[CPU XML](external_sdk_continuation_20261011/artifacts/full_isolated_current/cpu.xml)、[CUDA XML](external_sdk_continuation_20261011/artifacts/full_isolated_current/cuda.xml)、[用例并集](external_sdk_continuation_20261011/artifacts/test_union_summary_final.json)、[身份核对](external_sdk_continuation_20261011/artifacts/collection_identity_receipt_final.json)。这不是最后一次修复之后重新运行整套测试的声明。

先前错误使用 `--import-mode=importlib` 导致 14 项旧测试交叉导入收集错误；完整 CUDA 可见的混合进程产生 31 项 CUDA 初始化、CPU 生命周期夹具及模块清点污染失败。失败日志均保留，正确隔离后通过。主机第三方 ROS pytest 插件自动加载失败亦保留，项目入口关闭插件自动加载。[收集失败](external_sdk_continuation_20261011/artifacts/full_cpu.log)、[混合运行失败](external_sdk_continuation_20261011/artifacts/full_cpu_default.log)、[插件失败](external_sdk_continuation_20261011/artifacts/gpu_clean.log)。

## 真实训练及深度传感器

本轮使用 `data/pretrained_models` 中的 TRACE、AMASS/PACER 及公开 PACER 权重。权重来源和实际哈希保留在每次运行输入、结果及前轮模型注册表中。第三层候选 v4 还测试了前轮从指定模型训练出的 SDK 6.1 检查点，v5 测试了指定目录中的公开 `pacer_no_shape/Humanoid.pth`；两次仍失败，未将它们提升为默认模型。

原生 `CrowdSim.ppo.train_ppo.train_loop` 在 Warehouse/NovaCarter 中使用 **4 名 SMPL 行人、4 台机器人**、真实 RTX 深度相机和外部 SDK 运行。观察脚本只记录生产代码返回的数据，不替换传感器或物理实现。每台机器人的 `distance_to_image_plane` 原始输出均为 `224×224×1`，处理后的深度为 `224×224`，有限、非零，深度回退标志为空；深度 CNN 使用两层通道和共享 `padding=0`，验证本轮修复的实际输入。

| 阶段 | 真实结果 | 更新证据 |
| --- | --- | --- |
| 独占 GPU 训练 | 退出 0，64 个机器人 transition，4 次 PPO 更新，20 次深度观察读取（含 bootstrap） | vector、depth、map、fusion、critic 五组共 48 个张量改变 |
| 从检查点续训 | 退出 0，64→80，新增 16 个 transition、1 次 PPO 更新，5 次深度观察读取 | 五组继续改变；57 个 Adam 参数状态各推进一次更新，57 个模型张量改变且全部有限 |

[训练观测](external_sdk_continuation_20261011/artifacts/robot_depth_serial/robot_depth_observation.json)、[训练实际退出及源码身份](external_sdk_continuation_20261011/artifacts/robot_depth_serial/robot_depth_owner.json)、[续训观测](external_sdk_continuation_20261011/artifacts/robot_depth_resume/robot_depth_observation.json)、[续训实际退出](external_sdk_continuation_20261011/artifacts/robot_depth_resume/robot_depth_owner.json)、[检查点与 Adam 验证](external_sdk_continuation_20261011/artifacts/robot_depth_checkpoint_verified.json)。检查点使用已知配置类的受限加载，原始观察脚本以 `.py.txt` 保留在证据目录。

第一次相机任务与另外两个物理任务并发导致 16GB 显存耗尽，出现 Vulkan `ERROR_OUT_OF_DEVICE_MEMORY`，由任务所有者中断并保存失败。该次不计入成功训练，随后独占 GPU 重跑通过。[OOM 日志](external_sdk_continuation_20261011/artifacts/robot4_depth_train.log)、[中断记录](external_sdk_continuation_20261011/artifacts/robot_depth_interruption.json)。

以上证明短程权重、Adam 和课程状态续训；未证明原生机器人训练的物理状态/RNG 精确续接、长期收敛或导航成功率。前轮 **256 并行物理环境、131,072 个 transition、16 次更新、8,192 个恢复后 transition** 的 PACER 训练及部分环境重置隔离证据仍有效，见[前轮报告](EXTERNAL_SDK_VERIFICATION_20261011_ZH.md)。多 GPU 不以单 GPU 的并行环境结果替代。

## 场景物理验收及未解决问题

使用原场景几何、路线和固定输入，控制频率 30Hz；每次申请 1,800 个控制步、60 秒物理仿真。实际退出码由 `subprocess.Popen.wait` 获取，源码和输入身份、清理结果另记。退出 0 与姿态、接触、速度等独立质量验收分别记录。未降低验收门槛，所有候选仅保存在证据目录，**没有作为默认 CLI 配置采用**。

| 场景/实验 | 实际控制步 | 实际退出 | 验收结果 |
| --- | ---: | ---: | --- |
| YOHO，4 行人 | 1,800 | 0 | 接触局部筛选、姿态、速度通过；0 跌倒、无初始化后重置 |
| SecondFloor，1 行人 | 1,800 | 0 | 本轮三项质量门通过；0 跌倒 |
| NewMetro 原配置，4 行人 | 1,800 | 0 | 未通过：静态接触峰值约 61.82N，actor0 抬臂比例超限，actor1/2 低速比例超限 |
| NewMetro v1/v2 | 各 1,800 | 各 0 | 接触与姿态通过，低速比例仍超限 |
| NewMetro v3 | 1,800 | 0 | 接触通过，actor2 右臂、1.5m/s 速度上限及低速比例仍有失败 |
| NewMetro v4 | 1,764 | 1 | 循环路线停滞超过 20 秒，任务明确拒绝 |
| ThirdFloor 原配置 | 151 | 1 | 所有采样轨迹及因果制动均未获安全准入 |
| ThirdFloor v1 | 1,685 | 1 | 提前制动造成长期停滞，路线没有有效推进 |
| ThirdFloor v2/v3 | 260 / 170 | 各 1 | 轨迹准入失败 |
| ThirdFloor v4/v5，不同 PACER 权重 | 835 / 130 | 各 1 | 轨迹准入失败 |

十三次运行均记录零跌倒、清理成功、运行期源码及输入未改变；这不消除上述质量失败。[完整物理结果汇总](external_sdk_continuation_20261011/artifacts/scene_summary.json)链接逐次质量收据和保留的输入配置。第三层证据指向地图代理、轨迹预测与物理跟踪之间的耦合问题；此为待验证诊断，尚未定位为已修复的单一源码故障。

YOHO 每名行人的 **7,200 个 120Hz 子步**全部保留，最大平面速度均不超过 1.5m/s。脚面诊断覆盖 1,800×4 帧、492 个中性 SMPL 脚面顶点，相对分析支撑面最大穿入约 7.28mm，无低于地面 1cm 的行人帧。该分析不证明实际接触滑移或视觉自然度。[子步速度](external_sdk_continuation_20261011/artifacts/yoho_substep_speed60.json)、[脚面诊断](external_sdk_continuation_20261011/artifacts/yoho_foot60.json)。

YOHO 状态回放渲染两段原场景相机视频，各 450 帧，拼接为 30Hz、30 秒、900 帧；最大关节位置误差约 0.199mm。视频两帧已人工查看，相机输出保留原几何及声明的辅助地面显示。视频是物理状态回放，不算新增物理仿真。[FK 核对](external_sdk_continuation_20261011/artifacts/yoho_video60/yoho_visual_fk_sdk61.json)、[视频收据](external_sdk_continuation_20261011/artifacts/yoho_video60/yoho_video_receipt_sdk61.json)。原始物理数组、视频、权重和完整源码快照保留在本地 `output/external_sdk_continuation_20261011`，未复制为新的大型仓库资源。

## 覆盖清点与剩余范围

coverage 7.15.2 测量当前 `src` 和 `scripts`，保留子进程、`execv` 和 `_exit` 数据，仅合并成功阶段。旧数据中源码行号已变化的两份 PPO 和 SDK 配置模块先剔除，再用当前源码的复测替换；物理质量失败的实验不混入通过阶段覆盖。动态编译虚拟文件无法映射到源文件的警告原样保留，不冒充已覆盖行。

| 执行测量范围 | 语句 | 分支 |
| --- | --- | --- |
| `src`，470 文件 | 42,097 / 55,363，**76.04%** | 10,919 / 17,072，**63.96%** |
| `scripts`，152 文件 | 6,969 / 16,948，**41.12%** | 1,744 / 4,860，**35.88%** |
| 两者合计 | 49,066 / 72,311，**67.85%** | 12,663 / 21,932，**57.74%** |

四个具体文件的语句和分支均为 100%：`external_runtime.py`（208/84）、`serialized_bank.py`（35/18）、`run_external_python.py`（21/10）、`run_project_tests.py`（85/20）。该数字包含受控输入和故障注入，并非全部硬件与输入的保证。[四文件覆盖](external_sdk_continuation_20261011/artifacts/target_coverage_after_existing.json)、[全局汇总](external_sdk_continuation_20261011/artifacts/coverage_summary_final.json)、[逐文件缺失行与分支](external_sdk_continuation_20261011/artifacts/coverage_final.json)。

Git 跟踪及非忽略新增 Python 文件（含历史快照）共 **2,640 文件、645,394 个物理行、25,087 个函数定义**，AST/编译清点无语法失败。这是完整语法清点，不能替代每行或每功能执行。执行测量的 4,676 个函数定义中，仍有 1,085 个没有观测到函数体源行；观测到一行也不能证明整个函数通过。[清点](external_sdk_continuation_20261011/artifacts/all_python_inventory_final_v2.json)、[函数观测](external_sdk_continuation_20261011/artifacts/function_coverage_final.json)。

全仓 Ruff 所选规则留下 12 项发现：两个重复导入、三份历史失败快照的未定义名称，以及四个旧测试文件中的七个 pytest 夹具注册别名。历史失败快照保持原字节。本轮修改源码及新增测试通过所选致命规则 E9/F63/F7/F82。[原始全仓结果](external_sdk_continuation_20261011/artifacts/all_python_ruff.log)、[修改范围结果](external_sdk_continuation_20261011/artifacts/changed_ruff_recovery_verified.log)。

| 未完成范围 | 当前状态 |
| --- | --- |
| 全项目每行、每分支、每功能 100% | 未达到；上述逐文件覆盖和函数记录列出缺失范围 |
| NewMetro / ThirdFloor 物理导航质量 | 未通过；候选均未采用，原始失败与局部改进分开记录 |
| 多 GPU 物理/分布式训练 | 当前单 GPU；用户确认暂时没有测试主机，记录硬件验证缺口 |
| 二进制 Isaac Sim 发行版实机运行 | 本机使用 pip SDK；二进制接口配置不作为实机运行通过 |
| 旧 IsaacGym 后端 | 两项真实跳过，当前 SDK 未提供该后端 |
| NavIsaacLab 历史模型跨分支运行 | 两组归档模型显式拒绝；张量可读不代表策略兼容 |
| RL 长期收敛、完整实验协议 | 短程训练、续训和局部场景验收不能替代这些结果 |
| 资源服务在线授权下载 | 已有资源校验与恢复通过；在线服务授权及完整重新下载仍需独立实测 |

本轮 **152 份证据、19,074,026 字节**的大小和 SHA256 见[清单](external_sdk_continuation_20261011/manifest.json)。失败日志与历史收据保持原样，清点、软件回归、物理验收和硬件缺口分别记账。
