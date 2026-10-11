# 全 Python 源码清单与可选基线审查

本报告针对 `src`、`scripts`、`third_party`、`examples`。逐文件读取、计算 SHA256、解析语法、收集所有定义 / 导入与重复定义候选；同时检查目标 SDK 导入和具体断点。机器清单见同目录 `all_python_*.json`。这种逐文件扫描不等于执行所有分支，也不保证无 bug。

## 扫描证据

最终扫描共 1,514 个 Python 文件 / 331,368 行，14,364 个定义、11,268 个导入语句；无 AST 解析失败。逐模块数字及SHA256见最新 `all_python_summary.json` / `all_python_inventory.json`；扫描是工作区当时快照。

Ruff 关键规则覆盖 F821/F822/F823/F405/F404/F402/F901/E9，显式禁用 gitignore 过滤。初始 20 个 F821、2 个 F822、1 个 F901 已逐项定位修正；最终 `ruff_critical_after.json` 不再包含这些规则问题。剩余 597 个 F405 为 `import *` 带来的未知符号候选，不能自动分类为运行错误。2 个 F402 为 PACER 局部循环变量 `mesh` 遮蔽模块名，所在函数未随后调用该模块，未发现实际错误，因此保留。

10 个重复定义候选均已查看：属性 getter/setter 属正常同名；Trace 环境 get_info 和 PACER 四元数函数为一致重复定义；FLUX 最末配置定义是源码明确注释的有意覆盖。未将它们直接算作功能缺陷。

## 功能模块与验证边界

| 范围 | 审查与本轮验证 | 尚未动态验证 / 限制 |
|---|---|---|
| src/protomotions | 全文件 AST / SDK API 清单；Newton 状态、PD、接触聚合、关闭回归 | 所有训练 epoch、所有机器人与多后端 GPU 组合并未穷尽 |
| src/CrowdSim、crowdsim_ppo、pedestrian等主项目 | 本子任务全文件静态覆盖，root / 其他子任务执行环境及主物理集成 | 端到端成绩应引用其他任务实际结果，不能由静态通过推导 |
| scripts | 全文件解析与导入清单 | 下载、转换、批训练、发布脚本的外部服务 / 全 CLI 参数未全部执行 |
| examples | 全文件解析；FSQ 返回类型解析修正并 CPU 调用 get_type_hints | 每个实验的完整训练和 simulator 运行未执行 |
| third_party/flux | 修正 pointgoal CPU 设备变量、坐标转换变量、Logoplanner 绘图缺 import；确定边界的 CPU 方法测试 | DINO / DepthAnything 权重加载、CFM 训练、完整实时基线未执行；独立 requirements 不应覆盖主 runtime |
| third_party/navdp | 69条旧导入已迁移，69/69目标源码路径可解析；Light/限幅/启动诊断CPU回归通过，矩阵与边界见NAVDP_API_MIGRATION_ZH.md | 原始冻结 requirements 指向 Lab 2.0.2 / Warp 1.6.1 / NumPy 1.26，不能用于目标3.0EA /6.1环境；全导航评估未执行 |
| third_party/pacer | 修正真实 NameError / TypeError 分支，mean KL 调度、flags、SMPL thread 异常、四元数移除 yaw 等 CPU组件回归 | 原始 IsaacGym 二进制及 rl-games==1.1.4 训练环境未导入，不宣称迁移整套 PACER 训练；缺失 attention 架构显式拒绝 |
| third_party/trace | 全静态源码与重复定义审查 | 原始 NumPy1.21/Lightning1.6/pyarrow7 等冻结依赖不支持当前 CPython3.12/NumPy2组合，原始训练需独立环境 |
| 其他第三方纯工具 | 全静态清单 | 平台工具、外部模型、输入数据依赖按用途保留，未全部实际执行 |

## 已落地的具体可选基线修正

1. FLUX `world_to_local()` 使用未定义 yaw，修为实际参数 ryaw；四个角度逆变换回归。
2. FLUX pointgoal `_dev` 未定义，修为实例 `self.device`，实际 CPU Tensor 推理方法覆盖轨迹生成和 critic 选择（编码器 / 模型调用边界注入）。
3. FLUX / NavDP Logoplanner 绘图缺 NumPy 导入，正常导入模块并生成 PNG。
4. PACER common_agent 缺 nn / optim 引入；standard_epoch 更新引用不存在 kls，修为已有聚合 av_kls，实际方法测试。
5. PACER network_builder 缺 layers 导入、SAC MLP 初始化引用不存在 cnn_init，使用相同源码已定义的 mlp_init。
6. PACER ResidualBlock 请求不存在的 ChannelAttention / SpatialAttention 时明确抛出 NotImplementedError，默认无 attention 架构不变；不伪造作者权重对应结构。
7. PACER run 缺 wrappers 引入；flags.__all__ 去除不存在 summation；SMPL thread `except e` 修为 Exception；补充单四元数 de_heading；`raise NotImplemented` 修为 NotImplementedError。
8. FSQ 配置类返回类型移至可解析的顶层符号，保留既有配置构造行为。
9. 三处 light 无限创建循环、单体/向量prim误用、批量差速控制忽略速度上限、PACER移除np.int相关路径均修正并有CPU回归；详情见NavDP报告。

`vendor_cpu.xml` 早一轮 **15 passed**；最终 `vendor_navdp_cpu.xml` **29 passed**，含光源、SDK前置诊断、轨迹、差速限幅；`final_newton_vendor_cpu.xml` Newton邻近+vendor共 **64 passed, 2 skipped**。测试文件 `src/protomotions/tests/test_optional_vendor_regressions.py` 在源码中明确标注：纯模块正常导入；旧 simulator 耦合定义使用 AST 提取并提供外部边界。它们是组件测试，不是 FLUX / NavDP / PACER 端到端 GPU 运行证据。

## 需要继续独立验证的 API

- NavDP 完整 AppLauncher、camera / RTX 输出、Dingo USD articulation、RslRlVecEnvWrapper 以及 reward / reset 的实际批形状。
- PACER 原始 IsaacGym state tensor、viewer、GPU PhysX、rl_games 1.1.4 网络注册和 checkpoint tensor 兼容。
- FLUX 原始 vision backbone 权重 / transform 约定、CFM 轨迹采样与真实图像观测。
- Trace 原始训练 scheduler / Lightning API / 数据缓存。主项目迁入的 Diffusion 代码应单独使用项目 tests / 评估结果验证。
- Newton CUDA graph、MujocoWarp GPU 联系和 GUI 视频；已在 `NEWTON_MIGRATION_ZH.md` 给出明确后端限制。

所有此处修正均留在工作区，未提交或推送。第三方归档依赖文件未合并到主环境，避免旧版本 pins 污染 Isaac Sim 6.1 runtime。


## 本轮完整静态范围与最终收据

初始扫描1512文件/330820行，后续新增兼容及回归源文件后文件数1514；最终工作区快照数字见 `all_python_summary.json` 和逐文件SHA清单。此处四个scope所有Python源文件均纳入（包含gitignored的third_party），无AST失败；并非所有非Python资源/二进制已被此扫描器执行。

NavDP进一步的真实API顺序/数据桥修正、N>1 reset和停滞判断见 `NAVDP_API_MIGRATION_ZH.md`。最新vendor回归 `vendor_navdp_quaternion_cpu.xml` 为36passed；实际installed SDK CPU utility/Proxy验证见 `navdp_live_sdk_cpu.json`。先前combined64passed2skipped收据仍保留作阶段证据。root另统一执行GPU物理/CUDA/录像，其结果不能混作本子任务CPU收据。

`agent_changed_source_files.json` 给出本子任务所有源码修正相对路径和当前SHA；`artifact_copy_manifest.json` 给出审计报告/机器清单/CPU XML/GDB日志和辅助脚本的复制清单（排除cache及重复staging）。所有修正与报告保留在本地；未提交、推送。


---

打包说明：本页保留审查当时的阶段结论。最新证据与复制范围见 [审计索引](README.md)；未随包复制的大清单、源码快照与完整日志见 [来源索引](SOURCE_INDEX.md)。项目源码位置以仓库相对路径表述；SDK 绝对路径仅记录被审查机器。
