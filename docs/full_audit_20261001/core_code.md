# 核心代码与 FLUX/NavDP 补充审计

审查日期：2026-10-01。当前工作树，未修复业务代码。下列复现使用真实 MotionManager/SLURM 生成器或未修改函数体的 AST 隔离执行；FLUX HTTP 处理复现用内存 PNG 和模型替身，不启动模型或服务。证据为 `core_reproductions.py` / `core_reproductions.json`。P1 表示主要入口或关键状态/安全问题；P2 表示条件触发、结果正确性或交付问题。

## K01 · P1 · FLUX 部署接口混淆图像维度，成功推理后仍返回 500

位置：`FLUX/baselines/flux/deployment_server.py:132,136,144,147,151`。

解码将 H×W 图像 reshape 为 `[1,W,H,C]`，并未保留原来的空间布局。深度结果只有四维，后处理却使用 `depth_np[0,-1,:,:,0]` 的五维索引；RGB 的 `img_np[0,-1]` 也只剩一行。正常推理返回后必然在 144 行抛出 IndexError，被异常处理转换为 HTTP 500。

使用 4×3 内存 PNG、保持处理函数体和异常处理不变，替身 `agent.step_pointgoal` 成功返回：实际 RGB `[1,4,3,3]`、深度 `[1,4,3,1]`，预期分别 `[1,3,4,3]`、`[1,3,4,1]`；返回 `500 / too many indices for array`。这证实接口后处理错误，未验证真实模型初始化及前处理是否还有更早的失败。

建议统一 B/H/W/C 与时间维契约，覆盖非正方形输入和一次完整请求。该问题独立于 FLUX 权重是否存在。

## K02 · P1 · SLURM 将 W&B 凭据写进脚本和日志

位置：`protomotions/train_slurm.py:830,1089,1094,1100`。

凭据被拼进 `export WANDB_API_KEY=...`；完整生成脚本随后打印、保存到 `delete=False` 的临时文件并上传作业脚本。使用替换后的凭据提供者、固定无效假值，真实 `build_job_command` 确实将该值嵌入脚本。本轮没有读取真实凭据、提交作业或对外发送消息。

建议通过作业环境或集群秘密设施传递，生成脚本与日志不包含凭据。

## K03 · P2 · 安装失败仍向其他 SLURM rank 发布就绪

位置：`protomotions/train_slurm.py:770,775,811,813`。

安装命令由分号串联，失败后仍 `touch /tmp/_pip_done_${SLURM_JOB_ID}`。使用独立临时目录中的假 pip（返回 42）运行真实生成的安装门：门退出码为 0，ready marker 存在。没有实际安装包、运行训练或启动 Kit。IsaacLab warmup 的失败标记逻辑正确，但它不能取代前面的安装成功检查。

建议对安装本身检查退出码，并向所有 rank 传播失败状态。

## K04 · P2 · 更新采样权重重新启用不满足长度条件的动作

位置：`protomotions/envs/motion_manager/motion_manager.py:108,438`；调用者 `protomotions/envs/mimic/mimic_evaluator.py:124`。

初始化按 truncate + env_dt 排除短片段；权重更新只重新应用显式 exclude ID，没有重新应用长度条件。真实 MotionManager、两个合成片段长度 `[0.1,2.0]`、truncate=0.5、dt=0.1：权重从 `[0,1]` 更新成 `[0.5,1]`，短片段重新可抽样。恢复权重也没有统一重建资格 mask。

建议持久维护可抽样条件，并在初始化、更新、恢复时统一应用。

## K05 · P2 · 环境多于动作时忽略显式动作 ID 列表

位置：`protomotions/envs/motion_manager/motion_manager.py:124`。

`num_envs > total_motions` 的提前返回发生在显式列表处理之前。真实 MotionManager、2 个动作、3 个环境、合法列表 `[1,1,1]`，`available_motion_ids` 实际为 None。显式配置因此不再控制每个环境的动作。

建议优先验证显式列表，再处理 first/last/random 等自动子集。

## K06 · P2 · latest 检查点原地覆盖且多个文件没有共同代号

位置：`protomotions/agents/base_agent/agent.py:446,484,501`。

环境状态、主 checkpoint 和推理 checkpoint 分别直接写最终路径。现有跨 rank 错误汇总能处理捕获到的异常，但不能保留进程被杀或磁盘写入中断前的完整 latest，也没有保证恢复时各文件属于同一代训练状态。

本项为源代码静态确认，没有执行中断写入实验。建议临时文件原子替换，配套 generation/manifest，保留上一份可恢复状态。

## K07 · P2 · FLUX 旧推理接口和两份 LogoPlanner 绘图函数含未定义名称

位置：`FLUX/baselines/flux/deployment_server.py:58`（yaw）；`FLUX/baselines/flux/policy_network.py:240`（_dev）；`FLUX/baselines/logoplanner/deployment/visualization.py:9,24`、`NavDP/baselines/logoplanner/deployment/visualization.py:9,24`（np）。

保持原函数体的隔离 CPU 调用，四处均复现 NameError。影响必须按调用分支区分：当前 `/eval_dual` 内联计算坐标，不调用坏的 world_to_local；当前 GRPO step_pointgoal 使用 forward_train_pointgoal，不调用坏的基础 predict_pointgoal_action。不能把它们说成当前 GRPO 默认服务一定触发的错误；K01 才是其已证实的入口错误。两份绘图函数则在使用时失败。

Ruff 的另一条 CrowdSim `CrowdNavigationManager` 未定义报告出现在延迟类型注解，不能混算为立即执行 NameError；Python 3.8 target 的括号上下文提示也不是当前 Python 3.12 的语法错误。

## K08 · P2 · 导航 HTTP 调用没有超时

位置：`NavDP/utils_tasks/client_utils.py:12,17,41,72,110`；`FLUX/utils_tasks/client_utils.py:46,54,79,138,178`。

reset/step 的 requests.post 没有 timeout。服务接受连接但不返回时，调用方会在请求处持续等待，训练或评估循环没有显式失败出口。本项为生产调用代码静态确认，没有实际制造服务阻塞。

建议设置连接/读取超时、检查状态码，并按任务契约记录或终止失败 episode。

## K09 · P2 · 当前 Newton CUDA FK 验收失败，根因未定位

位置：`protomotions/tests/test_newton_simulator_fk.py:522`；默认样本 `examples/data/g1_crouch_to_run.motion`。

本轮完整测试实际初始化 Newton/CUDA，并设置、读取单帧 G1 状态。root position、root rotation、DOF position 均通过；body position 平均误差 **0.003096623 m**、最大 **0.010051 m**，超过测试平均误差阈值 **0.001 m**。这说明当前 G1 资产/动作/FK 契约尚未通过现有验收；测试本身也注明曾存在小量差异。

没有证明错误来自 Newton 版本、动作转换或 MJCF，也不能泛化为 SMPL/IsaacLab 的物理结果。应比较对应身体、MJCF 参数及 MotionLib FK，再决定修复或有依据地调整门限。

## K10 · P2 · recording 测试替身与当前导入不匹配

位置：`protomotions/simulator/base_simulator/record.py:313–318`；`protomotions/tests/test_recording_mixin.py:237,291,365`。

当前 try/except 两边完全相同，均导入 MoviePy 内部 ImageSequenceClip。实际安装的 MoviePy 1.0.3 支持此路径，先前两帧视频编码已通过；因此旧报告的 MoviePy 版本冲突已经解决。三项测试仍只替换顶层 `moviepy.ImageSequenceClip`，替身不是 package，当前内部路径导入导致测试失败。

应让导入策略与测试替身一致，再恢复视频及 sidecar 回归测试。不能把这三项 mock 失败解释为实际 MoviePy 编码必然失败。冗余 fallback 本身也不提供第二种兼容路径。

## 覆盖与边界

- 本轮初始全量语法扫描：938 个 Python、14 个 shell 文件，零语法错误。包含 FLUX/NavDP；随后新增的审计脚本单独实际执行。
- ProtoMotions + CrowdSim + 根 tests：1632 passed、4 failed、2 skipped、4 deselected，101.14 秒。分项与完整命令见 `test_summary.json`，失败追踪见 `tests.log` / `tests.xml`。
- CrowdSim 30 项已有测试均通过，但本轮专项复现仍发现关键行为缺陷；“测试通过”不足以证明未覆盖的完整调用链正确。
- 没有运行 FLUX/NavDP 全部独立测试、Isaac Sim Kit、长时间行人 rollout 或真实机器人部署。
