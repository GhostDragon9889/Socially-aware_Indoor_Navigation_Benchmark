# 仓库结构与代码审查

> 2026-10-01：最新代码复核见 [全面审查报告](CODE_REVIEW_20261001_ZH.md)，新增人体子集 reset、四元数和旧偏好奖励实证，并区分普通 PPO 与带 KL 更新的数值保护。以下保留此前审查记录。

审查日期：2026-09-29。基准提交：`7a8417a`。审查对象包含当前工作区中尚未被 Git 跟踪的 CrowdSim、FLUX、NavDP、src、tests 和配置文件。

本次进行了目录与入口梳理、922 个 Python 文件的语法扫描、14 个 Shell 文件的语法检查、关键训练/采集/评估/恢复调用链阅读、现有测试和 CPU 最小复现。**这不代表 922 个文件都经过逐行语义证明，也不代表五个仿真后端已完成运行验收。** 外部模型、场景及仿真依赖限制了端到端验证。

只新增本报告及 `REPOSITORY_SCAN.csv`，没有修复或改动业务代码。清单逐文件记录路径、行数、顶层类/函数和语法检查结果；语法通过不等于逻辑正确。

## 项目结构

```text
ProtoMotions/
├── pyproject.toml                 # protomotions 包、依赖、CLI、打包规则
├── requirements_*.txt             # 仿真环境依赖
├── FULL_PIPELINE.md               # 新导航流水线说明；存在与实现不一致的内容
├── protomotions/
│   ├── train_agent.py             # 配置构建、训练、恢复入口
│   ├── inference_agent.py         # 策略加载与仿真推理
│   ├── cli.py                    # 安装后的命令行入口
│   ├── train_slurm.py             # SLURM 上传、安装、调度、监控
│   ├── train_lepton.py            # Lepton 作业与代码快照
│   ├── _train_launch_common.py    # 启动参数和共享配置
│   ├── agents/
│   │   ├── base_agent/            # 训练生命周期、分布式、检查点
│   │   ├── ppo/ amp/ ase/         # 强化学习算法
│   │   ├── mimic/ supervised/     # 动作跟踪与监督学习
│   │   ├── fine_tuning/ peft/     # 微调与参数高效训练
│   │   ├── common/ optimizer/     # 网络模块与优化器
│   │   └── evaluators/ callbacks/ utils/
│   ├── envs/
│   │   ├── base_env/              # 组件化环境、step/reset、观测上下文
│   │   ├── control/               # 控制目标与状态组件
│   │   ├── obs/ rewards/ terminations/
│   │   ├── motion_manager/        # 动作片段采样、子集、权重和时间
│   │   └── context_paths.py / context_views.py
│   ├── simulator/
│   │   ├── base_simulator/        # 后端公共接口
│   │   └── isaacgym/ isaaclab/ newton/ genesis/ mujoco/
│   ├── components/               # motion_lib、pose_lib、scene_lib、地形
│   ├── robot_configs/            # G1、H1_2、SMPL/SMPL-X、SOMA23 等
│   ├── utils/                    # 组件构建、配置、旋转、几何、导出
│   ├── data/assets/              # MJCF、USD、网格与材质
│   └── tests/                    # 125 个 test_*.py 文件
├── CrowdSim/                     # Isaac Lab 人群导航与机器人 PPO
│   ├── crowd_sim.py              # 人群/机器人联合运行入口
│   ├── protomotions_runtime.py   # 对接 ProtoMotions 检查点与仿真
│   ├── scene_setup.py            # 机器人、场景、相机和可视化装配
│   ├── nav_manager.py            # 路径、邻居、奖励、终止、reset；2665 行
│   ├── world/                    # builder、地图、A*、社会力模型
│   ├── control/drive.py          # 差速/运动学控制、碰撞保护、相机同步
│   ├── ppo/                      # 策略、课程、训练、奖励模型微调
│   ├── flow/                     # 专家轨迹采集、恢复、路线控制
│   ├── pref/                     # 偏好对构建、教师评分、奖励模型训练
│   ├── tools/                    # 策略评估、渲染和诊断工具
│   ├── utils/                    # 配置、地图元数据和传感器记录
│   ├── config/                   # 场景、环境、采集、奖励模型配置
│   ├── humanoid/                 # 行人避障诊断
│   └── tests/
├── src/crowdsim_ppo/              # 纯 PyTorch 二维 PointGoal 参考环境
│   ├── point_goal_env.py         # 向量化环境
│   ├── ppo_policy.py             # 与 CrowdSim 策略实现重复
│   ├── goal_curriculum.py        # 课程学习
│   ├── pipeline.py              # 配置与对象构建
│   └── train.py / evaluate.py
├── configs/point_goal.json
├── tests/                        # 独立参考环境的测试
├── FLUX/
│   ├── baselines/flux/           # Flow 策略、GRPO Agent、HTTP 服务
│   ├── baselines/navdp/          # NavDP 兼容服务及客户端
│   ├── baselines/                # GNM、ViNT、NoMaD、iPlanner 等基线
│   ├── eval_* / teleop_*         # 仿真评估和遥操作
│   ├── socialnav_metrics.py      # 人机距离和侵入时长指标
│   └── configs/ utils_tasks/ wheeled_robots/
├── NavDP/
│   ├── baselines/navdp/          # 扩散策略及 HTTP 推理服务
│   ├── baselines/logoplanner/    # LoGoPlanner 与部署控制
│   ├── baselines/                # 其他导航基线及重复的深度模型代码
│   ├── eval_* / teleop_*
│   └── configs/ utils_tasks/ wheeled_robots/
├── examples/experiments/          # Python 实验配置
├── examples/tutorial/            # 分阶段教程
├── data/scripts/                 # 动作转换、过滤、打包与接触标注
├── pyroki/                       # G1/H1_2 动作重定向
├── usd_convert/                  # MJCF 展平、转换与视觉网格修补
├── deployment/                   # ONNX 导出、输入协议、MuJoCo 验证
├── scripts/                      # 集群循环和机器人辅助脚本
├── docs/                         # 文档与本次审查产物
└── .github/workflows/             # 上游公共 CI
```

实际目录以逐文件清单为准；树中合并展示了同职责目录。

| 扫描范围 | Python 文件数 | 检查重点 |
|---|---:|---|
| protomotions | 351 | 训练生命周期、动作采样、检查点、集群启动、配置和后端接口 |
| CrowdSim | 59 | PPO、环境反馈、评估、轨迹采集、偏好奖励与场景装配 |
| src / tests | 7 / 3 | 独立环境、训练恢复和课程采样 |
| FLUX / NavDP | 201 / 210 | 推理服务、控制、评估、训练回报对应关系；含第三方深度模型 |
| deployment / examples | 7 / 47 | 导出输入协议、实验和教程入口 |
| data/scripts / pyroki | 26 / 2 | 转换、采样率、重定向接口 |
| usd_convert / scripts | 5 / 4 | 资产转换与启动辅助 |

主要调用链：

```text
ProtoMotions:
CLI → train_agent → 实验配置 → 组件构建 → BaseEnv + Simulator → Agent → checkpoint

CrowdSim:
YAML → world.builder → ProtoMotions runtime + 场景/传感器
     → CrowdNavigationManager → PPO/课程 → checkpoint
     → flow 采集 → 偏好数据 → 奖励模型 → PPO 微调

独立参考环境:
JSON → PointGoalBatchEnv + RobotPPOTrainer → rollout → GAE → PPO update

FLUX / NavDP:
仿真 RGB/Depth/Goal → HTTP 客户端 → 策略服务 → 候选轨迹/评分
                 → 世界坐标转换 → MPC/差速控制 → 仿真和指标
```

FLUX/NavDP 是工作区内的独立子项目，不能从目录存在推断它们已接入根包的安装、训练数据消费或 CI。

## 已确认问题

优先级：P1 应先处理，会阻断主要入口、使评估失真或暴露凭据；P2 为条件触发的正确性、可靠性或交付问题；P3 为局部工具错误。每项同时标明验证方式，未运行仿真的结论不冒充仿真结果。

### R01 · P1 · 专家采集 main 使用三个未定义变量

位置：`CrowdSim/flow/collect_ppo_buffer.py:783`、`:861`。

`main()` 读取 collection 配置时没有赋值 `repeat_robot_routes`、`target_episodes`、`route_repeats`，但调用 `collect()` 时直接使用。它们是另一个函数的参数，不是当前作用域的变量。配置和 CLI 都无法补上 Python 局部变量。

验证：隔离仿真依赖执行原样 `main()` 函数体、关闭 W&B，得到 `NameError: name 'repeat_robot_routes' is not defined`；尚未进入 collect。修复应从最终合并后的 collection 配置读取三个值，并增加 CLI 入口测试。

### R02 · P1 · CrowdSim 评估循环重复使用初始行人观测

位置：`CrowdSim/tools/eval_policy.py:360`、`:371`、`:403`；参照 `protomotions/envs/base_env/env.py:346`。

评估只在循环前取得一次 `obs`，随后丢弃 `env.step()` 返回的新观测。`BaseEnv.get_obs()` 克隆动态观测，因此旧字典不会自动追踪新状态。人形策略持续根据旧状态产生动作。行人 done 的索引又只被赋给局部 `done_indices`，没有用于下一轮 reset。

影响：人群运动与正常运行路径不同，避障结果和成功率不能作为可靠评估。应接收 step 的最新观测，并按正常训练/运行流程处理行人 reset。验证方式：完整追踪生产调用链，尚未运行 Isaac Lab 对比实验。

### R03 · P1 · SLURM 把 W&B API key 输出并保存进作业脚本

位置：`protomotions/train_slurm.py:827`、`:830`、`:1089`、`:1094`。

远端没有凭据而使用本机/交互输入 key 时，真实 key 被直接拼入 `job_cmd`。完整脚本随后打印到终端、写入 `delete=False` 的临时文件，并上传为 `submit.sh`。凭据会进入日志及持久脚本。

验证：使用假 key 构造命令和脚本，两个字符串均包含假 key；没有读取或输出真实凭据。应通过受控环境或秘密存储传递，打印时脱敏，并清理临时文件；避免在可持久传播的脚本里嵌入 key。

### R04 · P2 · 单样本 minibatch 会产生 NaN 并污染 PPO 参数

位置：`CrowdSim/ppo/ppo_policy.py:857`、`:978`；相同代码见 `src/crowdsim_ppo/ppo_policy.py`。

`returns.var()` 默认使用样本方差。minibatch 只有一个元素时返回 NaN，随后加入 `1e-8` 不能消除 NaN。batch 生成器保留尾批，所以既可能由 minibatch_size=1 触发，也可能由总样本数除 minibatch_size 余 1 触发。

验证：3 个 rollout 样本、minibatch_size=2、1 个 PPO epoch，`value_loss=nan`，11 个参数张量含非有限值。两份策略文件在审查时完全相同。应采用明确的总体方差或 rollout 级稳定尺度，并对非有限 loss/梯度拒绝更新；同时覆盖 update 和 update_with_kl。

### R05 · P2 · 评估后的权重更新重新启用过短动作

位置：`protomotions/envs/motion_manager/motion_manager.py:108`、`:438`；`protomotions/agents/evaluators/mimic_evaluator.py:124`。

初始化根据 `sample_time_truncate_s + env_dt` 将过短动作权重置零，但 evaluator 的权重下限会把零重新变成正数。`update_sampling_weights()` 只重新应用显式 exclude IDs，没有保留长度资格约束。

验证：原样 MotionManager 类体的隔离 CPU 复现中，权重 `[0,1]` 经默认形式的 floor/update 变为 `[0.5,1]`。应持久保留可采样 mask，在初始化、恢复和每次更新时统一应用。

### R06 · P2 · 环境数多于动作数时显式动作列表被忽略

位置：`protomotions/envs/motion_manager/motion_manager.py:123`、`:150`。

`num_envs > total_motions` 时函数提前返回，显式列表处理逻辑根本不会执行。配置支持逐环境列表且没有要求 ID 唯一，因此重复 ID 的合法列表也受影响。

验证：2 个动作、3 个环境、`subset_method=[1,1,1]`，得到 `available_motion_ids=None`，后续走随机采样。应先处理显式列表，仅对自动子集模式使用环境数判断。

### R07 · P2 · 独立环境裁剪目标后破坏课程距离与朝向

位置：`src/crowdsim_ppo/point_goal_env.py:75`、`:85`。

采样目标后按地图边界 clamp，真实距离会缩短，真实方位也会改变；heading 仍根据裁剪前的 angle 生成。因此配置中的距离和朝向约束不是实际 reset 分布。

验证：seed=42、10000 个环境、距离 2–4 米、朝向偏差 0–15 度：1875 个距离小于 2 米，1956 个角度超过 15 度；最短约 0.099 米，最大偏角约 93.18 度。应采用满足边界和距离约束的采样，并以最终目标方向生成 heading。

### R08 · P2 · CrowdSim 默认评估仍启用训练课程

位置：`CrowdSim/tools/eval_policy.py:229` 至 build_env 调用；`CrowdSim/nav_manager.py:140`；`CrowdSim/ppo/goal_curriculum.py:119`。

评估配置直接送入 builder，没有关闭 `navigation.rl.goal_curriculum.enabled`。默认 YAML 开启课程且初始 stage=0；manager 构建时即修改导航距离。报告中的 7–10 米任务实际可能变为 2–4 米。FULL_PIPELINE.md 所说“关闭课程”没有对应实现。

验证：使用实际 YAML 和实际课程类，7/10 的导航距离被改为 2/4。应在评估构建前显式固定距离、朝向和课程开关，并将最终解析配置写入结果。

### R09 · P2 · 网格评估首次 reset 覆盖已选机器人起点

位置：`CrowdSim/tools/eval_policy.py:340`、`:360`；`CrowdSim/nav_manager.py:397` 至初次 reset 分支。

脚本先将机器人传送到网格起点，再调用 `env.reset(None)`。manager 的第一次完整 reset 会重新写入 `_crowdsim_robot_spawn_poses`。因此首次 episode 不保证从所选网格点开始。另外，手工 teleport 没有走正常机器人 reset 的状态清理与相机 prim 同步流程。

验证：调用顺序与 reset hook 静态追踪。应先完成环境初始化，再通过统一的指定路线 reset 接口设置起点、朝向、目标、控制缓存及传感器同步；验收时比较首帧真实位置与选定起点。

### R10 · P2 · NavDP 多环境评估的状态尺寸和 MPC 归属错误

位置：`NavDP/eval_pointgoal_wheeled.py:200`、`:203`、`:93`、`:227`。

`--num_envs` 允许大于 1，但只读取机器人 0 的速度，形成长度 1 的数组，与长度 N 的相机位置一起 stack。规划线程还在循环中反复覆盖单一全局 `mpc`，控制循环随后对所有环境使用最后一个环境的控制器。

验证：N=2 的状态组装得到 `ValueError: all input arrays must have the same shape`。MPC 问题由赋值和调用链确认，修好尺寸后仍需处理。应逐环境计算速度、保存 MPC，并在锁保护下发布与轨迹对应的完整快照。

### R11 · P2 · FLUX 停止判断跨环境混用评分

位置：`FLUX/baselines/flux/server.py:228`、`:273`、`:315`。

使用全 batch 的 `all_values_np.max()` 判断是否停止。两个环境分别全部低于和高于阈值时，前者也不会停止。fallback 转向符号同样跨 batch 求均值。

验证方式：静态数据维度追踪；例如阈值 -3、两个环境最佳评分 -10 与 0，全局判断为 false。应按 `axis=1` 计算停止 mask，分别处理每个环境。NavDP 的部分 agent 方法也使用全局 max，需要同类检查。

### R12 · P2 · FLUX 训练时把已执行回报关联到重新采样的轨迹

位置：`FLUX/baselines/flux/policy_agent.py:437`、`:553`、`:632`、`:643`；`policy_network.py:474`。

rollout 只存观测、reward 和 done，没有保留实际执行轨迹/候选 ID/采样噪声。更新时重新随机生成候选，将其中最大 critic score 回归到先前执行动作得到的回报。训练目标因此不对应实际发生的动作；候选排序损失又以自身预测评分构造相对优势。

验证方式：静态追踪 buffer 字段与随机采样路径，未运行 GRPO 收敛实验。应保存实际执行候选或可重放噪声，将监督目标绑定到该动作；若训练候选选择策略，则需要与执行选择对应的概率及可验证的组内回报。

### R13 · P2 · 多处 latest 检查点直接覆盖

位置：`protomotions/agents/base_agent/agent.py:484`、`:501`；`CrowdSim/ppo/ppo_policy.py:1048`；独立 PPO 同副本。

直接 `torch.save(..., latest_path)` 会在写入完成前破坏旧文件。磁盘满或进程被终止时，自动恢复可能发现一个存在但不完整的 last/latest 检查点。核心环境状态和 agent 状态还分开覆盖，可能来自不同保存轮次。

验证方式：保存与恢复路径静态追踪，未进行破坏性磁盘故障试验。应写同目录临时文件、完成后原子替换；多文件检查点应使用版本目录或清单保证代际一致。采集器已有 `_atomic_torch_save()`，但训练器尚未采用同等保护。

### R14 · P2 · SLURM 安装失败仍发布完成标记

位置：`protomotions/train_slurm.py:764`、`:811` 至 `:815`。

安装命令由分号连接，失败后仍执行 `touch /tmp/_pip_done_${SLURM_JOB_ID}`。其他 rank 看到 ready 标记后继续启动，而不是收到安装失败。周围逻辑没有为该 pip 命令检查返回码。

验证方式：生成命令的控制流检查。应仅在安装成功时发布 ready，失败写入独立状态并返回原始错误码；所有 rank 检查同一个失败状态。

### R15 · P2 · 完整流水线文档与当前安装交付不一致

位置：`FULL_PIPELINE.md` 的 Environment 段；`pyproject.toml` 的 optional-dependencies、scripts 和 packages.find；`requirements_isaaclab.txt`。

文档指向不存在的 `docs/ISAAC_SIM6_LAB3_MIGRATION_ZH.md`，并要求执行不存在的 `CrowdSim.tools.check_isaaclab3` / `validate_isaaclab3`。声称 requirements 应用 `constraints_isaaclab3.txt`，但该文件不存在，requirements 也没有相应 `-c`。声称 `pip install -e ".[test]"` 安装独立 crowdsim-ppo，但根项目没有 test extra，打包规则只包含 protomotions。

验证：文件存在性与打包配置逐项核对。应选择统一的安装入口和受支持版本基准，补齐真实文件或修正文档；独立 src 包需明确安装规则/入口，或明确使用 `PYTHONPATH=src`。本报告补齐了原先缺失的审查文档链接，其他不一致仍未修复。

### R16 · P2 · empty 场景模式仍依赖人形检查点和动作文件

位置：`CrowdSim/world/builder.py:95`、`:100`、`:121`、`:300`。

`empty_mode` 分支在通用 `_validate()` 之后；后者无条件检查 checkpoint 和 motion_file。用户仅想打开仓库场景生成地图时，仍会因为人形资源缺失而被阻断，与 FULL_PIPELINE.md 的说明相反。

验证方式：入口分支和前置验证顺序。应按模式建立所需资源列表；正常模式还应检查 `resolved_configs_inference.pt`，以及路径实际内容是否为未解析 LFS pointer。

### R17 · P2 · 导航请求没有超时，规划失败后保留旧轨迹

位置：`NavDP/utils_tasks/client_utils.py:72`；`NavDP/eval_pointgoal_wheeled.py:79`、`:125`、`:208`。

HTTP post 没有连接/读取超时；服务接受连接但不返回时，规划线程可能永久等待。异常分支只记录错误，不清除已有轨迹；控制循环没有检查轨迹生成时间或 planning_error，可能一直跟随旧计划。

验证方式：HTTP 参数与线程消费者静态追踪，未启动真实服务。应设置超时、检查 HTTP 状态，并附带 episode/generation ID 与轨迹时效；失败或过期时向仿真发送明确停止动作。

### R18 · P2/P3 · 基础推理与绘图工具存在未定义名称

位置与影响：

- `FLUX/baselines/flux/policy_network.py:240`：`predict_pointgoal_action()` 的 `_dev` 没有定义，直接调用立即 NameError。注意：当前 `flux/server.py` 使用 `forward_train_pointgoal()`，因此不能说当前服务器的每次 pointgoal 请求都会触发此问题。
- `FLUX/baselines/flux/deployment_server.py:58`：`world_to_local()` 使用 `yaw`，应使用函数局部 `ryaw`。当前请求处理器另有内联坐标转换，影响限定在调用这个工具函数的路径。
- `FLUX/baselines/logoplanner/deployment/visualization.py:24` 与 NavDP 同路径：使用 `np.arange` 等但没有导入 numpy，调用绘图函数会失败。

前两项已通过原样函数体隔离执行复现 NameError；绘图问题由静态检查确认。应补齐变量/导入，并测试这些对外方法，避免只测服务器当前绕过的路径。

## 维护、性能与覆盖问题

以下与上述可定位错误分开，不把所有代码风格问题当成运行缺陷：

- CrowdSim 与 src 的 `ppo_policy.py` 在审查时完全相同，单份 1087 行；已确认的 NaN 问题必须修两处，后续容易漂移。建议抽成共同的轻依赖库。
- `nav_manager.py` 集中地图、采样、社会力、控制、观测、奖励、reset、日志及 monkey patch，状态耦合明显。优先在保持行为测试的前提下拆出 episode 生命周期和观测/奖励组件。
- `train_lepton.py:389` 维护自己的 rsync 排除表，没有使用 `_train_launch_common.py` 的共享表，遗漏 .worktrees、虚拟环境和本地工具状态等排除项。相关目录存在时会被上传；应统一配置并增加 staging 清单预览。
- CrowdSim 的 reached、collision、timeout 可能同一步同时为真，但 logger 使用各类事件数之和作为 episode 分母；课程又直接把 reached 当成功。建议明确互斥终止原因与成功定义，保留独立事件计数，避免统计口径不一致。
- 新增目录处于未跟踪状态；它们尚不属于当前提交。分享当前 commit 无法复现这个工作区。
- 上游 CI 的测试入口仅为 `protomotions/tests/`，未覆盖 `tests/` 和 `CrowdSim/tests/`；所有 job 限定 `NVlabs/ProtoMotions`，在自己的 fork 默认跳过。lint 会扫描提交中的变动 Python 文件，不能说 lint 永远排除了新目录。
- CrowdSim/src/tests 的现有测试通过，没有覆盖 collector main、实际评估循环、单样本尾批和采样分布等已发现缺口。FLUX 的 `test_flask_server.py` 是需要运行服务的手动探测脚本，不能当作已通过的自动单元测试。
- 资产目录中仍有 49 个 Git LFS pointer（36 个 .usd、13 个 .usda）。只能确认元数据，不能判断对应二进制资产的关节、碰撞、材质和渲染是否正确。
- 配置化后端涉及不同运行环境；本报告只核查本地代码与配置，没有在线核验外部软件当前发行版本。

## 验证记录

| 项目 | 结果 | 解释 |
|---|---|---|
| Python AST 扫描 | 922 / 922 可解析 | Python 3.12；不证明兼容声明中的所有 Python 版本 |
| Shell `bash -n` | 14 / 14 通过 | 不证明集群运行成功 |
| `tests/` + `CrowdSim/tests/` | 38 passed，2 warnings | CPU 执行；warnings 为不完整测试对象析构时缺少 `_trajectory_timestamp_file` |
| 核心完整收集尝试 | 325 项收集，86 个模块收集错误 | 当前环境缺少 lightning、tensordict、dm_control、omegaconf 等 |
| 可收集的 39 个核心模块 | 297 passed，24 failed，4 skipped | 24 个失败均为缺依赖：yaml、typer、dm_control、trimesh；没有据此宣称业务断言失败 |
| 另一现有环境中的 launcher 子集 | 69 passed | 与上述部分测试重叠，不累计成“额外 69 项” |
| 独立 PPO 闭环 | 通过 | CPU、4 环境：训练到 64 步，保存后恢复到 96 步，再评估 4 episodes；只证明入口连通，不证明策略质量 |
| 最小复现 | collector、NaN、动作权重、动作子集、reset 分布、评估距离、NavDP shape、FLUX 工具 NameError | 使用真实轻量模块或未修改的 AST 类/函数体隔离重依赖 |
| Isaac Lab / Gym / Newton / Genesis / MuJoCo 全流程 | 未完成 | 未安装/补齐完整依赖及外部数据，也未启动长时间仿真训练 |
| GPU 性能、真实渲染、ONNX 部署一致性 | 未验证 | 不能以静态审查替代 |

本次 PyTorch 验证使用已有环境 `/home/lfm/anaconda3/envs/openAIbase/bin/python`；未安装或升级用户依赖。禁用无关 pytest 自动插件，避免环境自带插件干扰。

复现测试命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. \
  /home/lfm/anaconda3/envs/openAIbase/bin/python -m pytest \
  --import-mode=importlib -q tests CrowdSim/tests
```

核心测试、收集日志和最小训练输出保存在本机 `/tmp` 中；临时文件不是仓库交付依赖。

## 建议修复顺序

1. 修复采集入口、评估观测/reset、W&B 凭据输出，保证主要流程与结果可信。
2. 修复 PPO 数值边界、动作采样约束和评估课程，增加直接覆盖这些故障的回归测试。
3. 统一检查点原子保存、安装失败传播、HTTP 超时与轨迹代际管理。
4. 修正安装文档、独立包交付和 CI 范围，再用完整仿真环境执行短训练、恢复、采集、奖励训练、微调及评估闭环。

不能据当前测试结果认定整仓已经可用于可靠的导航 benchmark 或长时间无人值守训练。
