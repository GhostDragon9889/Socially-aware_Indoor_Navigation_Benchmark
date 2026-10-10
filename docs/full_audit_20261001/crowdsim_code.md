# CrowdSim 与轻量 PPO 当前代码审查

审查基准为 2026-10-01 实际工作区。范围包括 CrowdSim 的配置、场景构建、人体模型恢复、SFM/MaskedMimic 导航目标、Drive、PPO 训练/评估/采集、偏好奖励，以及 `src/crowdsim_ppo`。读取了已有报告，但以下结论重新核对当前源码和 `/project/EXP/SangforAndSaut/IsaacLab-Protomotions` 的固定源码。没有修改业务代码、安装依赖或启动 Kit。

当前依赖安装完成后，主要阻塞仍是调用契约与正确性。以下有 6 项 P1、10 项 P2；P1 为主入口中止或关键闭环状态错误，P2 为条件触发的正确性问题。CPU 证据见 [crowdsim_evidence.json](crowdsim_evidence.json)，复现入口见 [reproduce_crowdsim.py](reproduce_crowdsim.py)。张量 articulation fixtures、AST 闭包测试和真实物理仿真严格区分。

## P1：优先处理

### CC01 · 检查点恢复导入不存在的兼容函数

- 位置：`CrowdSim/protomotions_runtime.py:122`，调用 `:132`；当前替代入口 `protomotions/utils/inference_utils.py:18`。
- 触发：所有经过共享 `build_runtime()` 的正常 CrowdSim 入口。
- 证据：在新安装环境实际导入返回 `ImportError: cannot import name 'apply_backward_compatibility_fixes'`。该函数不是缺少安装包，当前源码没有定义。
- 后果：人体 runtime 构建中止，资源齐全也无法越过此处。
- 修复：按当前 `inference_agent.py` 的配置恢复流程接入 `apply_all_inference_overrides`，保留 CrowdSim 的投射物关闭、控制时钟和动作库覆盖；不能用空函数掩盖迁移问题。

### CC02 · 根四元数读写与固定 IsaacLab 格式不一致

- 位置：`CrowdSim/control/drive.py:149`、`:272`、`:363`；`CrowdSim/nav_manager.py:2348`；`CrowdSim/scene_setup.py:686`；`CrowdSim/tools/smpl_mesh_visualizer.py:169`、`:536`。
- 触发：固定后端返回 XYZW 的机器人/人体姿态。
- 证据：已安装源码 `base_articulation_data.py:1327` 明确使用 `QUAT_XYZW_ELEMENT_NAMES`；`base_articulation.py:601` 写姿态也使用 XYZW；旧 `write_root_pose_to_sim():2724` 仅转发，没有 WXYZ 转换。真实 Drive 方法将 XYZW 单位旋转 `[0,0,0,1]` 读成 yaw **180°**；零 yaw 写出 `[1,0,0,0]`，后端所需是 `[0,0,0,1]`。真实 mesh 四元数函数把单位旋转下的 `[1,0,0]` 变为 `[-1,0,0]`。
- 后果：朝向观测、运动控制、spawn、相机/USD 同步及人体蒙皮姿态错误。
- 修复：在后端边界统一格式，覆盖读、写、spawn/reset、USD 和 mesh；不要只改一个 yaw 公式。`test_drive_kinematics.py:13` 的 stub 仍用 WXYZ 单位旋转，现有测试通过无法验证目标后端约定。
- 类型边界：固定源码 `utils/warp/proxy_array.py:209`、`:227`、`:262` 有 Torch 属性、索引和 `__torch_function__` 桥接。不能把 `.clone()`、`.cpu()` 等全部判为类型不兼容。

### CC03 · 行人子集 reset 忽略全局 env ID

- 位置：`CrowdSim/nav_manager.py:899`、`:925`、`:938`、`:957`；未来路径函数 `:2533`。
- 触发：只重置非零、非连续行人 ID，或子集中包含冻结槽位。
- 证据：当前生产方法 AST 加真实 `MaskedMimicContext`；对 `env_ids=[1]`、全局时钟 `[10,20]` 和 0.5s 偏移，输出目标时间 **10.5**，应为 **20.5**；`reached=[True,False]` 时输出 mask **0**，应为 **1**。路径生成器仅置零以隔离时间/mask 索引，未运行物理。
- 后果：reset 的行人读取另一行人的时间、路径、速度和到达状态，冻结槽位也可能被当作活动行人。
- 修复：使用 `ctx.env_ids` 建立局部行到全局 ID 映射，所有活动/冻结分类与导航数组读取使用同一映射。

### CC04 · 评估循环不更新人体观测，也不消费人体 done

- 位置：`CrowdSim/tools/eval_policy.py:360`、`:371`、`:403`；`BaseEnv.get_obs()` 返回观测克隆，`env.py:348`。
- 触发：`run_evaluation()` 连续运行多个 step。
- 证据：循环前 reset 得到 obs，循环内忽略 `env.step()` 返回的新 obs；`done_indices` 仅赋值，没有下一步 reset 消费。当前源码再次确认，未运行 GPU episode。
- 后果：行人策略持续使用 episode 初始身体观测，摔倒/终止后不能正常恢复，人群评估分布失真。
- 修复：更新人体 obs、处理对应 reset，保持机器人指定评估路线不被人体 reset 改写；验收多步身体状态及异步终止恢复。

### CC05 · 专家轨迹采集主入口引用三个未绑定变量

- 位置：`CrowdSim/flow/collect_ppo_buffer.py:861`–`:863`。
- 触发：CLI `main()` 执行 `collect(...)` 参数求值。
- 证据：`repeat_robot_routes`、`target_episodes`、`route_repeats` 只有 `collect()` 形参及内部使用，main 未从已经修改的 `config['collection']` 中读出。别的函数形参不能提供 main 的绑定。
- 后果：`NameError`，采集不能开始；设置相应 CLI/YAML 也无效。
- 修复：从 CLI/YAML 合并后的 collection 配置取得这三个值，验证 main 到 collect 的完整传参。

### CC06 · 训练/采集/评估入口让 pretrained 行人归一化统计继续更新

- 位置：`CrowdSim/protomotions_runtime.py:237`–`:238`；`CrowdSim/ppo/train_ppo.py:384`、`train_ppo_finetune.py:260`、`tools/eval_policy.py:365`、`flow/collect_ppo_buffer.py:367` 的 model 推理。
- 触发：这些入口经 `agent.setup()`、`agent.load()` 后直接 forward，没有 `agent.eval()`。
- 证据：`BaseAgent.setup()` 的 materialization 只临时 eval，随后恢复原 training 状态；load 不切模式。`NormObsBase.forward()` 在 training 下调用 `record_moments()`，实际立即更新均值/方差；当前 MaskedMimic inference 配置含多处 `normalize_obs=True`。真实 CPU NormObsBase 在 `torch.no_grad()` 下将均值从 **0→0.01**，eval 后同样 no_grad 推理不再改变均值。
- 后果：机器人训练、专家采集和评估中的行人控制统计偏离 checkpoint，依赖运行历史并随长时间运行漂移；`no_grad()` 只关闭梯度，不关闭这种状态更新。
- 修复：共享 runtime 在完成恢复后明确设置行人 agent 为 eval；纯推理按官方入口使用 `load_training_state=False`。主展示入口 `crowd_sim.py:125`、`:189` 已调用 `agent.eval()`，本项不适用于该入口之后的推理。
- 边界：未构建完整 MaskedMimic checkpoint 或仿真；状态更新由当前真实归一化模块的 CPU 运行与入口调用链共同确认。

## P2：条件正确性

| ID | 位置与触发 | 当前证据和后果 | 修复方向 |
|---|---|---|---|
| CC07 | `CrowdSim/ppo/ppo_policy.py:857`，`src/crowdsim_ppo/ppo_policy.py:857`；rollout 尾批只有 1 个样本 | 3 样本、batch=2 的真实 CPU 普通 update 产生非有限 value loss，两个副本各 **11 个参数张量**被污染。KL 分支 `:978` 也计算 NaN 方差，但 `:1025` 跳过 NaN 梯度，本例参数保持有限，不能混称污染 | 采用合适的总体/rollout 方差与有限 loss/梯度检查；两个副本同步处理或去重 |
| CC08 | `tools/eval_policy.py:231`、`:340`、`:360`；默认课程开启，先手动传送再 full reset | 真实课程初始化将声明的 **7–10m 改为 2–4m**。实际首次 reset hook AST 在记录写入 fixture 中把手动 XY **[9,9] 覆盖为 spawn [1,2]**；非物理测试。后续传送也绕过 Drive clear/sync 的统一 reset 契约 | 评估固定任务分布，关闭/固定课程；通过统一指定路线 reset 清缓存/同步相机，记录真实起终点 |
| CC09 | `nav_manager.py:1936`、`train_ppo.py:632`，微调 `train_ppo_finetune.py:340` | reached/collision/timeout 独立且可重叠。真实 logger 对一次 reached+timeout 输出 **episodes=2、SR=0.5**；微调同样相加 | 完成数用 done 计数，事件与互斥终止原因分别统计，明确碰撞同时到达的成功定义 |
| CC10 | `world/builder.py:95`、`:100` 在 `empty_mode` 分支 `:121` 之前 | 实际调用 `build_env(empty_mode=True)` 因缺人体 checkpoint 抛 FileNotFoundError，未触发仿真初始化 guard；导出地图前仍要求已存在的地图 | 按模式预检，empty 只需要场景；完整模式检查推理配置与真实资源内容 |
| CC11 | `pref/pref_train_reward.py:102`、`:119`；变长 episode 同 batch | eval 模式真实网络：同一 2 帧轨迹单独评分 **-1.03457**，和 4 帧轨迹同 batch 时 **-1.61079**；真实两帧之和仍 -1.03457。补齐帧被计入奖励 | 返回有效帧 mask 并应用到 attention/求和；验证阶段 eval。限定旧分支，不套用固定长度 train_offline_reward |
| CC12 | `src/crowdsim_ppo/point_goal_env.py:75`、`:85`；目标越地图边界时 clamp | seed42/10000：**1875** 个距离小于设定2m，**1956** 个初始偏角超过15°；最小距离0.099m、最大偏角93.18° | 重采样满足边界与距离的目标，用最终目标方向生成朝向 |
| CC13 | `src/crowdsim_ppo/point_goal_env.py:164`；`num_neighbors=0` | 初始化/observe 支持空邻居，实际第一次 step 的 `.min(dim=-1)` 抛 **IndexError**；默认 num_neighbors>0 不触发 | 空邻居最近距离设为 infinity，并明确配置有效范围 |
| CC14 | `utils/map_metadata.py:76`、`world/builder.py:323`、`world/map.py:282`；ROS map `negate=1` | loader 读出 negate，但 config 丢弃此字段，map 始终 `grid>=threshold` 判 free；2×2黑白地图实际 free为 **[False,True]**，应为 **[True,False]**。另 origin第三分量 yaw 被丢弃 | 保留并应用 negate/原点旋转，或对不支持值显式拒绝；当前默认 negate0/yaw0 不触发 |
| CC15 | `protomotions/simulator/base_simulator/record.py:329`、`:357`；GUI录像且control≠30Hz | BaseSimulator.step `:751` 每个控制步调用 render，record每次采一帧；CrowdSim控制默认25Hz，MP4及.motion仍硬编码30FPS。25Hz采样写30FPS会回放 **1.2倍速度**、动作时间缩短 | 记录实际采样间隔，按控制dt写FPS，或按30Hz重采样；这是源码时序确认，未录制 IsaacLab GUI |
| CC16 | `nav_manager.py:1339`–`:1350`；重新采样/A*规划失败 | Catch RuntimeError后将原合法折线路径改为旧start→goal直线，没有可通行/连通检查；直线可能穿越墙，即使旧路线本来绕障碍合法 | 保留已验证旧路径，或重试有效目标；失败时停止/报告，不把未验证直线当导航参考 |

## 当前用户打开的 record.py

本轮只读，保留其工作区改动。`record.py:313`–`:318` 的 try 和 except 都导入 `moviepy.video.io.ImageSequenceClip.ImageSequenceClip`；对已安装 MoviePy 1.0.3，该路径有效。except 并不提供另一种兼容路径，属于冗余处理，不应据此把安装判为失败。还需将注释中的版本表达式改为准确描述。上表 CC15 的采样时间问题早于这次 import 改动，与安装后 MP4 编码成功是不同验收项。

## 补充风险与验证边界

- PPO 两实现当前字节相同；真实 CPU 缺陷两边都存在。`CrowdSim/ppo_policy.py` 是重导出 shim，不能误算为第三个独立实现。
- `CrowdSim/ppo/ppo_policy.py:1048` 及离线 reward checkpoint 仍直接覆盖目标文件；进程中断可使 latest 不完整。原子保存问题由综合审查跨模块处理，本报告不另计重复发现。
- `world/builder.py:260` 对人体 mesh 初始化失败日志后继续，`nav_manager.py:471` 对每步 mesh 更新错误静默忽略。主循环继续不能证明可见人体与骨架一致；相机输入训练也不能笼统认为外观失败“training unaffected”。
- 主用 `build_offline_preferences` 按 episode UID 分 train/val，再生成各 split 内偏好对；没有把旧变长补齐缺陷扩大到这个固定窗口分支。
- root 执行了本轮全测试套件，本子任务没有重复大套件。新增 CPU 复现使用 Python 3.12.13、Torch 2.11.0+cu128，并记录 CrowdSim、src 与 record.py 源码 SHA256。测试成功/缺陷复现均不证明完整 Isaac Sim physics 已启动。
- 旧报告 F01/F02/F03/F04/F05/F08/F11/F12/F13/F16/F17 的 CrowdSim 部分当前仍有效；安装依赖改变不能消除这些调用契约缺陷。F06 MoviePy 依赖冲突应以本轮安装配置审查为准，不沿用此前“尚未安装”的结论。

复现命令：

```bash
cd /project/EXP/SangforAndSaut/ProtoMotions
PYTHONPATH=src:. /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/full_audit_20261001/reproduce_crowdsim.py
```
