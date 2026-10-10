# ProtoMotions / NavIsaacLab 全面修正与验收

日期：2026-10-01。实施仓库：`/project/EXP/SangforAndSaut/ProtoMotions`。
参考仓库：`/project/EXP/NavIsaaclab2.0` 及其 `protomotions/`，全程只读。

本轮在上一轮 Isaac Sim 6 / Isaac Lab 3 行人迁移之上，实际修改了训练、保存恢复、
后端状态、动作和场景数据、录制、独立安装与 CI。旧审计报告中的失败数字、资源状态
和 hash 是历史快照；本报告及 `completion_20261001/` 是本轮结果。

## 1. 审查范围与项目结构

```text
ProtoMotions/
├── protomotions/
│   ├── agents/                 PPO、AMP/ASE、MaskedMimic、GPC 等模型/训练
│   ├── envs/                   reset、奖励、观测、动作采样与任务
│   ├── components/             MotionLib、PoseLib、SceneLib、地形
│   ├── simulator/              IsaacLab PhysX、Newton、MuJoCo 等后端
│   ├── robot_configs/          SMPL、G1 等骨架与控制参数
│   ├── utils/                  配置、导出、原子 checkpoint 等
│   └── tests/                  核心和后端测试
├── CrowdSim/
│   ├── world/                  环境构建、占据地图、规划与 SFM
│   ├── humanoid/simulate.py     独立真实行人运行入口
│   ├── protomotions_runtime.py  冻结 MaskedMimic、动作库与运行配置
│   ├── nav_manager.py          目标、传感器观测、奖励、终止与日志
│   ├── ppo/                    机器人 PPO 与偏好奖励微调
│   ├── pref/                   偏好模型训练
│   ├── flow/                   专家轨迹采集/恢复
│   └── config/, maps/, tests/  配置、地图与集成测试
├── src/crowdsim_ppo/            可独立安装的无仿真 PointGoal PPO
├── data/scripts/, scripts/     AMASS/CSV 转换、诊断、环境与静态审查
├── configs/, examples/         算法/实验配置
├── env/isaaclab3_sim6/          精确环境 manifest、lock 与恢复说明
└── docs/completion_20261001/    本轮证据、原文件快照与修改清单
```

逐文件索引：[source_inventory.csv](completion_20261001/source_inventory.csv)；
机器可读的语法、导入、定义、参考仓库差异及参考仓库独有文件：
[source_inventory.json](completion_20261001/source_inventory.json)。
工具为 [audit_repository.py](../scripts/audit_repository.py)，执行 AST/compile、
shell/YAML/TOML/JSON 解析与 Ruff 未定义名称等错误检查，不执行被审查源码。
最终范围为578个文件；参考仓库独有的5个旧 `agents/masked_mimic/` 文件由当前
`SupervisedAgent` / `MaskedMimicModel` 路径替代，不是缺失的行人运行依赖。

上述范围包含项目源码、脚本、配置和测试；沿用此前范围，排除 FLUX/NavDP 内部实现、
模型与数据二进制、生成输出及第三方 SDK。环境和资源另行验证。
逐文件静态覆盖不等于每一行语义已经被证明，也不等于所有后端和生产规模已执行。
参考文件相同与否不构成正确性判断，当前 ProtoMotions API 比内嵌参考版本更新。

## 2. 固定且实测的环境

| 项目 | 当前值 |
|---|---|
| Python | 3.12.13 |
| Isaac Sim 包 | 6.0.0.1 |
| IsaacLab 产品 VERSION / Python 包 | 3.0.0 / 12.0.0 |
| IsaacLab commit | `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8` |
| ProtoMotions | 3.1.0，editable 当前源码 |
| Torch / GPU | 2.11.0+cu128 / RTX 4080 SUPER 16GB |
| 独立 USD / Kit USD | usd-exchange 3.0.0 的 USD 26.8 / Kit 自带 USD 25.11 |

官方 [Lab 3 beta2 兼容 Sim 6.0](https://github.com/isaac-sim/IsaacLab/discussions/6249)；
其他3.0开发版本可能要求6.1。本项目按精确提交和已验证 profile 固定，不按“3.0”名称
自动升级。Sim API 迁移参考 [官方 6.0 指南](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/migration_guides/isaac_sim_6_0/index.html)。

新 `crowd` extra 补齐 Pillow/OpenCV 元数据；独立 `crowdsim-ppo` 仅依赖 NumPy/Torch。
本轮锁刷新保留273个包的版本，仅更新 editable 项目的依赖元数据。
独立进程检查唯一 USD 提供者、234个 RECORD 哈希、USD 文件读写和转换器导入。
Newton/schemas/coverage 三项既有元数据覆盖仍明确保留，见
[环境说明](../env/isaaclab3_sim6/README.md)；不能将其当成所有后端均已兼容的证明。

## 3. 已修正的代码问题

| 范围 | 修正及实际影响 |
|---|---|
| 行人迁移基础 | 保留已验证的 ProxyArray、XYZW、D6 关节名、局部 reset、共享碰撞世界、地形偏移、冻结推理和 SMPL 蒙皮；没有覆盖成旧参考 API |
| 后端速度合同 | IsaacLab 使用 link origin 的位置和速度；Newton/MuJoCo 处理质心、关节原点及局部/世界角速度的区别，getter/reset/impulse 保持同一合同 |
| 三轴关节 | 真实 PhysX/SMPL 支持 exp_map。Newton/MuJoCo 的串联 hinge 三轴不等价，默认显式拒绝不兼容 exp_map；显式 hinge 仅用于配套数据/策略，单轴 G1 不受该 guard 影响 |
| PoseLib / 数据 | 处理 MJCF joint pos/ref、旋转格式及旧缓存字段；拒绝当前表示无法表达的非共点三轴/slide；修正 CSV 重采样时钟、单帧及 SceneObject 帧数/FPS；新地形缓存使用 tensor 并验证 shape，保留受限的旧本地 NumPy cache 读取 |
| 地图出生点 | 与 ROS origin corner、像素中心、yaw、negate 一致；透明/越界区域视为未知，半径向上取整，连续间距检查；无法满足安全位置时报错，取消任意网格 fallback |
| PointGoal 采样 | 可行位移和起点联合采样，避免 clamp 改变课程距离/角度；真实10k样本距离/朝向违规由1875/1956降为0/0 |
| PPO 数值 | 两份 CrowdSim PPO 在 Adam step 前检查 loss/gradient 的 NaN/Inf；真实 Inf backward 四分支停止且参数/optimizer 未污染；核心 PPO population moments 修复单样本 std/var；尾批仍被消费 |
| 终止统计 | done 只计一次 episode，事件允许重叠；success=reached 且无 collision；训练、微调、评估、采集统一，不再将重叠事件相加作分母 |
| 偏好奖励 | 有效帧 mask 同时作用 attention 与求和；validation eval/no_grad；非空拆分、首轮有效保存、finite loss/gradient 检查；同一短轨迹不再受同批 padding 长度影响 |
| 保存/恢复 | 核心模型和所有任务环境状态在同一带 generation 的 checkpoint 中原子发布，保留 `.previous`；恢复所选文件本身，保留 inactive shard 历史；warmstart 不恢复环境；新文件不回退旧 sidecar，旧文件告知 generation 无法证明 |
| 分布式保存 | snapshot/写入失败向所有 rank 传播；实际双 rank CPU Gloo 验证成功与异地 rank 失败均不死锁 |
| SLURM | 脚本不嵌入 API key 值；安装成功才发布 ready，失败传播退出码；job/step/restart/node 独立 gate，防止旧 marker 放行 |
| 入口/传感器 | `--help` 不启动 SDK；动作只含 mean_action 时可运行；首帧/reset 后先 render 再读 depth；Q/窗口终止生效；异常在 Kit 关闭前输出；日志部分初始化也能释放 |
| 录制/标记 | 不原地改写控制器 marker；实际 dt 决定 FPS；取消后立即重启不会删掉新 session，停止后立即重启先保存旧 session；长帧数按数字排序；先原子保存 `.motion`，视频错误保留数据/PNG并关闭 encoder，清理仅针对当前目录 |
| 安装/CI | 新独立轻量 package、默认 JSON 和 CLI；core wheel 保持核心 package，sdist 补齐源工作流并排除嵌套 build/dist 的旧源码副本；排除 SMPL 授权资源/权重；fork CPU CI 启用且覆盖三测试目录，目标 Python3.12，完整 LFS job 保留 canonical 限制；真实推理配置测试正确标注 LFS 并识别指针，避免普通 CI 对指针 torch.load |

训练数值对照及兼容说明：[crowdsim_fixes.md](completion_20261001/crowdsim_fixes.md)。
保存和 SLURM 细节：[training_checkpoint_slurm.md](completion_20261001/training_checkpoint_slurm.md)。
逐文件修改与原文件快照：[source_change_manifest.json](completion_20261001/source_change_manifest.json)。
没有暂存或提交；原有 USD、FLUX/NavDP、数据集和用户已有改动没有批量覆盖。

### 已定位的 G1 / Newton 历史失败

`examples/data/g1_crouch_to_run.motion` 的旧 body FK 与当前 G1 MJCF 不一致，
首/中/末采样最大偏差10.051mm，集中在腰部和上肢。当前 PoseLib 与真实 Newton
对当前资产的 FK 可达到约1.5μm。测试现在将缓存一致性与后端 FK 分开报告，仍保留
1mm 的后端精度要求。**旧动作没有被伪造、重写或声明已修复**，缓存检查仍返回不一致。
若要训练使用该旧动作，应从原始 DOF/动作来源按当前骨架重建全部 position/rotation/
velocity/contact，并检查对应预训练策略是否仍适用。

诊断：[g1_cached_motion_asset_consistency.json](completion_20261001/g1_cached_motion_asset_consistency.json)。
当前 SMPL AMASS 缓存与骨架一致，采样最大误差约0.53μm：
[smpl_motion_asset_consistency.json](completion_20261001/smpl_motion_asset_consistency.json)。

## 4. 实际验收

| 验证 | 结果与证据 |
|---|---|
| 完整三目录回归 | **1,886 passed、2 skipped、6 deselected**，42.65s；未排除 Newton FK：[JUnit](completion_20261001/final_regression.xml)、[日志](completion_20261001/final_regression.log) |
| 本机真实推理配置 | 另外单独执行上述 deselect 中的两个 SMPL 配置/FPS 用例，**2 passed**：[JUnit](completion_20261001/local_inference_config_tests.xml) |
| 地图/录制/析构必要回归 | 23 passed：[JUnit](completion_20261001/root_lifecycle_mapping_tests.xml) |
| 实际 PhysX SMPL FK | AMASS 及非小角度姿态 exp_map 最大0.584μm，hinge oracle 最大208.467mm；物理推进0步：[原始状态与报告](completion_20261001/isaaclab_smpl_fk.json) |
| 四人真实物理/网格/RGB | **passed**，240步、30Hz、8秒，子集reset 1次、30个归一化 buffer 未变化；最低骨盆0.735m，排除reset瞬移后的各人步行2.65–4.82m：[报告](completion_20261001/pedestrian_runtime_result.json) |
| 实际 MP4 解码 | **80帧、10 FPS、8秒、960×720**，帧变化、物理状态有限，渲染不推进物理：[视频检查](completion_20261001/pedestrian_video_validation.json)、[MP4](../output/completion_validation/smpl_render_final/pedestrian_motion.mp4) |
| 安装工件 | 重建 core/独立 package 的 wheel 和 sdist；离线临时安装及任意 cwd 的 train→resume→eval 证据见 [最终工件验证](completion_20261001/release_artifacts_final.json) |
| 环境/静态覆盖 | 唯一 USD 与固定版本检查、最终逐文件语法/Ruff、lock dry-run 结果见本轮证据目录 |

两项 skip 是未安装的旧 IsaacGym 摩擦测试，六项 deselect 需要真实 LFS payload；
其中本机已有资源的两个 SMPL inference 用例已单独执行，四项其余 payload 测试未执行。
完整回归包含真实 Newton GPU / MuJoCo 状态检查及 CPU Gloo；多数其他测试是 CPU
或接口 fixture，不能等同全部 SDK 训练闭环实测。弃用/prototype 警告仍存在。
GitHub Actions、生产 SLURM/NCCL、多机和长时间训练收敛没有实际执行。
独立复核发现并已修正快速录制切换和 CI 指针问题，见
[integration_review.md](completion_20261001/integration_review.md)。

## 5. 当前运行方式

在仓库根目录使用已配置的环境：

```bash
source ../IsaacLab-Protomotions/.venv/bin/activate
python -I scripts/isaaclab3_environment.py check

# GUI 持续行走，关窗或 Q 结束
python -m CrowdSim.humanoid.simulate --num-humanoids 4

# 无窗口，真实 SMPL 网格和物理，PNG/NPZ/MP4/result 在关闭 Kit 前落盘
python -m CrowdSim.humanoid.simulate \
  --headless --steps 240 --num-humanoids 4 --reset-at 120 \
  --capture --output-dir output/pedestrian_simulation

# 全部已满足本机资源的测试，CPU/GPU 后端按可用环境运行
PYTHONPATH=src:. GLOO_SOCKET_IFNAME=lo \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest --import-mode=importlib protomotions/tests CrowdSim/tests tests \
  -m 'not needs_lfs' -q
```

轻量 PPO 不需要 Isaac Sim、SMPL、AMASS 或 ProtoMotions core：

```bash
python -m pip install -e ./src
crowdsim-ppo-train --device cpu --total-steps 4096 --output output/point_goal
crowdsim-ppo-eval --device cpu --checkpoint output/point_goal/latest.pt --episodes 20
```

`crowdsim-ppo-train` 是2D PointGoal PPO，不是行人姿态控制器。
实际机器人训练、专家采集和偏好微调命令见 [FULL_PIPELINE.md](../FULL_PIPELINE.md)。
安装 core 的 `crowd` extra 只补齐依赖，并不将 source checkout 的 CrowdSim 变成 wheel 模块；
运行完整场景需要保留源仓库。sdist 包含源码，SMPL 授权资源和模型权重需要自行按配置接入。

## 6. 资源和仍需完成的论文复现工作

**本机独立行人 SFM + MaskedMimic 的必要资源齐备**：SMPL MaskedMimic checkpoint/
配套 inference 配置、SMPL 24 bodies / 69 DOF MJCF、AMASS validation、neutral SMPL 参数、
开阔地面和地图。AMASS train/validation/test 共11,786段、4,668,382帧已经准备；
本轮没有再次解压或将它们移动/复制为另一套数据。

| 工作/资源 | 当前情况 |
|---|---|
| 论文条件轨迹 diffusion | 缺论文对应实现、匹配权重及 H/N/M 条件和训练数据定义；现有 SFM 不提供该网络 |
| 论文 SMPL navigation AMP | 缺对应任务、观测/奖励/判别器/训练定义和匹配 actor；现有 MaskedMimic/其他机器人 AMP 不能据名称替代 |
| 精确 AMASS locomotion 子集 | 缺论文2,848段的 manifest、起止帧、筛选、划分；通用 AMASS 完整库不能证明该精确子集 |
| 200s 等论文实验协议 | 缺完全对应的 benchmark、随机种子/统计与配置验收；8秒功能验证不是论文指标复现 |
| 特定机器人/场景 | 默认 robot PPO、Carter、部分四地图/KLT/叉车/HDR 等仍取决于所选配置；最小独立行人入口不使用这些资源 |
| 外观 | neutral SMPL 实体蒙皮可运行，当前纯色；服装 UV/纹理和多样人体外观数据未补齐 |
| 其他后端 SMPL | Newton/MuJoCo 的 exp_map 三轴全链路控制/Jacobian adapter 尚未实现，已防止默默运行错误姿态；目标 IsaacLab PhysX 已实测 |
| 旧 G1 动作 | 缓存/资产不一致未重建；保留了可重复诊断，不能当作兼容训练数据 |

这些缺口已与代码缺陷分开记录，没有生成虚假的权重、数据集清单或论文成绩。
本轮真实验证达到目标环境中的现有行人模拟闭环；严格论文复现与生产训练仍有上述工作。
进一步资源证据见 [此前资源专项](PEDESTRIAN_RESOURCE_AUDIT_20261001_ZH.md) 与
[源码/论文对照](migration_20261001/reference_comparison.md)。
