# NavIsaacLab 原始源码与当前行人迁移对照

日期：2026-10-01。原始参考 A：`/project/EXP/NavIsaaclab2.0`，HEAD `d742f0c5b421bbc281e2112e7e50427e0336c16a`；当前工作树 B：`/project/EXP/SangforAndSaut/ProtoMotions`，HEAD `7a8417a9f55b1586b0d6e8f46578c5bc9d7ce1f1`。对照包含未提交修改，HEAD 仅作定位。机器证据及逐文件哈希见 [reference_comparison.json](reference_comparison.json)。**文件统计、模块行号和原 SHA-256 是四人最终采集之前的中间源码快照，不能当作最终交付代码封存；最终验证见第 7 节及[最终迁移报告](../ISAAC_SIM6_LAB3_MIGRATION_ZH.md)。**旧目录对照报告保留为迁移前快照。

## 1. 版本名称必须分开

| 对象 | 真实声明/安装证据 | 结论 |
|---|---|---|
| A 的 NavIsaacLab 项目 | `README.md:1` 为 NavIsaacLab 2.0；根 `pyproject.toml:6–7` 为 `crowdsim-ppo==0.1.0` | 项目的 2.0 不代表其内嵌 ProtoMotions 是 2.0 |
| A 的内嵌 ProtoMotions | `PROTOMOTIONS.md:3,23` 明确为 ProtoMotions 3；包 `__init__.py` 无版本声明 | 应称“Nav fork 内嵌的较早 ProtoMotions 3 系源码”；精确上游 tag/发布号没有确认，不能写成 ProtoMotions 2.0 |
| B 的核心包 | 根 `pyproject.toml:7` 为 `protomotions==3.1.0`，专用环境 editable 使用本工作树 | 新的 MaskedMimic/AMP 组件接口与 A 有实际差异，不能整目录覆盖 |
| 目标后端 | Isaac Sim `6.0.0.1`；IsaacLab 产品 `VERSION=3.0.0`、包 `12.0.0`；源码固定 `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8` | 产品线和 Python 包编号不同；复现依据是源码 pin 与 profile，见 [环境说明](../../env/isaaclab3_sim6/README.md) |

中间源码快照重算 A/B 的 `protomotions/` 文件：A 432 个（Python 164），B 623 个（Python 353）；相同 178、修改 248、仅 A 6、仅 B 197。A 有 138 个 LFS 指针，B 有 49 个。文件计数不表示行为差异数量，也不表示验收完成度。

## 2. 当前行人控制链与论文链

当前实际控制链：地图/A* 全局路径 → SFM 局部目标 → MaskedMimic 稀疏 Pelvis 目标/context → SMPL 69 个关节动作 → IsaacLab PhysX articulation → 状态反馈与 SMPL 蒙皮。生产入口由 `CrowdSim/world/builder.py`、`CrowdSim/protomotions_runtime.py`、`CrowdSim/nav_manager.py` 连接。

本地论文已读取 PDF 第 6、7 页并查看实际图 4：IV-A 是以历史轨迹 H、邻居轨迹 N、占据地图 M 为条件的 U-Net 轨迹扩散，预测未来 5 秒；IV-B 是带动作风格判别器的 AMP 全身控制器，按轨迹、能耗和方向奖励训练 SMPL actor/critic；IV-C 指定 2,848 段标签筛选的 AMASS locomotion 动作。论文 SFM+AMP 消融保留同一个 AMP 控制器。因此当前 **SFM+MaskedMimic 是已迁移运行的控制链，仍未完成论文 Diffusion+AMP 或 SFM+AMP 对照方法的复现**。论文文字证据见 [文本提取](../full_audit_20261001/navisaaclab_paper.txt)，原 PDF 为仓库根目录 `Xia et al. - 2026 - NavIsaacLab Generating Realistic Crowd via Parallel Robot Learning for Benchmarking Human-aware Nav.pdf`。

## 3. 核心接口逐项对照

下表路径均相对于 A/B 根目录；对应绝对路径和源码 SHA-256 在 JSON 的 `module_pairs` 中。

| 接口 | A 原始实现 | B 当前实现与迁移状态 |
|---|---|---|
| MaskedMimic agent/model | `protomotions/agents/masked_mimic/{agent,model,config}.py`；专用 `MaskedMimic` agent | `agents/supervised/agent.py:34`、`masked_mimic_model.py:41`、`masked_mimic_config.py:103`；算法迁入通用 SupervisedAgent，模型负责 rollout 状态。旧 Python 类路径/配置不能直接复用；现有 B 配套 checkpoint 已严格加载 |
| AMP | `agents/amp/agent.py` 内含训练逻辑 | `agents/amp/agent.py` 使用 mixin，主体在 `agents/amp/component.py`；有通用 AMP 框架不等于具备论文 SMPL 导航 actor/config |
| 配置恢复 | `CrowdSim/protomotions_runtime.py:122,132` 调 `apply_backward_compatibility_fixes`；A 的该函数 `utils/inference_utils.py:77` 没有实际迁移逻辑 | B `CrowdSim/protomotions_runtime.py:126,136` 按当前 `apply_all_inference_overrides` 恢复配置、切换后端，显式覆盖 MotionLib。没有复制旧空函数掩盖接口错误 |
| 推理模式 | A runtime setup/load 后未明确冻结归一化统计 | B runtime `:258–260` 使用 `load_training_state=False`、`agent.eval()`、关闭模型梯度；最新物理短跑检查 30 个归一化 buffer |
| 仿真配置/底层数据 | `isaaclab.sim.PhysxCfg`、`SimulationCfg(physx=...)`、`root_physx_view` 与 Torch 原生读法 | B `simulator/isaaclab/simulator.py:13,101,435` 使用 `isaaclab_physx.physics.PhysxCfg`、`physics=`、`root_view`、Warp/Torch view；不能只改 import 后保留旧读写假设 |
| 四元数 | A `simulator/isaaclab/config.py:76` 为 `w_last=False`，即旧后端 WXYZ；A 共同 `RobotState` 本来就是 XYZW | B 后端 `config.py:64` 与 `native_quaternion_order` 为 XYZW；共同 RobotState 仍为 XYZW。CrowdSim Drive/mesh 通过显式边界转换，mesh 内部与相机配置仍各按其接口格式使用，不能全局替换四元数顺序 |
| SMPL 资产入口 | A `robot_configs/smpl.py:65–66` 同时给 MJCF 与预制 USD；scene `:172–173` 加载 USD，该 SMPL USD 当前为 LFS 文本指针 | B 选真实 `mjcf/smpl_humanoid.xml`；scene `:157` 调 MJCF→USD。这条当前路径无需将旧 SMPL USD 指针作为强制下载项 |
| 多轴关节/D6 | A 使用旧预制 USD 和其关节名称 | B `utils/mjcf_to_usd.py:253–257` 安装 D6 转换修复；`mjcf_d6_workaround.py` 处理合并关节坐标框架与 MJCF spring gains；`actuator_groups.py:36` 将语义 `L_Hip_{x,y,z}` 映射为后端 `L_Hip_x:{0,1,2}` 等名字。只有骨架文件相同仍不足以证明动力学等效 |
| 环境子集 context | A MaskedMimic 目标管线按全部环境处理 | B `envs/context_views.py:599` 保留 `env_ids`；control `:415,427` 处理子集。CrowdSim 当前 `nav_manager.py:926` 用全局 ID 对齐局部行，避免读取另一行人的目标时刻/冻结状态 |
| 人体可视网格 | A `simulator/isaaclab/simulator.py:166,182` 内嵌 CrowdSim mesh hook | B 的控制/物理框架与 CrowdSim 蒙皮层连接；`tools/smpl_mesh_visualizer.py` 明确处理 native XYZW 与内部 WXYZ、旧模型受限加载和失败时的可见性恢复。现有纯色真 SMPL 外观已接入 |
| 动作数据/重采样 | 两者均有 MotionLib 的打包 `.pt` 与 `gts/grs/dps/dvs` 等数值字段 | B 增加分片/lifecycle/context 功能；当前 AMASS 的 24 bodies/69 DOF/XYZW 与 B 匹配，138 段 test 库已在新环境逐段插值。短片段资格的 update/load/显式 IDs 与部分 resample 修复见 [专项验证](motion_sampling_ppo.md) |

## 4. 物理骨架与现有权重

A/B 的 `protomotions/data/assets/mjcf/smpl_humanoid.xml` **字节完全相同**，SHA-256 `58dc997c830b3ded2d2b4528797ed2768f9528b2f25aa008e541fe3a9083e7cc`：24 个 body、69 个 hinge DOF/69 个 actuator，加 1 个根 free joint。根自由度不是 69 个策略动作的一部分。实际生成的 PhysX/D6 命名由 B 单独映射。

| 当前预训练模型目录 | 骨架 bodies/DOF | 已有证据与用途边界 |
|---|---|---|
| `masked_mimic/smpl` | 24/69 | 当前行人控制器；checkpoint + 配套配置已通过 CPU strict load，最新实际 PhysX 短跑使用这套权重 |
| `motion_tracker/smpl`、`smpl-terrains` | 24/69 | PPO 跟踪器 strict load 通过；未确认是论文导航 AMP。配置默认动作路径含 G1 数据，需要明确覆写为 SMPL MotionLib，CrowdSim 已覆写 |
| `motion_tracker/soma-bones`、`soma_bones_fsq`、`soma_bones_fsq_amp_muon` | 23/66 | 三套 strict load 通过；SOMA 骨架不能直接替代 SMPL；FSQ AMP 名称也不证明任务等于论文 |
| `motion_tracker/g1-bones-deploy` | 33/29 | G1 AMP 模型 strict load 通过；不能替代人体 SMPL actor |
| `gpc_prior/soma_bones` | 23/66 | 自回归 latent prior strict load 通过；不等于论文的 H/N/M 条件 2D 轨迹扩散 |

八套权重的严格加载来源为同日 [资源机器证据](../full_audit_20261001/resources.json)，`missing_keys/unexpected_keys` 均为空。本次文档对照没有重新加载全部模型。A 原始目录未发现对应 checkpoint，不能宣称已将 A 的旧权重自动转换成 B 权重。

## 5. 已存在与尚未确认的资源

**当前链路已存在：** MaskedMimic SMPL 模型/推理配置、真实 MJCF、AMASS train/validation/test（11,303/345/138 段，共 4,668,382 帧），以及已安全接入的 neutral SMPL 参数模型。`data/smpl/SMPL_NEUTRAL.pkl` 链接到已有 arena 模型，源文件未移动；真实 CPU LBS 已验证 6,890 顶点/13,776 面，详见 [安装证据](../full_audit_20261001/smpl_installation.json) 与 [蒙皮证据](../full_audit_20261001/smpl_geometry_verification.json)。

**纯色人体可运行，逼真贴图仍缺：** 尚无已验证的 SMPL UV/纹理资源。四份 DynBench 占据地图、KLT/Forklift 与两个 HDR 的缺口见原 [资源审计](../full_audit_20261001/resources.md)，它们按所选场景适用，不能全部计为本轮无机器人最小行人测试的必需项。机器人 PPO checkpoint 也是机器人推理/采集或训练产物，不是行人控制权重。

**严格论文仍未确认：**

1. 匹配 IV-A 的条件轨迹扩散实现、checkpoint、配置以及接入 CrowdSim 行人目标生成的实际入口。
2. 匹配 IV-B 的 SMPL 24/69 导航 AMP actor 与任务配置、观测/归一化、轨迹接口、动作映射和奖励契约；从头训练还需 AMP 判别器/训练超参数。
3. IV-C 的 2,848 段 locomotion clip 精确清单、片段范围、标签筛选、划分与预处理。当前通用 AMASS 不等同于此清单。
4. 高层轨迹训练数据与 H/N/M 条件构造、采样时钟、训练/验证切分。没有依据假定某个公开轨迹数据集单独就能补齐。
5. 论文行人评估的 8 场景、3–15 人、200 秒配置，以及 15 关键点/30 Hz、heading/scale 对齐、cluster/entropy/Motion-MMD 的完整任务和评估 manifest。

这里的“未确认”限定于当前审查的代码/资源，不能据此推断作者从未发布。现有 FLUX/NavDP、SOMA AMP 与 GPC 权重也不能按名字当作对应论文行人模型。

## 6. 中间实际验收及边界

中间有效结果为 [smpl_physics/result.json](../../output/migration_validation/smpl_physics/result.json)：**2 人、120 步、30 Hz、4 秒、1 次 reset，真实 SMPL mesh 创建成功；最低 root 高度 0.8717187 m；排除 reset 跳跃的累计行走距离分别 1.5775133/1.2234433 m；30 个归一化 buffer 已检查。**最终四人验证已完成，见下一节。

这些行走距离来自 `walked_distance_m`，不能用包含传送/reset 位移的 `root_displacement_m` 替代。`result_before_ground_alignment.json` 的最低 root 高度约 -77 m，即使旧脚本标记 passed，也不满足现在的有效验收；它保留为地面偏移修复前的失败证据。

这证明当前 Sim6/Lab3 下的 SFM+SMPL MaskedMimic 短时物理/蒙皮闭环已越过初始化与基础状态合同。尚不能证明 200 秒稳定性、密集人群避碰、论文动作质量或 Diffusion+AMP 指标。

## 7. 最终验证更新

本节更新此前的两人中间状态。最终实现、运行步骤、范围和排除项以[最终迁移报告](../ISAAC_SIM6_LAB3_MIGRATION_ZH.md)为准；本次只更新这份对照文档和 JSON，没有改业务代码或环境。

| 最终验证项 | 实际结果与证据 |
|---|---|
| 四人共享物理世界 | [最终 result.json](../../output/migration_validation/smpl_render_final/result.json) 为 passed；SFM+SMPL MaskedMimic / IsaacLab PhysX，30 Hz、240 步、8 秒、2 次 reset |
| 身体状态 | 最低 root 高度 **0.8431218 m**；排除 reset 跳跃后的累计行走距离 **4.9869 / 4.7205 / 3.9094 / 3.7631 m**；30 个归一化 buffer 已检查 |
| 真实可见网格与录像 | 最终报告确认四个 SMPL 全部可见；[MP4](../../output/migration_validation/smpl_render_final/pedestrian_motion.mp4) **67,074 字节**。本次 ffprobe 复核 **960×720、80 帧、10 FPS、8.000 秒**；控制仍为 30 Hz，录像降采样不额外推进物理 |
| 25 Hz 时钟 | [25 Hz result.json](../../output/migration_validation/smpl_25hz/result.json) 为 passed：2 人、100 步、4 秒、1 次 reset，最低 root 高度 **0.8022825 m**；20 帧、5 Hz capture，与控制时钟分别记录 |
| 最终目标回归 | [日志](runtime/migration_final_tests.log) 与 [JUnit](runtime/migration_final_tests.xml)：**1,750 passed、2 skipped、5 deselected**，30.00 秒；JUnit 为 0 errors / 0 failures。已知 Newton/G1 CUDA FK 误差项与 4 项 LFS 测试未选，2 项旧 IsaacGym 测试跳过，不能称所有后端全量无遗漏通过 |

再次只读复核专用环境的精确版本：**Python 3.12.13、ProtoMotions 3.1.0、Isaac Sim 6.0.0.1、IsaacLab Python 包 12.0.0、Torch 2.11.0+cu128、usd-exchange 3.0.0、pytest 9.0.3**；相邻 IsaacLab 源码 `VERSION` 为 **3.0.0**，HEAD 仍为 **4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8**。未将产品 3.0.0 与包 12.0.0 混用，也未把当前环境写成 Sim6.0.1 或 Lab3 EA。

以上证明现有控制链的短时共享世界行走、重置、姿势映射、真实蒙皮和 MP4 输出。最小测试场景是无障碍地面，不能扩大为缺资源的仓库/医院等完整场景验证；也没有完成论文 200 秒密集人群评测。第 5 节列出的 matching diffusion、SMPL 导航 AMP、2,848 段动作与评估 manifest 等严格论文缺口继续成立。
