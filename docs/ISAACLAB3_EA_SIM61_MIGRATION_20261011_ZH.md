# IsaacLab 3.0 EA / IsaacSim 6.1 迁移审查

后续复审、额外缺陷修复、GPU恢复及32/128并行环境验证见[增量复审报告](ISAACLAB3_EA_SIM61_REAUDIT_20261011_ZH.md)。本页保留前次运行的版本与证据，不将其自动视为后续源码的测试结果。

审查日期：2026-10-10～11。项目：`Socially-aware_Indoor_Navigation_Benchmark`。修改保留在本地，**没有提交或推送**。下列结论绑定固定 SDK、源码/资源 SHA 和真实父进程退出码。测试范围及所有早期失败均保留；不能据此保证所有输入、分支和硬件无 bug。

## 1. 目标版本与原环境差异

| 项目 | 本次默认环境 | 保留的旧环境 |
|---|---|---|
| IsaacLab 产品 / Python 包 | 3.0.0 EA / 17.0.2 | 3.0 开发版 / 12.0.0 |
| 固定 SDK 提交 | `ae37b028ea415c91ea2bc32609efcd759ed2b974` | profile `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`；本机旧 checkout `5951d48dde083986489957a7b108ad48a46822a7` |
| IsaacSim | 6.1.0.0 | 6.0.0.1 |
| Python | 3.12.13 | 原环境保留 |
| Torch / Warp | 2.11.0+cu128 / 1.16.0 | 旧锁保留 |
| NumPy / MuJoCo / Newton | 2.5.1 / 3.11.0 / 1.5.2 | 旧 NumPy 2.3.1、Warp 1.15 开发版等保留 |
| 独立 USD 提供者 | usd-exchange 2.3.0 / USD 0.25.5 | 旧 USD 0.26.8 |
| 项目默认解释器 | `.venv/bin/python` → `.runtime/isaaclab3_ea_sim61/.venv/bin/python` | `.runtime/` 下原环境保留 |

本机 SDK 为 `/project/EXP/IsaacLab`；旧 `/project/EXP/SangforAndSaut/IsaacLab-Protomotions` 未修改。项目采用独立 **32 包 overlay**，继承已核验的 SDK，核验 **116 项依赖约束和 169 个 USD RECORD 文件**，不向 SDK 安装项目包或叠加 usd-core/Torch/Warp 二进制提供者。当前默认 profile、锁和安装步骤见 [环境](../env/isaaclab3_ea_sim61/README.md)、[安装](guides/installation.zh-CN.md)、[依赖](guides/dependencies.zh-CN.md)。

已阅读[用户指定的 EA 文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html)、[官方迁移指南](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/migration/migrating_to_isaaclab_3-0.html)、[关节与身体排序说明](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/concepts/joint_and_body_ordering.html)、[接触传感器](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/concepts/sensors/contact_sensor.html)、[已知问题](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/refs/issues.html)及 [Sim 6.1 发布说明](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html)，并以本机固定源码核对实际接口。EA 网站后续变化不能替代本次固定版本证据。

## 2. 已修正的缺陷

| 范围 | 修正与实际验证 |
|---|---|
| 默认安装 / 依赖 | 更新中英指南、VSCode、profile；官方 Git 固定提交、六后端互斥、Python 3.12 上界、tinyobjloader 精确预发布版本；原子安装、已有 SDK 覆盖保护、单一二进制提供者。官方 Git 源 universal uv 解析 dry-run exit0；在已有目标 SDK 上实际建立项目 overlay。完整全新 SDK 下载/安装没有再做一次。 |
| PhysX API | ProxyArray/Torch 视图、XYZW 旋转、根/身体/关节顺序、contact 拼接和 env 索引、空 SceneLib、Actuator、USD scene collector 新接口。 |
| SMPL 关节 | 新转换器默认 69 个并联 hinge 在 PhysX 中实际得到错误的 23 控制维度；选择资产中已有 PhysX D6 variant，实际核验 69 DOF / 24 bodies，拒绝错误维度。原资产字节不变。 |
| N4 场景构造 | 实际 YOHO 四人构造失败：场景 provider 泄漏 session-layer EditTarget，SDK 从 root-layer 复制不到 robot。以 `Usd.EditContext` 限定项目场景 spawn 的编辑目标，保持 provider 原几何/物理字节与 SDK 正常复制方式；四个真实 USD 回归及真实 N4 运行验证。没有修改 SDK 或变更 actor 数规避错误。 |
| 生命周期 | GDB 找到关闭后残留的 SimulationView、PhysX hook 和 Timeline 引用。仅释放终端状态下本任务持有的引用，停止并清 Context 后完整 App.close；6.1 由 SDK 卸载框架。无强制退出、fast shutdown 或伪造 exit0。成功父进程实际退出0；原构造失败实际退出1并保留。 |
| Newton | ModelFlags、目标坐标布局和 free root、接触聚合/环境映射、奇数 decimation、真实 CPU/GPU FK/速度、释放流程；不支持的 exp_map、任意室内 USD、非平地明确拒绝。 |
| RTX | 新 `usdrt.population` 扩展依赖；官方 headless 启动消息设置。真实 RTX、相机、三帧 H264 编码/完整解码和正常退出均通过。 |
| 动作 / 路线 / 相机 | 均匀时间轴、单帧动作和非有限值；PointGoal/驱动参数及动作先校验；A* 端点、角点和样条穿障、重复点曲率；ProxyArray RGB/深度桥接。 |
| 可选基线 | NavDP/FLUX 旧 namespace、SingleXFormPrim、四元数/相机约定、多环境 stuck 状态、灯光循环、启动前版本检查。40 个忽略目录内的源码修改已保存为固定 commit/SHA 补丁，在干净副本实际应用、重复执行和外来改动拒绝均通过。完整可选基线 GPU/服务任务尚未验收。 |
| 预训练资源 | 保留原权重/SHA；两份作者定制参考配置在加载前明确拒绝缺失实现，不伪造类。其余 sidecar、实际模型和本地资源校验见详细报告。 |
| 分发 / 说明 | wheel/sdist、五核心模块、六 CLI 冷启动、安装后 PointGoal 训练/恢复/评估；新增固定源码补丁和机器可读测试矩阵；严格 Sphinx HTML 构建 exit0。 |

详细报告：[SDK](migration_sim61_lab3ea_20261010/sdk_audit/README.md)、[功能](migration_sim61_lab3ea_20261010/functional_audit/FUNCTIONAL_AUDIT_ZH.md)、[模拟器](migration_sim61_lab3ea_20261010/simulator_audit/FINAL_SUPPLEMENT_ZH.md)、[安装解析](migration_sim61_lab3ea_20261010/install_metadata_audit/AUDIT_ZH.md)。可选源码补丁见 [说明](../configs/compatibility/reference_patches/README.md)。

## 3. 静态与动态测试分母

静态扫描 `src`、`scripts`、`third_party`、`examples`、`tools`、根目录 `tests`；`packages` 是兼容链接，不重复统计。**1,687 个 Python 文件 / 362,615 行 / 16,453 个定义 / 12,708 条 import**，全部 AST 解析通过；F821/F822/F823/F901/E9 为零。124 份当前配置/环境/VSCode/CI/根 manifest 的 JSON/YAML/TOML 语法初查通过，新增补丁清单亦实际校验。静态候选保留；10处重复定义候选已分类核对：4处合法property setter、3处AST完全相同的参考定义、3处FLUX明确标注的最终配置覆盖。import-star候选继续保留；解析不是全代码行为证明。

[逐文件 SHA/语法](migration_sim61_lab3ea_20261010/sdk_audit/extended_scope/all_python_inventory.json)、[逐定义范围](migration_sim61_lab3ea_20261010/sdk_audit/extended_scope/all_python_definitions.json)、[import 清单](migration_sim61_lab3ea_20261010/sdk_audit/extended_scope/all_python_imports.json)、[重复定义核对](migration_sim61_lab3ea_20261010/sdk_audit/extended_scope/duplicate_definition_review.json)、[配置语法](migration_sim61_lab3ea_20261010/sdk_audit/configuration_syntax_final.json)。

| 测试记录 | 结果 | 解释 |
|---|---|---|
| 原始完整 CPU 第二轮 | 4,129 passed / 294 failed / 102 errors / 14 skipped | 保留原日志和失败栈，包含旧源码、旧 SDK SHA 和历史结果守卫，不能改写成全部通过。 |
| 当前精确清单最终完整 CPU | **4,286 passed / 7 skipped**，额外 61 subtests passed | 4,293 节点；真实 ORCA 在本次完整执行中已启用，最新配置变更后再次按原清单复测，162.85 s；前两轮的4失败/1失败是测试夹具顺序依赖，修复后按相同清单重跑，无新增排除项。 |
| 实际 CUDA 补测 | **5 passed** | 与 CPU 跳过节点精确对应，包括 Newton solver/FK、baked FK、接触与随机化；29.06 s。 |
| 按节点合并 | **4,291 passed / 2 skipped** | 剩下2项是旧 IsaacGym fixture，不能视为已在 Py3.12 运行。 |
| 历史清单 | **354 节点** | 原始清单 SHA `ed0cdd78e3f41fc53422b4352dc38a5710d2d49f39f2e428e2ea47a5e221f028` 不变；未删除、未改为 xfail。原始失败证据保留。 |
| 最终 collection | **4,647 节点 / 330 测试文件** | 当前4,293 +历史354；collection13.41 s。 |

完整结果见 [逐测试节点矩阵](migration_sim61_lab3ea_20261010/functional_audit/functional_test_matrix_latest.json)、[CPU XML](migration_sim61_lab3ea_20261010/functional_audit/current_clean_sim61_final.xml)、[CUDA XML](migration_sim61_lab3ea_20261010/runtime/gpu_five_actual.xml)、[当前清单](migration_sim61_lab3ea_20261010/functional_audit/current_nodes.txt)及[历史清单](migration_sim61_lab3ea_20261010/functional_audit/historical_nodes.txt)。原始第二轮 [XML](migration_sim61_lab3ea_20261010/functional_audit/original/full_cpu_02.xml.gz)、[失败栈](migration_sim61_lab3ea_20261010/functional_audit/original/full_cpu_02_failures.json.gz)使用可逆 gzip，不删栈。**整个仓库零失败**不成立。

前次实际 CPU 语句覆盖：CrowdSim56.86%、crowdsim_ppo78.06%、deployment28.57%、social_navigation40.47%、ProtoMotions agents95.87%、components98.02%、utils95.38%。测量版本中的342个生产文件有341个当前字节相同，1个已改变；[版本对应](migration_sim61_lab3ea_20261010/functional_audit/coverage_source_version_match.json)和[逐文件/函数未覆盖行](migration_sim61_lab3ea_20261010/functional_audit/functional_file_coverage_matrix_final.json)均保留。最终4,293节点运行没有重新计量覆盖，不能把旧行号当作新代码全覆盖。

## 4. 真实 SDK / AMASS / 场景与训练

AMASS 固定 **2,848 train / 22 validation / 34 test**，三份银行完整字节与 SHA 核验；validation/test 未进入本次训练梯度。固定清单是本地清单，作者的精确2,848 ID未公开。本次 AMP 使用全部2,848专家动作；Diffusion 适用轨迹子集为1,131，不能宣称所有2,848都进入 Diffusion 训练。

四个原始 profile 分别使用4/4/1/1行人。单人场景不能用来证明密集人群互动；本地有界路线不代表全场景覆盖或论文八场景协议。

| 场景 | 行人数 | 实际仿真时间 / 控制步 | 父进程退出码 | 11项本地可靠性检查全部通过 | 每人路径均速 m/s | 证据 |
|---|---:|---|---:|---|---|---|
| YOHO | 4 | 60 s / 1800 | 0 | True | 0.693, 0.779, 0.700, 0.637 | [质量](migration_sim61_lab3ea_20261010/physics/YOHO_1800_sim61_02_quality.json) / [实际退出](migration_sim61_lab3ea_20261010/physics/YOHO_1800_sim61_02_execution.json) |
| NewMetro | 4 | 60 s / 1800 | 0 | True | 0.697, 0.689, 0.700, 0.686 | [质量](migration_sim61_lab3ea_20261010/physics/NewMetro_1800_sim61_quality.json) / [实际退出](migration_sim61_lab3ea_20261010/physics/NewMetro_1800_sim61_execution.json) |
| SecondFloor | 1 | 60 s / 1800 | 0 | True | 0.393 | [质量](migration_sim61_lab3ea_20261010/physics/SecondFloor_1800_sim61_quality.json) / [实际退出](migration_sim61_lab3ea_20261010/physics/SecondFloor_1800_sim61_execution.json) |
| ThirdFloor | 1 | 60 s / 1800 | 0 | True | 0.084 | [质量](migration_sim61_lab3ea_20261010/physics/ThirdFloor_1800_sim61_02_quality.json) / [实际退出](migration_sim61_lab3ea_20261010/physics/ThirdFloor_1800_sim61_02_execution.json) |

ThirdFloor原profile第一次60秒运行正常退出且零跌倒，但8个控制帧出现左手静态接触，最大26.8 N；两项接触门槛失败。原始结果保留，最新profile改用已实际验收的保守AMASS AMP、真实0.65m身体包络和更长身体预测/避障余量，场景/路线/接触门槛均不改。重新验收结果另列，不能把首次失败改成通过。

每个60秒测试包含7,200个120 Hz真实 PhysX子步、69 DOF/24-body SMPL、全帧状态与逐子步法向接触。路线推进、跌倒、人体包络/地图 overlap、1 N以上 peer/static 接触、地面支撑和未过滤接触残差使用独立报告检查。速度允许避障/拐弯减速；规划限速不是实际物理速度保证。SecondFloor实测路径均速约0.393 m/s，低于先前0.5 m/s下界目标，且最长18.1秒未到下一个路线顶点；本轮没有通过降低门槛或修改记录消除该性能缺口，后续需要模型/转弯行为优化。ThirdFloor接触修正后的路径均速约0.084 m/s、只到达1个顶点、最长49.6秒未到下一顶点；接触门槛通过同时出现明显过度制动/通行退化，当前profile保留quality-pending标记，不能当作可用社交导航基准或步态优化完成。

**AMP 真实训练 smoke**：8 env ×32 rollout×12 updates，3,072次实际物理转移；专家采样覆盖全部2,848 train clips；actor/task encoder/critic/discriminator均实际更新，零跌倒、26次reset、完整关闭及父进程exit0。checkpoint包含模型、优化器、normalizers、replay和runner，CPU完整装载与导出推理精确一致；错误默认发布SHA正确拒绝。没有执行GPU中途物理恢复，没有保存PhysX求解器cache，不能据此保证逐步重放等价。此权重是兼容性smoke，**没有晋升为正式行人权重，未证明收敛或动作风格提升**。[训练实际退出](migration_sim61_lab3ea_20261010/training/amp12_sim61_execution.json) / [checkpoint恢复边界](migration_sim61_lab3ea_20261010/training/amp12_sim61_checkpoint_roundtrip.json)。

PointGoal真实train16,384→resume24,576→8次evaluate全部正常；成功率0.25、碰撞率0.625仅证明流程运行，不能视为导航质量达标。另有隔离wheel安装后的128→256训练/恢复/评估通过。

**RTX录制 smoke**：真实Sim6.1渲染3帧、960×720、30 fps、H264、0.1秒，完整ffprobe/ffmpeg解码exit0、父进程exit0。来源是已经物理验收的状态回放，`fresh_physics_run=false`，不是新物理测试或60秒成片。[解码收据](migration_sim61_lab3ea_20261010/render/SecondFloor_three_frames_02_decode.json)。此前四段60秒视频继续保留Sim6.0历史身份。

## 5. 复测步骤

按[中英安装指南](guides/installation.zh-CN.md)建立固定SDK，恢复[许可数据与资源](../CLONE_REPRODUCTION.md)。已有匹配SDK时，在项目根目录：

```bash
python scripts/bootstrap_reproduction.py prepare-runtime --lab-root /project/EXP/IsaacLab
.venv/bin/python scripts/isaaclab3_environment.py --help
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m pytest -q @docs/migration_sim61_lab3ea_20261010/functional_audit/current_nodes.txt
```

完整测试的实际ORCA补充包来源见[收据](migration_sim61_lab3ea_20261010/functional_audit/rvo2_test_overlay_provenance.json)。未准备ORCA/CUDA/旧IsaacGym时的跳过数会不同；不能与本机最终执行直接等同。临时审查目录不是核心物理运行依赖。

真实GPU测试选择新的output目录，保留失败输出：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_trace_pace_verified.py --output output/migration_sim61/SecondFloor -- --config configs/reproduction/four_scene/SecondFloor_profile.json --scene configs/reproduction/four_scene/SecondFloor_scene.yaml --steps 1800 --measure-contacts
.venv/bin/python scripts/evaluate_trace_pace_reliability.py --run output/migration_sim61/SecondFloor --execution output/migration_sim61/SecondFloor_execution.json --output output/migration_sim61/SecondFloor_quality.json
```

## 6. 尚未由本次证据验证的范围

- 每一行动态执行、全分支/全部硬件无bug；这次有逐行静态范围、逐函数清单和实际执行矩阵，未执行范围明确保留。
- 动作自然度、真实摩擦切向冲量、contact-patch足滑、1 N以下接触；足部速度/人体包络等代理量不能代替真实测量或人类视觉评估，法向残差可能发生力抵消。
- 作者八场景、精确作者动作IDs/定制权重及论文全指标；当前本地可靠性通过不表示达到或超越论文。
- NavDP完整7入口、外部HTTP模型服务、匹配权重/场景、teleop/X/uinput；部分可选依赖尚缺。API/CPU修复和源码补丁不能代替整套GPU执行。
- Genesis、旧IsaacGym、全部Newton机器人/场景、多GPU训练、GPU AMP中途恢复；CPU/Gloo或单GPU用例不能替代这些路径。
- Sphinx成功仅是离线严格HTML构建；外部inventory访问和全部重型autodoc运行没有由它验证。
- wheel安装不包含许可数据、SDK、全部原生资产；原生四场景流程仍需源码checkout及资源恢复，不能宣称仅wheel即可完整复现。

## 7. 证据与本地交付

[证据复制索引](migration_sim61_lab3ea_20261010/evidence_copy_index.json)记录原文件与仓库副本SHA、可逆压缩和复制范围。大数据银行、权重、视频和源码副本保留本机原位置，不混入小型文档证据。历史失败、旧源封存和对应SDK均保留。

本次没有执行commit、push或外部发布。主仓库HEAD仍应为审查起点 `bb48a0b79d5b73cde6c3ba85163737e1fec8a194`；最终状态收据另列实际SHA和未提交改动。
