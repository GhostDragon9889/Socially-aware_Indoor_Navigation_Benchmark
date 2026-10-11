> **本轮最终结果指引**：此页保留子任务阶段记录。2026-10-11最终完整清单为4,293当前+354历史；CPU4,286通过/7跳过，按节点合并真实CUDA5项后4,291通过/2旧IsaacGym跳过。没有新增历史排除项。最新统计、四场景、原始失败和分发结果见[总报告](../../ISAACLAB3_EA_SIM61_MIGRATION_20261011_ZH.md)与[最新矩阵](../functional_audit/functional_test_matrix_latest.json)。

# CPU 功能审查与迁移证据（2026-10-10～11）

## 范围与结论边界

仓库：`/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark`。
目标 SDK：Isaac Lab 3.0 EA（包17.0.2）、Isaac Sim6.1.0.0、Torch2.11.0+cu128、NumPy2.5.1、MuJoCo3.11.0。
使用父任务的独立 runtime overlay，`CUDA_VISIBLE_DEVICES=''`。本子任务没有启动 GPU/Kit，没有提交或推送。GPU/真实SDK运行由父任务和模拟器审查子任务负责。

`source_function_inventory_final.json`记录342个生产Python文件、82,035源代码行的AST解析和函数/类/依赖清单，无语法错误。这是静态覆盖范围，不表示每行都运行过。`functional_file_coverage_matrix_final.json`给出实际CPU语句行覆盖、未执行行和每个函数范围。没有测量全分支覆盖，也不承诺不存在任何bug。

## 测试记录

| 运行 | 通过 | 失败 | 错误 | 跳过 | 说明 |
|---|---:|---:|---:|---:|---|
| 全量CPU第一轮 |3992|402|106|14|完整日志`full_cpu_01.log/xml`，61个子测试通过 |
| 全量CPU第二轮 |4129|294|102|14|219.96秒，61个子测试通过；原始失败全部保留 |
| 当前功能4236节点首轮 |4217|5|0|14|192.39秒，61个子测试通过；5项均为现行fixture/上游引用配置/包元数据差异 |
| 当前功能4261节点文档补齐前 |4246|1|0|14|187.70秒，61个子测试通过；GPC链接当时尚未补齐，原日志`current_cpu_final.log/xml`保留 |
| GPC文档精确回归 |1|0|0|0|`gpc_docs_final.log/xml` |
| 当前功能4261节点干净末轮 |4247|0|0|14|148.74秒，61个子测试通过；独立完整运行`current_cpu_clean.log/xml` |
| 真实ORCA隔离扩展 |8|0|0|0|`orca_actual.log/xml`，含原来7个条件跳过的真实求解器验证 |
| 父任务真实CUDA补测 |5|0|0|0|`../runtime/gpu_five_actual.xml`，29.06秒；Newton真实GPU、FK、contacts、randomization |
| 父任务新增SDK61生命周期 |12|0|0|0|`../runtime/sdk61_lifecycle_cpu_final.xml`；另行节点，不混入本4261清单 |
| 最后两项fixture修复回归 |37|0|0|0|`current_remaining_01.xml`；v7实际A*/SDF/Jacobian、五模块包发现、缺候选契约 |
| 真实AMASS数据回归 |61|0|0|0|`actual_amass_tests.xml`；使用真实2848训练/22验证/34测试动作 |
| 真实双进程Gloo |2|0|0|0|`gloo_actual.xml`，确实执行CPU collective与远程rank失败传播 |
| 真实SMPL/MuJoCo/Newton/媒体回归 |130|0|0|0|`targeted_repaired_02.xml`；实际SMPL蒙皮、MJCF运动学、25Hz视频、v9原始producer协议 |
| 冷进程CPU准备接口 |14|0|0|0|`cold_preparation.xml`，包含独立Python证明不导入Torch/SDK |
| USD路径匿名复制/调用契约 |79|0|0|0|`portable_fixture.xml`；真实USD路径复制，fake stage接口不宣称物理执行 |
| 历史/当前版本来源拒绝反例 |4|0|0|0|`historical_rejections.xml`；旧SDK/source不匹配拒绝、当前v2未知编辑拒绝 |

各行运行有重叠，不能相加作独立测试数量。独立末轮4261节点全部现行CPU测试通过；原14跳过中7个ORCA节点随后实际通过。`functional_test_matrix_final.json`按节点归并最新结果为4259通过、2旧IsaacGym条件跳过、0失败/错误，未匹配节点为0。父任务已真实运行5个CUDA节点全部通过，日志`../runtime/gpu_five_actual.log/xml`；最新剩余仅2个旧IsaacGym导入条件。另有父任务新增生命周期12case全部通过，日志`../runtime/sdk61_lifecycle_cpu_final.xml`，不与4261节点数量重复相加。2个上游分支专用NavLab模型保留原文件和SHA，并在反序列化前明确拒绝，不以不存在的类alias假装兼容。所有先前失败日志保持原样。

## 明确修复的业务问题

1. `deployment/motion_utils.py`：使用均匀控制时钟，避免linspace伸缩；支持单帧零时长动作；拒绝无效raw/cache时钟、帧数、形状和非有限值。实际缓存往返逐帧一致。
2. `crowdsim_ppo/point_goal_env.py`：校验动力学参数与动作形状/有限值，无效动作不能先污染位置和episode计数。
3. `CrowdSim/control/drive.py`：非有限动作先拒绝，保留上次有效命令。
4. `CrowdSim/world/planning.py`：零净距不额外膨胀；非法/占据/越界端点拒绝；短步长也检查斜角穿障；重复点曲率输出确定；两点样条支持；平滑切入障碍时回退原已验证A*。
5. `CrowdSim/utils/sensor_stream.py`：通过Torch视图支持SDK61 ProxyArray相机RGB/深度/内参，真实保存PNG/PT回归。
6. `native_social_force_runtime.py`：登记当前`native_sfm.current_scene_sources.v2`的精确source SHA。保留历史SHA及逐字方法一致性；只反向恢复仓库位置表达，未改力积分公式。未知编辑/依赖变化仍拒绝。
7. `relocated_terminal.py`与生命周期调用：源目录必须解析为本仓库，拒绝相同名字的外部目录；目的目录可独立派生。
8. 本地`data/external_resources.json`：对已有真实AMASS包建立明确位置绑定，未修改数据、manifest或SHA。

## 真实数据资源结论

先前22错误/1失败不是数据包缺失，而是迁移后的`data/datasets/amass`目标路径没有配置。已校验并绑定：

- train：`data/diffusion_amp/amass_style_2848/manifest_labeled_v7_train.pt`，1,155,817,682字节，SHA256`46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`。
- validation：`/project/EXP/SangforAndSaut/Assets/motion/amass_smpl_validation.pt`，678,009,566字节，SHA256`ae0389770ad4a1dfb6d916b80bbe88b8ff89c7ab80a22d71b77597b6c302cf03`。
- test：`/project/EXP/SangforAndSaut/Assets/motion/amass_smpl_test.pt`，65,614,188字节，SHA256`ae3e55e7ab6934919f769e95a141fa1e97b0d305d1b96ce7503ea8fff49d9f23`。

三份完整字节/SHA均匹配v7清单，证据`actual_amass_bank_bindings.json`。registry是本地忽略文件，不把许可数据纳入代码发布。保持train/validation/test分离，不将独立评估动作加入训练。

## 当前CPU功能与历史协议矩阵

`current_cpu_nodes_final.txt`精确列出4261节点；`historical_excluded_nodes.txt`精确列出354条历史或旧行为差异。最终完整4615节点collect记录在`collection_final.log`；此前4590节点清单也保留。`functional_test_matrix_final.json`列出327测试文件、319个包含当前测试的文件、29个包含历史节点的文件，并保留精确测试node、显式import和逐节点结果。这不是把失败删除或改成xfail：全量第二轮294失败/102错误仍完整保留。

`failure_compatibility_matrix.json`逐项给出410条记录（396失败/错误+14跳过）、类别、原错误、源路径/实际SHA。`full_cpu_02_failures.json`保留全堆栈。主要原因是旧封存初始化core、transitive lifecycle、episode/minimum-horizon source、旧训练result/checkpoint/ownerplan被当前代码或后来运行改变。封存原source、旧SHA、旧日志都保持原样。

`source_guard_version_matrix.json`补充25条可静态解析的精确源码守卫，6项实测不匹配。例如旧ContactSensor要求`d8a7bb...`而SDK61文件为`000dde...`；旧reader lifecycle要求`be868...`，当前source为`6e28e...`；旧paired core要求`3395...`，当前为`59135...`。4个新增实测反例证明旧协议正确拒绝新版本；当前SFM v2只接受已登记源，并拒绝未登记变化。

A*角点安全修复确实改变2条旧浮点位golden路线。旧golden不改为新数值；这一算法行为差异明确保留在历史矩阵中。新路线与timed终点一致、障碍安全、足够lookahead的终点到达都有当前测试。

## 生产语句行覆盖（当前4261节点完整覆盖轮，不含tests）

| 模块 | 文件数 | 覆盖语句行/总语句行 | 覆盖 |
|---|---:|---:|---:|
| CrowdSim |197|15456/27183|56.86%|
| crowdsim_ppo |8|758/971|78.06%|
| deployment |7|350/1225|28.57%|
| social_navigation |12|276/682|40.47%|
| protomotions/agents |87|6483/6762|95.87%|
| protomotions/components |11|2977/3037|98.02%|
| protomotions/utils |20|1674/1755|95.38%|

覆盖报告来自coverage的真实执行行；动态派生`:relocated_checkout`等虚拟文件没有独立源文件，coverage json使用ignore-errors并把警告保留在`current_coverage_reporting_final.log`。这些虚拟函数不能被虚构算作每行实测。`src/datautils`不存在，不能把该目录标为已测试；实际数据预处理源码范围应看父任务完整仓库矩阵。

## 复测命令

在仓库根目录执行（独立runtime overlay，不改SDK）：

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=4 '/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/runtime/.venv/bin/python' -m pytest -q '@/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/functional_audit/current_cpu_nodes_final.txt'
```

全量复测删去`@...current_cpu_nodes_final.txt`参数即可。真实Gloo需要允许本机loopback通信；第一次沙箱网络拒绝不是collective逻辑错误，后续已实际通过。7个ORCA测试在独立扩展路径下已实际通过。原5个CUDA跳过已由父任务实跑通过，2个旧IsaacGym摩擦fixture仍需要旧模块；其精确节点分别列于`gpu_pending_nodes.txt`（保留原待测清单，实际后续通过）和`legacy_isaacgym_skipped_nodes.txt`。

## SDK依赖适配

目标MuJoCo3.11与dm-control1.0.40实际Physics构造不兼容；使用官方支持MuJoCo3.11的dm-control1.0.44。没有降低SDK MuJoCo，也没有修改SDK源码。模拟器子任务另行记录真实1000-step物理验证。官方release：<https://github.com/google-deepmind/dm_control/releases/tag/1.0.44>。

## ORCA真实复测与隔离范围

已有公共RVO2源commit `c2c46ba8d59556aa10faf03479293236efea154d`、TRACE commit `594b2af0824c5853e1a1884e167cd5ed097dd81c`均核验匹配固定值。仅将原CPython3.12扩展复制至W测试路径；SHA256 `eaf15a44e6ae67eb5eba8228ddc150f5052480e5e8fc0bf172e929daf9ef850d`、1,014,304字节；证据`rvo2_test_overlay_provenance.json`。没有安装旧runtime依赖或修改目标SDK/新runtime/receipt/lock。

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/functional_audit/optional_rvo2:src:.' OMP_NUM_THREADS=4 '/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/runtime/.venv/bin/python' -m pytest -q tests/test_diffusion_orca.py
```

8项实测包括600步正面对向避碰与多边形停步、实际60Hz碰撞核验、严格10Hz时间、确定性、场景split分离/篡改拒绝、同步真实邻居窗口、占据栅格旋转、真实Torch五状态输入。原CPU XML中的7skip不改；`orca_actual.xml`给出后续8pass。

## 分发与安装验证

已从独立工作副本成功构建wheel/sdist，并以原仓库src不在sys.path的隔离安装验证五核心模块、六入口、安装后PointGoal train128/resume256/eval4及SDK61冷准备。具体证据与适用边界见`packaging/PACKAGE_AUDIT_ZH.md`，没有复用原仓库核心源码来假装安装成功。SDK依赖显式继承已验证环境，因此不宣称完整fresh SDK网络安装已复现。
