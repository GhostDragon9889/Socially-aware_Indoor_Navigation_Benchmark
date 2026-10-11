> **本轮最终结果指引**：此页保留子任务阶段记录。2026-10-11最终完整清单为4,293当前+354历史；CPU4,286通过/7跳过，按节点合并真实CUDA5项后4,291通过/2旧IsaacGym跳过。没有新增历史排除项。最新统计、四场景、原始失败和分发结果见[总报告](../../ISAACLAB3_EA_SIM61_MIGRATION_20261011_ZH.md)与[最新矩阵](../functional_audit/functional_test_matrix_latest.json)。

# 构建、隔离安装与源码位置审查（2026-10-11）

## 实际范围

原仓库根目录为 `/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark`。本验证在W目录建立独立工作树快照，包含当前尚未提交的迁移文件；没有Git提交、推送、Git clone或完整新SDK网络安装。文件清单、每文件SHA及内部相对链接位于`source_snapshot_inventory.json`（3890文件，157,265,964字节）。所有核心源码来自该快照或实际wheel安装；原仓库src不在测试sys.path。

隔离venv只继承原已验证runtime补充依赖及Isaac SDK依赖目录，具体路径见`dependency_paths.json`。这证明包和源码位置迁移，没有把继承SDK依赖描述为从零安装证明。没有启动Kit/GPU，没有改变SDK、原runtime、其receipt/锁或运行中的源快照。

## 修复

`MANIFEST.in`原来仅包含旧`env/isaaclab3_sim6`。新增当前`env/isaaclab3_ea_sim61`的toml/lock/json/md/txt文件；实际sdist已经包含profile、manifest、uv.lock、requirements.overlay.lock、runtime_constraints_receipt.txt及bootstrap/runtime代码。解包后的`load_runtime_profile`真实校验所有已登记SHA，32项overlay pin读取通过。

## 成功构建的产物

| 产物 | 大小（字节） | SHA256 | 文件数 |
|---|---:|---|---:|
| wheel | 41111621 | `5328d04639e9f351faa2f21656b3a05436bc4d7583245436ed5df8af278ee388` | 683 |
| sdist | 45519073 | `b06c25bf1bd7f90b4079dfdc79c752241c28db317fa151fdd904e72536be83bb` | 2614 |

`build.log`确认PEP517构建exit0；两产物位于`dist/`。wheel包含protomotions、CrowdSim、crowdsim_ppo、social_navigation、deployment五模块及16个配置/地图文件；不包含两个测试包，不打包受许可限制的SMPL模型资产。完整归档检查见`artifact_inspection.json`。

## 隔离安装与实际入口

`pip install --no-deps dist/*.whl`成功。`isolated_install.json`记录所有五模块实际从测试venv site-packages加载（不是源码editable链接）；六个已安装console入口的真实脚本`--help`均exit0，并在每个子进程断言未导入IsaacLab/IsaacSim/Kit。`verify_install.py`保留可重复执行过程。

安装后实际运行：

- `crowdsim-ppo-train`，128步，保存checkpoint。
- 同一已安装入口加载checkpoint，恢复到256步；`checkpoint.step==256`、52个浮点tensor均有限。
- `crowdsim-ppo-eval`，4episode/2env，exit0。实际success=0、collision=0、mean_episode_length=40；这只证明安装与训练/恢复/评估接口，不是导航质量证据。

完整命令日志和检查结果见`point_goal_train.log`、`point_goal_resume.log`、`point_goal_evaluate.log`、`point_goal_installed_pipeline.json`。

## 源码工作副本与SDK冷准备

`SOCIAL_NAVIGATION_ROOT`显式指向W/source_snapshot，所有内部链接保持相对路径。已安装`social-navigation status`解析到工作副本；五个当前SFM exact-source绑定全部位于该副本，仍严格通过v2 SHA检查，没有借用原R的核心源码。`source_relocation.json`与`source_status.json`保留路径及身份。

SDK61冷准备返回`stock_sim61_runtime`、`installed_SDK_changed=false`、`physics_rerun=false`；版本17.0.2/6.1.0.0/Warp1.16.0，并实际断言无SDK/Kit模块导入；`sdk61_cold_preparation.json`保留结果。SDK文件来自明确的 `/project/EXP/IsaacLab`，它是本次复用的SDK，不是项目核心代码。

## 运行边界

installed wheel已验证提供五个包及PointGoal入口；这不等于每个源码绑定的native桥都能脱离checkout运行。`social-navigation status/simulate`本来就是source-checkout接口，需要`SOCIAL_NAVIGATION_ROOT`或在源码安装环境执行；纯wheel不会自行提供四场景/权重/AMASS或许可证模型。完整物理复现应按Git源目录的bootstrap及资源说明。MANIFEST对旧兼容根链接与third_party采用prune规则，因此不能把sdist/wheel单独安装等同完整历史源码封存协议复现；旧SDK6封存硬编码仍作为不可改历史证据，本轮未声称它们可在SDK61下运行。

## 复测命令

```bash
# 在独立source_snapshot内，工具/依赖只在W隔离venv
/path/to/packaging/.venv/bin/python -I -m build --no-isolation --outdir /path/to/packaging/dist
/path/to/packaging/.venv/bin/python -m pip install --no-deps /path/to/packaging/dist/*.whl
/path/to/packaging/.venv/bin/python -I /path/to/packaging/verify_install.py
```

完整绝对路径和实际环境保存在JSON，便于审查本次复用依赖与源码独立性的边界。
