# IsaacLab 3.0 EA / Isaac Sim 6.1 第二轮修复与未筛选全量回归

日期：2026-10-11。**本轮没有达到“每行、每项功能全部验证并修复”的验收条件。最终 4,745 个唯一测试节点中，4,645 项通过，41 项失败、57 项初始化错误，2 项跳过。** 初始化错误后的功能没有执行，不能算作已验证。没有零缺陷结论。

已继续修复项目源码、补充回归并实际运行 GPU 验证；未修改两个参考 SDK，没有提交或推送，保留工作区原有修改。此前报告中的“排除 354 项历史失败后的通过结果”不代表本轮的全量结果；本报告使用不筛选的完整 pytest 和实际 CUDA 补测。

## 1. 环境与分母

实际运行环境为项目 `.venv` 指向的 `.runtime/isaaclab3_ea_sim61/.venv`，复用 `/project/EXP/IsaacLab`。核验版本：IsaacLab 包 17.0.2 / 产品 3.0 EA、Isaac Sim 6.1.0.0、Torch 2.11.0+cu128、Warp 1.16.0、NumPy 2.5.1、usd-exchange 2.3.0。环境检查还核验了 169 个 USD RECORD 文件和 USD 保存/重开。[实际环境检查](../output/verification_completion_20261011/environment_check.json)

参考：[IsaacLab 3.0 EA 官方文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html)、[Isaac Sim 6.1 官方文档](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/index.html)。SDK API 的实际兼容结论来自本机固定源码和实测。旧 `/project/EXP/SangforAndSaut/IsaacLab-Protomotions` 用于历史接口对照，旧源码身份没有被改写为新版本运行证据。

| 检查范围 | 实测结果 | 可以说明什么 |
|---|---|---|
| 当前目录 Python 清单 | 1,733 文件、379,644 物理行；其中 701 个 third_party 文件 | 每文件读入、记录 SHA、AST 解析；不是逐行人工审查或逐行运行 |
| docs 历史/冻结 Python 源码 | 1,265 文件、318,734 物理行 | 全部可解析；不代表冻结 SDK 入口在 Sim6.1 可运行 |
| 上述合计 | 2,998 文件、698,378 物理行，0 个 AST 错误 | 显式范围内的 Python 语法检查 |
| 致命静态规则 | F821/F822/F823/F901/E9：0 项发现 | 保留仓库既有、有说明的规则豁免；不是所有 Ruff 规则或所有语言检查 |
| 未筛选 pytest | **4638 passed、41 failed、57 errors、9 skipped、82 subtests passed**；240.70 秒，父进程实测退出 1 | 执行仓库配置的三个测试目录，没有 `--deselect`、`-k` 或 `-m` 排除 |
| 真实 CUDA 补测 | 原 6 项 CUDA 测试 + 新 Social Pop-Art CUDA 保存/恢复/继续更新：7 项通过 | 将 CPU 运行中的对应 7 个跳过节点替换为实际结果 |
| 按唯一节点合并 | **4645 passed、41 failed、57 errors、2 skipped** | 原始 XML 不改写；不是将重复专项测试相加 |

当前清单去除了 5 个 `usd_convert/` 与 `tools/asset_conversion/` 指向相同文件的别名，详细记录在清单中。源码范围包含 src、scripts、tests、examples、tools、third_party、选定 data Python 目录、usd_convert 和根目录 Python 文件；排除环境、SDK、生成输出及其重复链接。docs 单独列出。其他语言、二进制资产、完整文档站和交互功能不在 Python AST 或上述覆盖率分母中。

剩余两个跳过项属于旧 IsaacGym 后端；本轮没有将它们作为 Sim6.1 已通过功能。可选 ORCA 的本地已核验二进制通过独立 PYTHONPATH 加载，其测试在全量回归中执行。[清单](../output/verification_completion_20261011/inventory.json)、[docs 清单](../output/verification_completion_20261011/historical_and_docs_inventory.json)、[静态结果](../output/verification_completion_20261011/static_findings.json)

## 2. 本轮主要修复

### 产物迁移与 CPU 证据读取

- 新增 `social_navigation.artifact_identity`：仅迁移已登记旧项目根目录及 src 包布局；仍校验完整 SHA、字节数、字段集合和精确类型。拒绝路径穿越、相似前缀、布尔值冒充整数、不同元数据和被篡改内容；命令字符串不会被泛化重写。
- 恢复终端 checkpoint、完成收据及状态读取在新目录下的身份比较。实际读取并计算了旧 S3-R 文件的 651,040,366 字节 SHA，与归档记录一致；**该校验没有加载权重，也没有证明旧 S3-R 模型在新 SDK 恢复训练成功**。[实际字节校验](../output/verification_completion_20261011/recovery_checkpoint_bytes.json)
- CPU 导航/风格读取器采用独立版本。根目录调整用完整字节逆变换验证；测量函数用固定归档源码与 AST 比较。生命周期依赖仅证明 CPU `record()` 函数不变，不据此证明 SDK 运行。
- 新 `native_map_audit_reader` 为旧 v7 地图审计提供路径迁移适配，冻结归档不变。读取器仍检查每个 tar 成员的大小和 SHA；重新计算容器 SHA 也不能掩盖内部源码篡改。源码推导保留完整逆变换证明。

### SDK6.1 资源生命周期与接触

- 修复导航生命周期进入 SDK6.1 分支后仍执行旧接触源码守卫的问题。
- 修复结束阶段替换 report 后，内层生命周期把清理结果写入旧 report 的问题。
- 清理前验证世界、articulation、producer、provider 和 context 的真实归属，释放 native facade/core 的互相引用及绑定方法引用。
- 支持组合、仅地面和原始 peer monitor 三种接触所有者。原始 monitor 显式记录 world/robot；在停时间线和 App.close 前验证对应物理视图已释放。
- 保留上一轮版本固定的 eventdispatcher 清理和完整 App.close 流程。所有原生成功结论同时要求子进程输出与父进程观察到的实际退出码，不用结果 JSON 代替进程退出。

### Social Pop-Art 数值训练与恢复

- 旧 Social 训练器从变化后的基类提取方法，导致原有 Pop-Art 插入不再匹配。新 `popart.sim61.v2` 使用固定的历史数值方法来源，并对当前依赖做独立校验；源码选择有完整逆变换证明。
- 修复 staged 配置模块对 src 布局的路径判断；派生模块使用独立命名空间，保持原始文件不变。
- CUDA 检查点保存/恢复发现并修复 CPU 检查点张量与 GPU 固定方差、冻结动作掩码混用的错误。RNG 缓冲保留在 CPU，模型和两个 Adam 的状态由各自加载器放到目标设备。
- 实测六轮 PPO/12 个优化器 minibatch；新增 CPU 和实际 CUDA 的更新→保存→恢复→再次更新测试。两套 Adam 的 moments、全部模型权重及 CPU 采样 RNG 一致；actor、critic、encoder、discriminator 都更新，统计量计数正确，所有参数有限。篡改固定 log_std 和冻结关节动作仍被拒绝。
- 新运行描述使用 `describe_runtime()`；旧 `describe()` 只读取并核验冻结的 v1 元数据。**新 v2 数值训练验证不等于旧 v1 物理训练/审计流水线已整体迁移，也没有自动接纳旧 v1 检查点。** 那些 World、owner、审计入口仍在下方未通过清单中。

### 测试有效性

- 将“未导入 Torch”的检查放到真实新进程，避免完整测试集收集阶段已导入 Torch 导致误判。
- 旧路径路线 bit 快照与已修复的防穿角算法冲突；两项测试改为验证几何可走性、密集线段采样、准确时序端点、索引、到达状态和输入不变性，保留旧 bit 值并明确验证版本差异。
- 两项旧 Sim6.0 源码守卫测试使用精确还原的临时旧源码输入，仍检查原完整 SHA 和错误模块来源；没有放宽生产代码的旧版本守卫。

## 3. 真实并行物理环境与退出

全部在一张 RTX 4080 SUPER 16 GiB 上串行运行，SDK 版本同上。

| 场景 | 实际执行与断言 | 父进程退出 |
|---|---|---|
| 4 环境组合接触 | 地面基线、墙接触、两 actor 接触；36 controls / 144 子步；5 个所有者视图在 App.close 前不可达 | 0 |
| 4 环境仅地面适配器 | 12 controls / 48 子步；COM、有限数值、时钟和 1 个视图释放 | 0 |
| 4 环境原始 peer monitor | 三类摆位；36 controls / 144 子步；墙/peer 通道隔离、对角自接触为零、4 个视图释放 | 0 |
| 128 环境物理恢复并继续优化 | 恢复六组 root/DOF 状态；64 个逆序非连续环境复位，其余 64 个保持隔离；再执行 4096 个实际物理 transitions | 0 |
| Social Pop-Art CUDA 数值恢复 | 合成 rollout；两套 Adam、模型、RNG 保存恢复，再次更新与篡改拒绝；不算物理 rollout | 0 |

组合/peer 模式的墙和人体接触使用已知摆位，验证非目标通道及其他环境没有串扰；这些是接触测量夹具，不是自主导航或人体安全力阈值验证。仅地面模式显式标记 `peer_and_static_channels_measured=false`，不能用其零占位数组证明这些通道测量通过，也没有单独证明所有地面支撑事件。

128 环境恢复时 root_vel 最大绝对误差为 2.98e-8，另五组状态误差为零。更新计数 12→13，实际新增 4096 transitions、26 次 timeout、0 次真实终止，四个网络参数均变化且指标有限。空复位和非连续索引复位也验证了隔离。**PhysX 求解器缓存没有恢复，不承诺之后的轨迹逐 bit 相同。**

此前冷启动 128 环境的 12 次更新、49,152 transitions、436 resets、0 falls，以及 2848 专家片段覆盖记录继续保留；本轮恢复使用该实际检查点。[此前冷启动结果](../output/isaaclab61_reaudit_20261011/amp128_cold_verified/result.json)

最新实际结果和外部退出收据：

- [组合接触](../output/verification_completion_20261011/latest_combined/result.json) / [退出收据](../output/verification_completion_20261011/latest_combined_execution.json)
- [仅地面](../output/verification_completion_20261011/latest_floor/result.json) / [退出收据](../output/verification_completion_20261011/latest_floor_execution.json)
- [原始 peer monitor](../output/verification_completion_20261011/latest_peer/result.json) / [退出收据](../output/verification_completion_20261011/latest_peer_execution.json)
- [128 环境恢复与更新](../output/verification_completion_20261011/latest_resume/result.json) / [退出收据](../output/verification_completion_20261011/latest_resume_execution.json)
- [Social Pop-Art CUDA XML](../output/verification_completion_20261011/popart_cuda_final.xml) / [退出收据](../output/verification_completion_20261011/popart_cuda_final_execution.json)

单 GPU 测试不能证明多 GPU DDP 或多机训练兼容；短训练不能证明长期收敛、论文效果或全部地图导航质量。

## 4. 实际覆盖率

最新合并覆盖率包含完整 CPU 父进程、7 项真实 CUDA 补测、三种原生接触模式和 128 环境恢复更新。实际执行的本项目源码 SHA 与各运行开始时记录逐文件核对，结果一致。[一致性结果](../output/verification_completion_20261011/coverage_source_consistency.json)

共 604 份 src/scripts 文件，70,161 条可执行语句，语句覆盖 **62.35%**，分支覆盖 **52.82%**。coverage 默认综合值不是纯行覆盖率。

| 范围 | 文件 | 可执行语句覆盖 | 分支覆盖 |
|---|---:|---:|---:|
| `src/` | 458 | 38398/53853（71.30%） | 9740/16462（59.17%） |
| `scripts/` | 146 | 5345/16308（32.78%） | 1438/4702（30.58%） |

AST 对应 4,554 个函数范围，其中 1,303 个没有执行到函数体语句。该统计不是一函数一功能的验收，也不能将进入函数等同于全部功能通过。

分母不包含测试代码、SDK、third_party、docs/data/tools/examples；pytest 子进程和无法映射回源码的动态编译文件不计入本轮覆盖率。coverage 的源映射警告保存在日志中，没有通过补零、隐藏警告或改分母宣称 100%。与上一报告仅 src 的 69.14% 不能直接比较总百分比。

[逐文件 HTML](../output/verification_completion_20261011/coverage_html/index.html)、[逐文件 CSV](../output/verification_completion_20261011/file_coverage.csv)、[逐函数缺失语句](../output/verification_completion_20261011/function_coverage.json)、[原始 coverage JSON](../output/verification_completion_20261011/coverage.json)、[覆盖率导出日志](../output/verification_completion_20261011/coverage_export_complete.log)

## 5. 仍未完成的功能

原 354 个历史失败节点中，**254 个原节点现在通过，98 个仍未通过**；另外 2 个旧路线 bit 节点已由上述几何契约测试替代。没有原 354 清单之外的新失败节点。所有失败和初始化错误保留在原始 XML，没有改成 xfail，也没有跳过它们以取得绿色结果。

| 未通过测试文件 | 节点数 |
|---|---:|
| `tests/test_native_execution_bridge.py` | 8 |
| `tests/test_native_paired_initial.py` | 2 |
| `tests/test_native_seed_protocol.py` | 3 |
| `tests/test_native_shared_social_evaluation.py` | 6 |
| `tests/test_native_tool_owner.py` | 8 |
| `tests/test_shared_social_episode_after_recovery.py` | 10 |
| `tests/test_shared_social_episode_comparison.py` | 1 |
| `tests/test_shared_social_episode_evaluation.py` | 12 |
| `tests/test_shared_social_episode_native.py` | 9 |
| `tests/test_shared_social_episode_reference_entry.py` | 17 |
| `tests/test_shared_social_evaluation_preparation.py` | 1 |
| `tests/test_shared_social_minimum_horizon_entry.py` | 15 |
| `tests/test_shared_social_recovery_evaluation_resume.py` | 6 |

这些不是可以忽略的标签：

1. episode-reference / minimum-horizon 入口仍依赖冻结的训练入口、World、观测和旧 SDK 源码身份。需要完成这些物理 World 与整条训练/审计流程的新版本迁移和实际运行，当前后续功能未验证。
2. 恢复评估、episode 串行发布及 owner 链仍有旧路径及源码身份绑定；目前会在旧日志/计划 admission 处退出，后面的评估不能认定通过。
3. native156 固定矩阵、执行桥和 tool owner 仍绑定旧协议源；部分场景注册还需要精确的旧 probe snapshot。没有替换其原始地图证据或擅自改写矩阵。
4. native / paired / seeded 冷入口和 full22 入口仍拒绝变化后的 deployment/World 来源；元数据预检未通过，因此本轮没有证明完整四场景、200 秒 paired 导航和 full22 评估可用。
5. 所有语句/分支覆盖、所有平台/GUI/容器构建、所有数据资产和长期训练未完成验证。故不能给出“每行验证完成”或“没有任何 bug”的保证。

[完整未通过节点及堆栈](../output/verification_completion_20261011/remaining_failures.json)、[可复跑节点列表](../output/verification_completion_20261011/remaining_nodes.txt)、[全部节点最新观察](../output/verification_completion_20261011/test_observations.json)、[机器汇总](../output/verification_completion_20261011/validation_summary.json)

## 6. 重现命令

在项目根目录、上述固定环境中运行。`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于隔离主机 ROS 等外部 pytest 插件，不筛选项目测试；测试目录由 pyproject.toml 确定。

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=output/isaaclab61_reaudit_20261011/optional_rvo2:src:. \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q --tb=short
```

CUDA 节点列表见 `output/verification_completion_20261011/gpu_nodes.txt`，另有新 Social CUDA 节点：

```bash
CUDA_VISIBLE_DEVICES=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. \
  .venv/bin/python -m pytest -q \
  'tests/test_social_popart_sim61.py::test_both_adams_resume_preserves_CPU_runner_rng_and_updates_all_networks[cuda]'
```

物理验证输出必须选择尚不存在的新目录，分别运行：

```bash
.venv/bin/python scripts/validate_contact_lifecycle.py --mode combined --output output/my_contact_combined
.venv/bin/python scripts/validate_contact_lifecycle.py --mode floor --output output/my_contact_floor
.venv/bin/python scripts/validate_contact_lifecycle.py --mode peer --output output/my_contact_peer
.venv/bin/python scripts/validate_parallel_amp.py \
  --checkpoint output/isaaclab61_reaudit_20261011/amp128_cold_verified/last.pt \
  --output output/my_parallel_resume --steps 32
```

实际命令、开始时间、源码 SHA、耗时及退出码都在各 `*_execution.json` 中。全量 CPU 原始记录为 [XML](../output/verification_completion_20261011/final_cpu_complete.xml)、[日志](../output/verification_completion_20261011/final_cpu_complete.log)、[父进程收据](../output/verification_completion_20261011/final_cpu_complete_execution.json)；退出 1 与 98 个未通过节点一致。
