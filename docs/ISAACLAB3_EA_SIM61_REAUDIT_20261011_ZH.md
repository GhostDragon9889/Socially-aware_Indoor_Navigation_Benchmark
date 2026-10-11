# IsaacLab 3.0 EA / Isaac Sim 6.1 增量复审与并行训练验证

> 后续未筛选全量回归已更新：4645 项通过、98 项未通过、2 项跳过；不构成全功能验收。请以[第二轮完整回归报告](VERIFICATION_COMPLETION_20261011_ZH.md)及其原始收据为准。下文保留此前轮次的范围和结果。

日期：2026-10-11。本轮保留工作区原有迁移修改，在当前代码上重新运行检查、复现缺陷并修复；没有修改两个参考 SDK，也没有提交或推送。

## 目标与证据边界

实际解释器为项目 `.venv/bin/python`，指向项目的 Sim6.1 overlay；SDK 为 `/project/EXP/IsaacLab`。实际核验 IsaacLab 包17.0.2、Isaac Sim6.1.0.0、Torch2.11.0+cu128、Warp1.16.0、NumPy2.5.1、usd-exchange2.3.0；169个USD RECORD文件、USD保存/重新打开检查通过。机器只有一张 RTX4080SUPER 16GiB，驱动595.99.02。

依据为[官方3.0迁移指南](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/migration/migrating_to_isaaclab_3-0.html)、[Sim6.1发布说明](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html)及两个指定本地 SDK 的真实源码。迁移指南要求明确处理 XYZW、ProxyArray 和索引/掩码写接口；本轮进一步核对实际实现，未仅凭名称判断兼容性。

本轮的静态清单覆盖源码文件及行数；人工审查重点为仿真适配、资产转换、控制诊断、训练目标、并行统计和检查点恢复。**AST或lint通过不等于逐行行为覆盖；本报告不承诺零bug，也不声称所有功能已完成人工逐行审查。**

证据目录：[`output/isaaclab61_reaudit_20261011`](../output/isaaclab61_reaudit_20261011/)。前次报告见[迁移报告](ISAACLAB3_EA_SIM61_MIGRATION_20261011_ZH.md)，其中四场景、录像和旧覆盖率属于前次测试，不能算成本轮新增实测。

## 本轮已复现并修复

| 范围 | 缺陷与修正 | 验证 |
|---|---|---|
| PhysX根节点速度 | 两参考SDK的articulation link setter将link速度交给COM接口；显式计算`v_com = v_link + ω × offset_world`，通过公共COM setter写入，重置与速度扰动共用；用根姿态旋转局部COM偏移，避免大世界坐标相减丢精度 | 非零COM/角速度/乱序subset回归先失败；补充90°旋转、刚体/关节形状与1e6平移，旧实现误差0.0875；最终128环境恢复根速度误差2.98e-8 m/s |
| 质量/惯量随机化 | 写mass后首次访问惰性default_mass，比例变成1；改为写入前读取当前body_mass/body_inertia | 原36/54惯量分量不符，修复通过 |
| USD接触路径 | 刚体就是default prim时得到非法`/Robot/`；保留合法`/Robot` | 使用真实OpenUSD路径回归 |
| MJCF缓存 | 同一显式输出目录不同配置互相覆盖，旧完成标记仍误报命中；按实际产物记录当前key并在锁内验证，失败前撤回标记；删除产物后重新转换 | False→True→False配置、跨缓存实例、并发转换、产物删除回归 |
| GPU训练恢复 | `torch.load(map_location=cuda)`将CPU采样器RNG也移入CUDA，`Generator.set_state`报错；在CPU解包，模型/优化器/replay由各自加载器迁到目标设备 | 实际GPU回归先出现ByteTensor错误，修复后通过；GPU物理恢复并继续更新 |
| PointGoal PPO | 超时后自动reset丢失终态，价值bootstrap被当作终止置零；保存终态观测，在timeout奖励加入其折扣价值，GAE仍截断episode边界 | 定向目标原应为6却为1；混合环境与真正终止回归通过 |
| 分布式PPO EMA | 原仅取rank0样本更新EMA；合并所有rank均值/方差/样本数后广播 | 真实双进程Gloo/DDP，rank0=1 env、rank1=3 env，两次梯度更新及参数同步 |
| 归一化边界 | 标量RunningMeanStd使用切片导致0维错误；float32计数在2^24后丢整数精度 | 标量EMA/累计统计和大计数回归 |
| 训练数值与指标 | 关闭梯度幅度检查时漏检Inf；单元素张量与rank权重广播导致加权平均错误 | Inf梯度及预期2.5却得到4.0的回归 |
| 控制诊断入口 | 旧WXYZ解释、直接切ProxyArray、不存在的controller导入、reset全零四元数、箭头转义序列误当退出 | 真实CPU Warp ProxyArray和键盘解析测试，两个CLI help实际运行 |
| 冷转换进程退出 | MJCF首次转换后，事件分发器原生清理capsule延迟到完整App.close后的CPython模块析构阶段才释放并崩溃，训练成功仍SIGSEGV；在完整App.close之前释放其两个已核验Python字典引用 | GDB定位、原始崩溃及两种无效处理真实退出139、成功与失败训练路径退出码分别0/1；25项生命周期CPU测试（含夹具隔离回归） |
| 容器安装命令 | 两个Dockerfile的未引用`torch>=2.2`被Shell解析为`>`文件重定向；补充引号 | 真实`/bin/sh`隔离参数试验：修复前参数截为`torch`并创建`=2.2`文件，修复后完整传参且无重定向文件；未构建镜像 |

主要修改位于`src/protomotions/simulator/isaaclab/`、`src/protomotions/agents/`、`src/CrowdSim/humanoid/training/trainer.py`、`src/CrowdSim/tools/`、`src/crowdsim_ppo/`，并扩展相邻回归测试。新物理验证入口为[`scripts/validate_parallel_amp.py`](../scripts/validate_parallel_amp.py)。

## 测试分母与当前结果

| 检查 | 本轮实测 |
|---|---|
| Python静态清单 | 1,726文件、378,513物理行，包含701个third_party文件；AST无错误，F821/F822/F823/F901/E9无发现；保留既有有说明的per-file/inline豁免 |
| JSON/YAML/TOML与workspace配置 | 126份语法解析通过 |
| Shell语法 | 16份脚本通过`bash -n`；不是实际执行或部署验证 |
| 未筛选完整pytest基线 | 4,277 passed、276 failed、80 errors、14 skipped，61 subtests；原始失败保留 |
| 失败分类 | 354项与已有历史清单精确匹配；另外2项为沙箱本机网络受限，Gloo复测通过 |
| 最终当前CPU清单（启用分支覆盖率） | **4,330 passed、8 skipped、61 subtests passed**，207.75秒，父进程退出0 |
| CPU跳过项的真实CUDA补测 | **6 passed**，48.56秒；包括GPU恢复、Newton/FK、接触和随机化 |
| 当前清单按节点合并 | **4,336 passed、2 skipped**；剩余2项为旧IsaacGym，不宣称在Py3.12可运行 |
| 最终collection | 4,692节点＝当前4,338＋历史354；与起始清单相比新增17节点、移除1个被四组参数化替代的旧节点，净增16 |

当前清单仅排除复审开始前已有的354项历史失败，未为本轮失败新加排除项。独立复核细分为：303项冻结源码/SDK/协议身份、49项历史产物/checkpoint/owner-plan身份、2项旧路线浮点bit快照；其中80项在setup阶段退出，没有执行后续行为。**直接对全仓运行pytest仍不是零失败**。历史测试未删除、未改成xfail，也未把历史检查点或实验结果改写为Sim6.1结果。

两个路线快照因当前防穿角修改而改变；独立复核现行端点、索引、到达标志和输入不变性仍通过。24个checkpoint身份失败及2个owner-plan失败涉及旧仓库绝对路径绑定，不能据此说数据字节损坏。这些历史读取/评估入口在新路径或新SDK下仍未恢复，不能以“历史”分类冒充功能通过。逐文件分母见[历史失败独立复核](../output/isaaclab61_reaudit_20261011/historical_review.md)。

证据：[最终CPU XML](../output/isaaclab61_reaudit_20261011/coverage_cpu_accepted.xml)、[CUDA XML](../output/isaaclab61_reaudit_20261011/gpu_six.xml)、[未筛选XML](../output/isaaclab61_reaudit_20261011/full_cpu.xml)、[失败分类](../output/isaaclab61_reaudit_20261011/failure_classification.json)、[当前节点](../output/isaaclab61_reaudit_20261011/current_nodes_final.txt)、[源清单与SHA](../output/isaaclab61_reaudit_20261011/inventory.json)、[配置清单](../output/isaaclab61_reaudit_20261011/configuration_syntax.json)。

CPU完整测试通过独立PYTHONPATH加载已验证SHA的ORCA二进制，不修改SDK依赖。专项测试数量与完整清单有重叠，不能相加。 JUnit suite总数包含61个subtest，以上统计按唯一testcase节点计算，不重复相加。最终[机器可读验收汇总](../output/isaaclab61_reaudit_20261011/validation_summary.json)校验了CUDA补测对应CPU跳过项、全部原生实际退出码及生产源码哈希。

完整回归还发现并修复生命周期测试夹具污染：其他测试的收集过程导入真实Carbonite模块，旧夹具误触原生清理对象，下一测试正确拒绝缺少释放凭据。夹具现在临时隔离并原样恢复该模块，新增预载绑定回归；不放松生产检查。修复前普通运行和coverage运行的各1项失败均保留，最终同清单无失败。见[隔离问题诊断](../output/isaaclab61_reaudit_20261011/lifecycle_collection_pollution_findings.md)。

## 本轮逐文件和函数覆盖率

使用coverage.py 7.15.2重新测量五个`src`包的CPU父进程，启用branch tracing；不把此前报告覆盖率、GPU执行、子进程、scripts或third_party拼接成当前覆盖率。范围为455份源码，53,543条可执行语句、16,318个分支；语句覆盖**69.14%**、分支覆盖**57.29%**。coverage默认的综合百分比为66.37%，不能将它误写成纯行覆盖率。

| 包 | 语句覆盖 | 分支覆盖 |
|---|---:|---:|
| CrowdSim | 58.85% | 49.15% |
| protomotions | 83.72% | 72.11% |
| crowdsim_ppo | 78.20% | 61.20% |
| social_navigation | 40.47% | 28.93% |
| deployment | 28.57% | 15.81% |

共有3,886个函数范围，其中1,068个在该CPU测量中未执行任何语句；执行了部分语句也不等于功能充分验证。真实GPU已经覆盖部分此处未计数的仿真路径，但没有测量就不增补百分比。两份带`:relocated_checkout`/`:single_asset`后缀的动态编译虚拟文件没有磁盘源码，导出使用`-i`并完整保留警告；不把它们视为100%覆盖。

证据：[覆盖率汇总](../output/isaaclab61_reaudit_20261011/coverage_summary.json)、[逐行HTML](../output/isaaclab61_reaudit_20261011/coverage_html/index.html)、[逐文件CSV](../output/isaaclab61_reaudit_20261011/coverage_files.csv)、[逐函数缺失行/分支](../output/isaaclab61_reaudit_20261011/coverage_functions.json)、[原始coverage JSON](../output/isaaclab61_reaudit_20261011/coverage_cpu.json)、[导出警告](../output/isaaclab61_reaudit_20261011/coverage_export.log)。这个分母直接说明目前不具备“每行、每分支、每功能全部测试通过”的证据。

## 真实物理并行训练与恢复

- 32 env ×32 rollout ×12 updates：**12,288物理转移**，actor/task encoder/critic/discriminator均实际更新，全部2,848训练专家clip采样；完整关闭，进程退出0。
- 32 env GPU恢复：六类根/关节状态写回并读取；16个反序偶数env重置后，另16个env的物理量/history/curriculum保持；继续1次PPO+AMP更新，新增1,024物理转移、8次timeout、0次真正终止；四网络更新，退出0。
- 128 env ×32 rollout ×12 updates：**49,152物理转移**、436次reset、0次跌倒，全部2,848训练专家clip采样。首次训练完成后进程SIGSEGV/139，保留为失败证据；cold-cache GDB定位到`carb.eventdispatcher`在CPython模块析构阶段崩溃。修复后强制使用全新转换缓存，完整训练及关闭成功，父进程确认实际退出0。
- 128 env GPU恢复：六类根/关节状态最大误差分别为0、0、2.98e-8、0、0、0；64个反序偶数env重置状态与独立源动作一致，另64个env的物理量/history/curriculum不变，空集合重置无副作用。继续第13次更新，新增**4,096物理转移**、26次timeout、0次真正终止，四网络参数均变化，replay正常使用，完整关闭且进程退出0。

冷转换修复位于`SDK61Lifecycle`，属于针对已检查二进制SHA的终止兼容处理，**不是公开SDK API**。它只在停止物理、释放视图后，检查capsule类型、两个拥有者字典身份和引用数，然后在完整App.close之前调用其原生析构。公开别名缺失、额外拥有者、不同二进制等异常保留为清理失败；同实例幂等调用要求已成功释放的模块身份、路径和SHA匹配。仍调用完整官方`App.close(skip_cleanup=False)`，不强制退出、不改SDK、不把训练失败改成成功。SDK更换后需要重新验证，不能直接复用该处理。GDB确认了关闭后的析构时序与崩溃位置，具体插件卸载先后没有单独证实。原生诊断详见[关闭问题证据](../output/isaaclab61_reaudit_20261011/runtime_review/native_shutdown_findings.md)。

最终生产源码再次完成[128冷训练](../output/isaaclab61_reaudit_20261011/amp128_cold_verified/result.json)和[128恢复](../output/isaaclab61_reaudit_20261011/resume128_verified/result.json)，父进程收据分别确认[退出0](../output/isaaclab61_reaudit_20261011/amp128_cold_verified_execution.json)、[退出0](../output/isaaclab61_reaudit_20261011/resume128_verified_execution.json)。另外故意只训练1 update ×8 rollout，专家覆盖不达标，仍完成原生清理并[真实退出1](../output/isaaclab61_reaudit_20261011/cold_negative_verified_execution.json)，[原始训练错误保留](../output/isaaclab61_reaudit_20261011/cold_negative_verified/result.json)，无segfault。这是预期失败路径验收，不计为训练成功。

恢复验证不保存或恢复PhysX求解器内部cache，不宣称逐步物理轨迹bitwise相等。新验证脚本限制普通AMASS奖励配置，读取保存的task horizons；输出`resumed_optimizer.pt`为模型/优化器等训练状态，不包含完整physical runner，不能冒充再次物理恢复的检查点。

训练时更新轮数、有限损失、网络参数变化、专家覆盖和真实退出码分别检查；短训练只能验证执行正确性，不能证明收敛、策略质量、动作自然度或论文指标。

## 复现入口

在项目根目录、当前固定SDK与资源齐备时，使用新的输出目录：

```bash
.venv/bin/python -I scripts/isaaclab3_environment.py --profile isaaclab3_ea_sim61 check
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  PYTHONPATH=output/isaaclab61_reaudit_20261011/optional_rvo2:src:. \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  @output/isaaclab61_reaudit_20261011/current_nodes_final.txt
PYTHONPATH=src:. .venv/bin/python scripts/finetune_pacer_amass.py \
  --output output/new_amp128 --updates 12 --num-envs 128 --rollout-steps 32 \
  --task-horizons-s 4.5 \
  --parent-checkpoint data/pretrained_models/local_amass_finetune/walking_turn_20261009/Humanoid_AMASS.pth
PYTHONPATH=src:. .venv/bin/python scripts/validate_parallel_amp.py \
  --checkpoint output/new_amp128/last.pt --output output/new_resume128
```

普通终端须可访问GPU与本机进程间socket；沙箱中的CUDA缺失或Gloo地址解析失败不能直接当作产品缺陷，也不能当作已通过。缺少ORCA测试提供者时跳过数会不同。

## 未覆盖或尚不成立的结论

- 多GPU/NCCL、多节点、其他GPU/驱动、长时间训练及收敛，本机单GPU验证无法替代。
- 四个室内场景、RTX完整录像、外部NavDP/FLUX服务与全部可选后端没有在本轮全部重跑；前次ThirdFloor过度制动、SecondFloor速度不足的质量问题仍不能标为解决。
- 354项历史测试在当前迁移源码、路径或规划行为下不兼容，部分历史读取/评估CLI仍不可用，整仓零失败不成立。
- 本轮没有做到每条语句/每个分支实际执行，也没有对所有第三方实现进行人工逐行审查；清单和静态检查只是明确覆盖范围。
