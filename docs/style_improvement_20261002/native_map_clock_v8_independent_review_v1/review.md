# 原生地图 v8 时钟修正：独立只读复核

## 结论

未发现已证实的阻塞缺陷。独立 CPU 检查与四份真实 USD 的复核通过；**真实 Kit Timeline 修复尚未验证**，不能据此宣布四场景原生地图全部通过。本次只新增本目录的审查证据，未修改 proposed、生产代码、原场景、训练环境，也未执行合并或重试被拒绝的生产写入。

审查对象是 [有界合并清单](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_map_clock_v8_proposal/merge_manifest.json)、[完整 staged patch](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_map_clock_v8_proposal/proposed.patch) 和 proposed 内的 9 个文件，总计 **245,852 B**。producer 精确 SHA256 为 `20173101ada3e5137d962abcdea574497e9624be7659984caa9869e7932d537a`，与 v8 派发常量一致。

## 已核实的边界

| 范围 | 独立证据 |
|---|---|
| 原 USD 文件 | producer 没有 USD `Save` 调用；三个 USD `Export` 接收者全部是 `GetSessionLayer()`，输出为独立审计目录中的 overlay/session 文件。完整合并清单只允许仓库内 9 条固定路径，拒绝链接、路径穿越、既有文件 SHA 变化；不包含原 USD、SDK、数据或冻结训练目录。 |
| session 所有权 | [helper](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_map_clock_v8_proposal/proposed/scripts/probe_paper_scene_maps.py:434) 在任何时间写入前要求当前 EditTarget 为 session，写入后检查同一目标、原根层 RAM 字符串与 TCPS；查询前再次检查 session 目标和根层 RAM SHA。 |
| 时间单位与顺序 | 先 session `SetStartTimeCode(0)`、`SetEndTimeCode(max(end,TCPS))`，再 Timeline 秒单位 setter、禁自动更新、设当前秒为 0；之后 reset 再显式设 0，执行原有静态快照步。查询前记录真实 getter，而非用请求值替换观测值。TCPS 不改。 |
| 实际时钟门禁 | 仍拒绝负时刻；查询起始秒必须恰为 0，完整查询前后 submitted counter 与 Timeline 秒必须不变。真实计数是 SDK submitted-step counter，不能表述为独立 PhysX 引擎时钟。 |
| v8 新准入 | [session 证据校验](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_map_clock_v8_proposal/proposed/CrowdSim/humanoid/runtime/native_map_v8.py:24) 要求精确 producer SHA、原范围声明、session 的 0 起点/结束公式、查询时实际范围/TCPS、根层身份及 RAM 哈希一致，还复核实际导出 session 的 SHA 与全局时间头。缺失、矛盾证据拒绝。 |
| v5/v7 保留 | 移除仅新增的 v8 import/派发后，provider 与 grid 整文件 AST 与 before 完全相同。`verify_v8` 移除仅 session 额外门禁、版本返回字段后，全部 v7 raw4/5/6、固定布局、Office API 缺项、padding 裁剪、全高度/地板 CSR、未知格、原生 0 时钟数学 AST 完全一致。旧 `native_map_v7.py` 本体未改；历史 v7 reader 改为核对原不可变归档，避免将新的 mutable checkout 当成旧历史源码。 |

## 四份真实 USD 的独立 CPU 结果

使用 standalone USD **26.08**；没有导入 Kit、IsaacLab、IsaacSim 或 Torch。Timeline 是明确标注的 setter 测试替身。

| 场景 | 原 start/end/TCPS | session start/end/TCPS | 默认与 t=0 的 Xformable 数量 |
|---|---|---|---:|
| Office | −1 / 0 / 60 | 0 / 60 / 60 | 4,097 |
| Warehouse | 0 / 0 / 60 | 0 / 60 / 60 | 1,442 |
| Full_Warehouse | 0 / 100 / 60 | 0 / 100 / 60 | 9,132 |
| Hospital | −1 / 0 / 60 | 0 / 60 / 60 | 1,842 |

当前 proposed helper 在四场景运行前后，原根文件 SHA、根层 RAM、默认和第 0 帧全部路径及矩阵字节哈希均未变；共核对 **16,513** 个 Xformable，两个时刻分别核对。原 candidate 及最新 proposed 的完整四场景 CPU 证明另记录了 **2,740** 个 source layer 文件的前后完整 SHA。这份独立复核重新验证根文件和变换，并引用已封存的全 layer 证明，没有把它称为新一轮全部 layer 文件重哈希。

最终独立脚本 **68 项检查全部通过**，详见 [verification.json](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_map_clock_v8_independent_review_v1/verification.json)。作者 staged 门禁的 37 个 CPU 用例保留于 proposal `tests.log/tests.xml`，包括旧 v5/v7 输出和 v8 损坏证据拒绝；这些属于 CPU/合成准入测试，不是原生地图实跑。

## 仍待真实验证的事项

1. 安装 SDK 的 [_timeline.pyi](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.timeline-1.1.0+6312fa25.lx64.r.cp312/omni/timeline/_timeline.pyi:232) 明确 start/end setter 会写当前 stage，单位为秒，并要求范围至少一帧。Python stub 没有证明 native 实现总是尊重当前 EditTarget；假 Timeline 也无法证明。新 root-RAM/目标门禁应保留，真实 v8 若发生 RAM/目标/时钟矛盾，仍需拒绝，不得以 CPU 证明替代。
2. helper 的运行时根层 RAM 基准从 **helper 开始到实际查询前**，不是从 `open_stage` 开始覆盖整个 `SimulationContext` 初始化。报告与字段解释应保留此时间边界。原文件不写入、全部 source layer SHA 前后检查是独立合同。
3. `source_time_range.USD_time_range_modified=false` 表示原来源层/文件保留；运行时 composed stage 范围确实被 session 覆盖，已由 `runtime_time_range_override` 显式记录，不能写成整个运行 stage 的时间范围未变。
4. 原 v7 的 Warehouse/Full_Warehouse 查询时 counter=2、Timeline=0；Office/Hospital 实测 counter=2、Timeline=−1/60 秒而失败。所有旧结果保持原样。当前没有 v8 原生执行结果，不能回填旧 Office/Hospital 为通过，也不能提前扩大地图或导航质量结论。
5. 合并工具为每文件原子替换，九文件整体不是跨文件事务；本目录只核对其默认只读验证、路径/字节边界与 before 备份，不执行 `--apply`。后续应用由 root 对具体 patch 做审批与新源码登记，既有运行按自身 prelaunch source archive 归属。

## 复核命令

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -B \
  docs/style_improvement_20261002/native_map_clock_v8_independent_review_v1/verify_read_only.py
```

复核只读 proposed/当前业务/真实 USD，输出限本审查目录。没有执行 producer 的 `--execute-physx` 或 merge 的 `--apply`。
