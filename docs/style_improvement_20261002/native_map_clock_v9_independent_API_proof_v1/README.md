# V9 匿名 runtime root：独立 API 与真实 USD 证明

## 结论

**58 项 CPU 检查通过；尚未执行 V9 Kit / PhysX。** 当前安装源码支持“单个匿名 runtime root + 原 USD 绝对路径 sublayer + 自有 session”的接入方式。原文件层不必成为 Kit 当前 stage 的 root，也不必复制、展平或保存。

四个真实场景的原 root 文件 SHA、原 RAM `ExportToString()`、原时间范围均保持不变；包装后及主动写入自有 root 元数据后，16,513 个 Xformable 的 default / t=0 变换逐字节相同。这里写入自有 root 是 CPU 所有权检验，**不是对 V8 异步 root 写入原因的复现**。

证据：[proof.json](proof.json)、[可复现工具](verify_anonymous_root.py)、[执行日志](proof.log)。测试使用独立 OpenUSD 26.08，未导入 `omni`、`isaacsim`、`isaaclab`、`torch`，未创建 App / 原生 Context。

## 安装 SDK 的实际 API

| 入口 | 当前源码行为与边界 |
| --- | --- |
| `isaaclab.sim.utils.stage.create_new_stage()` | `Usd.Stage.CreateInMemory()`；设置 Lab thread-local，并插入 `UsdUtils.StageCache`。位置：`stage.py:159`。 |
| `open_stage(anonymous_identifier)` | 真实 OpenUSD 支持 `anon:…:tmp.usda`；该 helper 设置 Lab thread-local。位置：`stage.py:223`。它本身不会调用原生 Omni Context attach。 |
| `SimulationContext(...create_stage_in_memory=False)` | 使用已准备的 thread-local stage，确保 cache ID，然后调用 `kit_context.attach_stage_with_callback(stage_cache.GetId(self.stage).ToLongInt())`。位置：`simulation_context.py:118–146`。 |
| `omni.usd.UsdContext.attach_stage_with_callback(stage_id, callback=None)` | 安装 stub 明确为异步 attach，ID 为 Python `int`。位置：`_usd.pyi:516`。未定义 `attach_stage`。 |
| `get_current_stage_id()` | 获取 / 插入同一个 cached stage 并返回 `ToLongInt()`。位置：`stage.py:555`。四场景真实 cache 查找均返回同一 stage。 |
| `SimulationContext.clear_instance()` | physics manager 先关闭，再调用 stage 清理；`stage.close_stage()` 先关 OmniContext、再清 cache / thread-local。位置：`simulation_context.py:813`、`stage.py:409`。 |

上述文件的绝对路径、字节数与 SHA 完整记录在 `proof.json.API_primary_sources`。安装 helper 文档提到 `attach_stage_to_usd_context`，实际模块中没有定义该函数，不能据此调用。`resolve_paths` 仅适用于复制层内容后改锚点，本方案保留原层，因此不需要它。

## 必须显式复制的 stage 元数据

单纯把原 USD 放入匿名 root 的 sublayer 后，四个场景都得到 **Y-up、0.01 m/unit、24 TCPS / FPS、0..0 时间范围**。原 source 的 stage 元数据不会自动成为匿名 root 的元数据。

在 attach 前必须在**自有匿名 root** 设置原 `upAxis`、`metersPerUnit`、`timeCodesPerSecond`、`framesPerSecond`、有效 `defaultPrim`；时间范围显式设为 `0..max(original_end, original_tcps)`。切换 session 后再做碰撞、PhysicsScene 与 query 时间覆盖。Office / Hospital 原范围 `-1..0 @60`，Warehouse `0..0 @60`，Full_Warehouse `0..100 @60`；原来源范围要独立保留，不能被 runtime 范围覆盖。

只读审查中的初版 staged helper 已复制 up-axis / meters / TCPS，但尚未复制 FPS / defaultPrim；已通知业务 owner 补全。此报告不修改候选或生产代码。

## 建议的最小调用顺序

1. 只读 `Sdf.Layer.FindOrOpen(original_absolute_path)` / preview stage，保存原 source RAM、原时间范围、全部原层文件身份和变换证据。
2. 使用公开 `create_new_stage()`；确认 fresh root 为 anonymous。只设置一个绝对路径原 source sublayer，并在 root 写入上述元数据。
3. 设置 edit target 为自有 session。保留所有原路径，在 session 写入既有碰撞与 PhysicsScene 覆盖。
4. `SimulationContext` 保持 `create_stage_in_memory=False`，避免新建空 stage 替换已准备的 stage。依赖 SDK 现有 attach，不另建 Context、不二次手动 PhysX attach。
5. 在自动 physics 更新关闭、无显式 step 的有界 App 更新中等待 OmniContext cache ID / root / session 与准备 stage 一致；记录真实 submitted counter。这里仍需真实 SDK 运行确认异步 attach 是否成功。
6. 设置固定 query 时间与 session 范围，检查原 source RAM 和全部原层文件仍相同。原匿名 root 可以拥有自己的元数据变化，但必须单独记录、保持 anonymous / 原 sublayer / session 所有权。
7. 查询前后均核对实际 range、time / submitted counter、原 source RAM、source 文件 / 层、default / t=0 变换与原始地面所有权门禁。清理遵循现有 SDK 顺序。

## 版本与证据边界

V5 / V7 / V8 现有身份和历史结果保持不变。V9 应新增 `original_source_*` 与 `owned_runtime_root_*` 的独立身份，不能把匿名 root 伪称原 file-backed root，也不能让原 source 范围变成 runtime 范围。

本工具重核四个原 root 文件及其 RAM / 变换；未重复重 hash 全部 2,740 个原层文件。完整层清单属于候选 owner 的独立证明。CPU helper 检验使用安装函数原 body，但刻意禁用 Kit context-sync；它证明 USD / cache API 行为，不能替代实际 Kit callback、native query、完整地图导出验收。V8 三场景失败与 Full_Warehouse 通过保留，不据此宣称异步改源原因已经证明。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python -B \
docs/style_improvement_20261002/native_map_clock_v9_independent_API_proof_v1/verify_anonymous_root.py
```
