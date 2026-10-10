# 独立 all-temporal-anchor view：供 root 审阅

只改变可抽取的窗口集合及其身份。原 states/maps、场景 split、几何、五状态 resampling、邻居排序/半径、history/future、NN/EMA/loss 不变。本计划阶段没有读取 NPZ、执行转换、NN、SDK 或 GPU。

## 固定作者依据

TRACE commit `594b2af0824c5853e1a1884e167cd5ed097dd81c` 的 UnifiedDataset 按 ego history/future 建 agent-time index；15m 只决定取样后的条件邻居，不筛掉零邻居。AgentBatchElement、collation、parse_batch、history encoder 都明确处理零邻居。逐文件固定 Git blob 证据见 `../TRACE_zero_neighbor_source_research_v1/primary_bindings.json`。

## 最少自有文件

1. `build_all_temporal_view.py`：读取两份原 frozen manifest，核对原 episode/map SHA 和 readonly，读取已有 states/times；逐 episode 写独立 `[agent, anchor] int32` NPY，只列所有 `agent=0..N-1, anchor=30..47`。保持原场景/episode 顺序。新 manifest 引用原绝对 folder、原 episode/map bytes/SHA，新增 window_index 记录；不复制或写原 NPZ/map。
2. `all_temporal_corpus.py`：私有加载原 Interaction/Maps/Mixed 源，准确应用小范围变更：读取独立 window_index、严格新 view 身份、允许 `0..19` 邻居、新来源/窗口身份。其余 numpy_batch、crop、padding、union 抽样与 RNG 保留原实现。每处替换 count=1，不使用不明确的 fallback。原源 bytes/SHA 和 exact delta 由 `source_derivation.json` 记录。
3. `check_all_temporal_view.py`：一次完整 CPU 数据检查及必要 numpy_batch 验证，不运行 Torch。与原 episode/map bytes、场景 split 对照；全窗口检查 history/future 边界、五状态 finite、邻居数量/全空 mask、地图占位和 geometry 属性不变；抽查原有窗口 batch 逐数组 exact，新零邻居 batch 真实 mask 全 false；union explicit indices 和 NumPy generator 保存/恢复 draw 相同。

生成的 `actual_QC.json` 记录每 split/domain/scene/agent/time 计数及 0..19 邻居 histogram、原窗口保留与新增窗口数、单人场景数。保留全部零邻居、静止和空地图场景；没有 collision/visible-map/速度/质量过滤。

## 新数据身份及训练 API

- stage：`docs/style_improvement_20261003/orca_all_temporal_view_staging_v1/`
- Maps：`data/diffusion_amp/orca_maps_all_temporal_1000_v1/manifest.json`
- Interact：`data/diffusion_amp/orca_interact_all_temporal_1000_v1/manifest.json`
- 原 domain manifest schema 保留以表示 states/maps 格式；新增必需 `view_schema=local_orca.all_temporal_anchor_view.v1`、`source_manifest={path,bytes,sha256}`、`window_selection`，loader 拒绝没有这些字段的旧 manifest。
- `MixedORCACorpus(maps_manifest, interact_manifest, split='train', *, device='cpu', seed=7, verify_hashes=True)`、`manifest_window_counts()`、`sha256()` 接口不变。
- `resume_identity()` 显式包含 view_schema、原 manifest SHA、新 manifest SHA、selection、window index 方案与 domain count。原 training_state 的 exact identity equality、全 union RNG 保存/恢复原样，旧 filtered checkpoint 不能作为此 view 的 exact resume。
- new view 各 episode 加 `window_index={path,bytes,sha256}`，`conditional_window_count=N*18`；episode 本身原 path/bytes/SHA、agent/radius metadata 不变。
- 混合采样仍 uniform union with replacement，不声称作者 shuffle 的 bitwise 次序。

## 基于实际 frozen manifests 的 split 计数（元数据，待 CPU 实证）

| split | Maps agents/windows | Interact agents/windows | union windows |
|---|---:|---:|---:|
| train | 4,522 / 81,396 | 8,425 / 151,650 | 233,046 |
| validation | 587 / 10,566 | 1,077 / 19,386 | 29,952 |
| test | 531 / 9,558 | 1,106 / 19,908 | 29,466 |
| all | 5,640 / 101,520 | 10,608 / 190,944 | 292,464 |

原 train 为 228,588，新增 4,458（all-temporal train 的 1.91285%）；原 all-split 为 286,857，新增 5,607。研究初版将 all-split agents 与 train windows 混用，得到 63,876 的错误估算；那份文件保留历史，新转换/QC不使用它。

原 corpus 一次读取 states/map 时验证原 sealed artifacts；新 manifest/window artifact 的 SHA 纳入新训练 metadata。新训练入口/40k plan 由 paper_method_next_iteration 负责，保留原主循环/EMA/Adam/NN，只接此 API 并替换数据/输出身份。

## 顺序

Root 先 whole-read 本文件和三个源文件；source-only peer 核对 exact delta。获得这次 source plan 确认后，才执行一次 CUDA 禁用的 CPU view 构建及数据 QC；失败保留日志，不修改 frozen 原文件。最终新 manifest SHA 和真实计数交 paper 写新独立40k plan。无未来效果、训练、模型权重或 bitwise physics 声明。
