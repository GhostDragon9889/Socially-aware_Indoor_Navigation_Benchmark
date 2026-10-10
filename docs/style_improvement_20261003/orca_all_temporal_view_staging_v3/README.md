# All-temporal data view：实际 CPU 完成

最终源码在本目录，3 个文件已 0444。新 view 只引用原 frozen states/maps，不重生成或重采样场景，不修改原 manifest、生产185、NN/EMA/loss 或默认配置。正式40k训练入口和计划由 `paper_method_next_iteration` 独立接线；本交付没有启动 NN、SDK 或 GPU，也不声明未来效果。

## 固定作者链

TRACE `594b2af0824c5853e1a1884e167cd5ed097dd81c`：

- [训练 datamodule](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/datasets/trajdata_datamodules.py#L44) 将15m用于条件邻居距离；[agent-time index](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/dataset.py#L392)仅按 ego history/future eligibility 建索引。
- [AgentBatchElement](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/data_structures/batch_element.py#L164)、[空邻居 padding](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/utils/trajdata_utils.py#L135)、[全空编码](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/models/trace_helpers.py#L160)均允许零邻居。
- 固定作者文件逐 Git blob 整 bytes 核验见 `../TRACE_zero_neighbor_source_research_v1/primary_bindings.json`；研究首版混用全 split agent 数与 train 窗口数的算术错误已在原 README 顶部明确更正。下面是本次真实转换计数。

## 实际计数

| split | Maps | Interact | union | 新增零邻居 |
|---|---:|---:|---:|---:|
| train | 81,396 | 151,650 | 233,046 | 4,458 |
| validation | 10,566 | 19,386 | 29,952 | 504 |
| test | 9,558 | 19,908 | 29,466 | 645 |
| all | 101,520 | 190,944 | 292,464 | 5,607 |

每 agent 的 anchors 从实际100帧、history31、future52合同推导为30…47；原邻居窗口全数组等于新窗口中 `neighbors>0` 的子集。保留全部原 scenes、agent radii、split、states、maps、空地图、静止窗口。Maps train 新增3,247、Interact train 新增1,211，全为真实零邻居窗口。

新 artifact 只有2份 manifest和2,000个独立 int32 NPY窗口索引，均0444，原 states/maps路径与 SHA 保持。最终数据位于：

- `data/diffusion_amp/orca_maps_all_temporal_1000_v2/manifest.json`：28,589,412 B，SHA `20ee0c92968659072806d59fce8a41e72fd323bacddec043e60222fc1800e8a9`。
- `data/diffusion_amp/orca_interact_all_temporal_1000_v2/manifest.json`：1,800,196 B，SHA `b9eab23129dc664a1ff6edb7e32e21e23db5cf6c68c32cfb23914ab44db8466d`。

## Root 可直接 whole-read 的最少文件

1. [all_temporal_corpus.py](all_temporal_corpus.py)：原3loader私有 exact delta；邻居允许0…19，history/mapcrop/padding/union数学保持原源。新 `resume_identity` 包含 view/schema、原/新 manifest SHA、selection/window storage，原 filtered exact resume 不匹配。
2. [build_all_temporal_view.py](build_all_temporal_view.py)：完整原 episode/map readonly+SHA、全窗口 subset/时间/邻居统计。
3. [check_all_temporal_view.py](check_all_temporal_view.py)：场景身份全对照，96个原窗口 batch整数组抽查一致；6个真实单人窗口全false neighbor mask，Maps空/非空和Interact缺图语义；private union RNG roundtrip与explicit selection不消费RNG。
4. [actual_QC.json](actual_QC.json)和[source_freeze_v3.json](source_freeze_v3.json)：真实结果与绑定。完整各 scene/agent/anchor/0…19 histogram见[actual_build_QC.json](actual_build_QC.json)。batch抽查不冒称全部raster crop像素已逐一重检。

真实 tool closure见[actual_CPU_closures.json](actual_CPU_closures.json)：build61706（8c0ae5→5f10ba）和QC64635（58e8d7→ee6c13），均exit0、CUDA隐藏、4线程，无Torch导入。

## 保留的失败

v1首次1059bb exit1：private Maps class value多一个C，import阶段，无数据输出。v2第二次91618/4064bb exit1：新guard误把原inner `val`与outer `validation`要求字面相等，已产生800个Maps train窗口，全部保留原位置。v3修正只采用精确 train/train、validation/val、test/test映射且换独立data v2路径；旧源码、日志、receipt及partial不覆盖。失败不是原数据错误，也没有因此改变任何窗口/NN数学。
