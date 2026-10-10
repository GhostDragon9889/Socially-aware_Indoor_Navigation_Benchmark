# TRACE 零邻居窗口：作者源码已确认

**计数更正（2026-10-03）：**下文初版的 63,876 / 21.84% 将全 split agent 数与 train window 数混用，保留为历史估算，不能用于训练注册。实际 frozen manifest 的 train 为 4,522 Maps + 8,425 Interact agents；all-temporal train 为 233,046，较旧 228,588 新增 4,458（1.91285%）。完整分 split 元数据见[更正 JSON](../orca_all_temporal_view_staging_v1/source_actual_split_counts.json)和[独立 view 方案](../orca_all_temporal_view_staging_v1/SOURCE_PLAN.md)。原 primary_bindings.json 保留初版记录；其中作者固定源码的零邻居结论不受此算术错误影响。

2026-10-03，只读研究。没有导入Torch/NN/SDK、读取NPZ轨迹或权重、运行转换/生成、修改现有数据/计划/生产185。固定TRACE作者commit `594b2af0824c5853e1a1884e167cd5ed097dd81c`，11个相关文件本轮逐一与Git blob比较，全部整bytes一致，见[来源及结论](primary_bindings.json)。本地作者缓存已足够，不需要以新版trajdata代替固定版本。

**结论：作者训练入口允许零邻居/单人窗口。当前至少一个15m真实邻居的筛选是本地额外子集规则，不是固定作者源码的必要条件。**现有冻结数据对其已经声明的neighbor-only合同仍是正确的；这项发现不证明恢复零邻居能改善模型或物理效果。

## 作者完整链路

| 证据 | 代码位置 | 含义 |
|---|---|---|
| ORCA生成 | [gen_dataset.py:318](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/orca_sim/gen_dataset.py#L318) | 人数从1至max_agents抽取，允许单人场景。 |
| 训练数据入口 | [trajdata_datamodules.py:44–80](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/datasets/trajdata_datamodules.py#L44) | agent-centric、固定history/future、pedestrian类型；15m传给`agent_interaction_distances`，无必须邻居的filter。单一UnifiedDataset包含Maps和no_maps。 |
| 建训练索引 | [dataset.py:392–410](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/dataset.py#L392)、[filters.py:35](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/filtering/filters.py#L35) | 逐agent按类型及ego可用历史/未来确定inclusive时间范围，索引阶段没有读取邻居距离。 |
| 取条件邻居 | [batch_element.py:164–190](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/data_structures/batch_element.py#L164) | 索引取样后才找15m内其他agent，允许返回`num_neighbors=0`。 |
| 拼batch | [collation.py:292–319,475–526](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/data_structures/collation.py#L292) | 显式处理整个batch最大邻居数为0的情况。 |
| 空邻居到NN接口 | [trajdata_utils.py:135–156](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/utils/trajdata_utils.py#L135) | N=0补一个NaN占位槽，后续availability为false；没有伪造真实邻居。 |
| 空邻居编码 | [trace_helpers.py:160–173](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/models/trace_helpers.py#L160) | unavailable槽不参与maxpool，全空时pool置零。 |

`algos.py:176`训练step直接使用`batch_utils().parse_batch`，所读固定训练链路未添加real-neighbor数量筛选。上述结论由索引、batch和编码实现共同支持，不是仅从“有10%邻居dropout”推断。

## 本地差别与最小修订范围

`orca_interact_usable_staging_v1/convert_official_raw.py:53–66`在全部合法agent/anchor之后用`eligible=(distances<=15.).any(axis=1)`筛选；Maps转换器`orca_maps_official_usable_staging_v1/convert_official_maps.py:109`也调用它。`interaction_corpus.py:81–83`又拒绝零邻居，Maps复用该batch实现。两域NPZ同时保存完整`states`，因此没有删掉这些状态本身。

本地`paper_interact.py:462–474`已支持N=0→一个zero占位且all-false mask；`:346–349`编码全空邻居为零。固定19槽、其中0个有效的batch也满足该数学。**允许零邻居不需要新NN、改变维度或生成伪邻居。**仍需要明确修改数据view的eligible index及corpus的最小邻居准入，不能只改manifest计数。

按root提供的本地100帧/训练agent元数据、31history/52future推算，合法anchors是30…47共18个：

| 训练域 | all-temporal推算 | 当前registered neighbor-only | 排除 |
|---|---:|---:|---:|
| Maps | 101,520 | 78,149 | 23,371 |
| Interact | 190,944 | 150,439 | 40,505 |
| 合计 | 292,464 | 228,588 | 63,876（21.84%） |

这是本地合同的算术，不是重新运行数据审计，也不声称是作者原始dataset的精确窗口数。作者原始轨迹长度、dt插值和本地100点重采样仍各自明确。

**建议供root决定：**如果本轮40k目标优先贴近作者窗口分布，正式启动前采用独立all-temporal-anchor view，读取已有states/maps，允许0…19邻居并保留所有原masked字段。保留原neighbor-only冻结产物；不用重新生成2,000场景，不改原NN/EMA/loss。新窗口union/count及采样RNG合同必须如实变更，旧228,588计数不能沿用。自然单人/远邻窗口改变ego轨迹和map条件分布；10%邻居dropout只去掉现有样本的条件，不能恢复这些未抽到的ego窗口。是否改善泛化仍未知；本报告没有授权或执行任何变更。
