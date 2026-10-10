# Protected terrain RMS：完整22条验证结果

本次真实GPU跟踪已完成；以下分析仅CPU读取实际终态与全部原始记录，未启动Kit/GPU，未改931隔离源码、181生产源码、指标实现或冻结登记，也未读取34条test。

## 全部序列的主结果

全部四组均完整22条、11,982指标帧；未截短、未筛除失败序列或跌倒帧。成功定义保持原门禁：序列内任何一帧的平均刚体位置误差超过0.5m即为跟踪失败；跌倒另计。Style计算排除22个初始reset帧及跨clip导数边界，保留全部11,982真实metric帧，11,960个同clip速度区间。

| 模型 | 真实更新/总转换 | 成功 | 跌倒 | 全帧位置误差m | 全帧旋转误差rad | 90D MMD² | 速度Wasserstein m/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 部署基线 backward_penalty1000 | 1000 / 8.192M | 11/22 | 0 | 0.300708 | 0.861454 | 0.0468211 | 0.143767 |
| mixed2000 state-only | 2000 / 16.384M | 19/22 | 0 | 0.200096 | 0.768363 | 0.0604406 | 0.0458342 |
| 原 state_action2000 | 2000 / 16.384M | 17/22 | 0 | 0.208382 | 0.725853 | 0.0625297 | 0.0583321 |
| protected2000 | 2000 / 16.384M | **15/22** | **0** | **0.247776** | **0.775494** | **0.0545289** | **0.0524929** |

Protected相对原state-action：风格MMD²与速度分布距离下降，但跟踪成功从17降为15，位置/旋转误差上升。因此不能称全方面改善，也不能凭风格指标推广为新部署默认。相对mixed2000和部署基线，还存在D输入、初始化/统计、课程和训练预算等历史差别，不把跨这些模型的差值归为单一RMS因果效应。

90D为固定heading局部pose+velocity表示，Gaussian MMD²、σ=1、速度时间尺度0.1s；主值为已有固定速度桶中支持桶的等权平均，不是所有帧无条件池化MMD。四组0–0.1、0.1–0.5、0.5–1、1–1.5、1.5–2m/s五桶均支持；2–3m/s参考仅4样本，未达到固定30样本门槛，保留null，不隐藏改桶。新指标详见[evaluation.json](evaluation.json)、完整14组比较见[comparison.json](comparison.json)。固定45D仅用于随后导航评估，本次没有对tracking另fit或冒充已算45D。

## 同帧误差分解

| 全部11,982帧的均值 | 基线 | mixed2000 | 原state_action2000 | protected2000 |
| --- | --- | --- | --- | --- |
| 根XY误差m | 0.228346 | 0.118753 | 0.123692 | **0.171763** |
| 根yaw误差rad | 0.600987 | 0.514223 | 0.447642 | **0.519688** |
| 去根平移/heading后的身体位置误差m | 0.119189 | 0.116580 | 0.125840 | **0.125261** |
| 去heading后的身体方向误差rad | 0.573434 | 0.507032 | 0.536737 | **0.517555** |
| 父关节相对旋转误差rad | 0.301823 | 0.256404 | 0.266489 | **0.267051** |
| 实际根平面速度m/s | 0.777697 | 0.898313 | 0.931414 | **0.904620** |

相对原state-action，根XY/yaw误差明显增加，而局部位置/父关节旋转变化很小。这是同帧几何分解所观察的区别，不能由此推断RMS改变的内部学习机制或把根误差简单加到姿态误差。完整分解保留原世界坐标MSE恒等式/交叉项及逐身体、逐clip、逐参考速度桶结果：[protected2000_decomposition.json](base_nine_comparison/protected2000_decomposition.json)。

Protected仍失败的7条：motion IDs **1、4、20、245、254、257、259**。失败包含Jog、BackwardsWalk、Walking和DanceTurns；0fall不等于0tracking failure。相对原state-action失去成功IDs1、4、254，新增261成功；逐帧首次失败与最大误差保存在[actual_summary.json](actual_summary.json)。Common-success的帧加权/逐clip两种分母仅作为次级解释，绝不替代ALL22或移除失败。

## 配对与真实来源

- 独立逐数组读取protected与三个比较模型的全部capture，形成**66个实际clip配对**：初始11项真实body/root/DOF/AMP-history数组，完整reference/clock/trajectory数组，**shape、dtype及C-order bytes均严格相同**。该结论不是仅比较均值或`array_equal`。
- 原13组evaluation/comparison字典逐项保持exact，没有重新计算或覆写历史数字；新组通过冻结d19c9工具调用原90D evaluator和decomposition，仅增加明确的新label/source admission。
- 实际控制器：sealed `step_2000.pt`，**653,044,761B**，SHA256 **e11a9a47d5e1147c4f894a519827d50539faf449a892c714d690957f1b267663**。Deps真实terminal准入2046/2046通过，保护hook恰2000次、参考suffix SHA `5ebd61c10871dfad790d1dbbf6e9d1e05e6e6f0642522900dc0412601c176137`；没有把500/1000阶段当终态。
- 931不可写隔离源码逐文件重hash，manifest `4227b3d8eef584ded2348519b4ac8d7809b21c16c9a8c2624a2d8d2b82ccdd07`；实际eval ledger观察70个owned执行、73个committed events。`runtime_finalization_observed=false`如实保留，不补造未观察导入或App关闭日志；终态与完整raw另外核验。外部SDK/native包不属于931 owned ledger。
- 固定验证参照为c274 AMASS22原parent sources，与声明训练1853 parent sources交集0；没有证明subject/session层独立，也不是论文作者2848 subset或八场景完整复现。

## 验证边界与后续

冻结CPU wrapper实际exit0；本补充summary实际exit0，全部66配对与7失败保留，Ruff通过。只新增本结果目录的工具/报告/归档，冻结registration和评估公式未改。新200秒导航仍由root按原登记串行运行，尚未由本报告推断导航稳定、接触或停滞改善。

本实验是明确的工程实现与本地验证协议。不能把15/22、位置0.2478m、旋转0.7755rad直接与论文64.6%、0.218m、0.343rad作等价排名，更不能因零跌倒或单个MMD改善宣称论文完整效果已达到。
