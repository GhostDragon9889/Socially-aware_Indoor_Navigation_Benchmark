# 官方ORCA两域与EMA/累计梯度：独立正式候选

**当前状态：真实GPU试训和完整GPU恢复已完成，40k正式训练尚未启动。**
batch400实际OOM退出1；200×2完成4更新，再使用原完整恢复函数继续至5更新，均实际退出0。
参见[当前GPU产物与启动证据](../orca_official_formal_actual_runs_v1/README.md)。
实际入口为 `train_formal_diffusion_BN_metadata_v3.py`，只修正BN批量描述的一行metadata，训练数学与下述源审查/CPU验证一致。
以下详细记录保留准备阶段的数据合同和来源；其中“尚未GPU执行”指当时状态，以本段和实际owner记录为准。
不修改185/q935、旧CPU-tested混合stage、默认权重或原数据。

## 数据接口

OfficialMapsORCACorpus复用原Interaction.numpy_batch的31/52时窗与19槽邻居数学，
仅新constructor加载真实map与合法empty/repeated geometry，并将unknown raster换为真实crop。
所有地图标注presence=True，包括45个合法空布局；Interact的no-map presence=False。
不依赖可见障碍物筛选，不按几何hash去重，不过滤原raw碰撞或zero-window场景。

实际ToolsPY三split empty/nonempty各一的B2共有6个Maps窗口，
又使用同一已加载union的两域B2共6窗口：presence均正确，真实31/52/19/raster shape/current history验证通过。
session17300/93e1f8→d907a4 exit0、5.407秒；未Torch/SDK/NN/GPU。
参见[实际NumPy接口](actual_numpy_interface.json)、[真实闭合](actual_numpy_closedreceipt.json)。

| split | Maps合法窗口 | Interact合法窗口 | 合计 |
| --- | ---: | ---: | ---: |
| train | 78149 | 150439 | 228588 |
| validation | 10211 | 19237 | 29448 |
| test | 9129 | 19692 | 28821 |

union仍按全部合法窗口有放回均匀抽样，明确不是作者每epoch shuffle相同序列。
新训练默认官方Maps manifest，不再旧localMaps；旧stage保留137689/150439的不同定义。
两域原split/episode/source身份分别保留，域内显式indices不额外消耗child RNG。

## 默认模型方法与最小新增训练状态

- 原Nav五状态NN/DDPM/52帧clean-loss不改，复用已冻结paper_interact.py。
- 默认trace_unknown_raster：真实no-map数据mask保持False，但encoder availability=True；
  全.5未知raster仍真正编码，条件dropout先填.5再编码。
- skip_encoder单独本地消融，传真实presence；不能称作者原始no-map方法。
- Adam/LR2e-4；默认不clip，显式max-grad-norm会写入recipe。
- formal入口准备的默认microbatch8×accum50=effective400；最终硬件配置由真实GPU pilot决定。
  一次完整累计后的Adam更新才计一次optimizer update。
- 40k更新、effective400共有16M窗口；默认8×50配置需2M次microbatch forward/backward，实际次数取决于最终硬件配置。不是B8×40k的320k窗口。
- BatchNorm见实际microbatch。优先试400×1；若真实闭合失败原因是OOM，才依次试独立200×2/100×4。
  后两者的BatchNorm见200/100，不能说等同整批400；峰值和速度目前未实测。

EMA按作者固定config(.995/every1/start2000)整state_dict处理，包含buffers/整数copy cast。
在每个optimizer更新的第一个microforward前作用于旧raw状态；
<2000 whole-state-copy，之后mean-update。不会以microstep计EMA或warmup。
默认验证/部署用EMA，raw指标另外记录。

## 完整checkpoint与恢复

training_state.py保存：

- `model`=EMA，为原PaperDiffusionPlanner部署加载兼容；`training_raw_model`独立保存训练raw。
- 原Adam完整state、EMA recipe/events/last-step、optimizer/microstep/sample计数。
- microsteps_in_current_update=0，只有完整更新边界保存。
- 两manifest/split/window/union规则+mode+micro/accum/LR/clip+EMA机制身份。
- 三个union sampler RNG与Torch/CUDA/NumPy/Python全局RNG。

恢复必须load training_raw_model +raw Adam，并单独load EMA；
不能把部署EMA权重与raw Adam组合称exactresume。旧raw-only checkpoint仅可初始化，不能resume新正式phase。
默认model key提供EMA，因此既有部署无需重新发明NN接口。

## 度量与已知空地图

metrics.py只把原Maps公式的polygons文件输入换为实际manifest.obstacles；数值公式不改。
空几何无有限边界，原数学+inf仅JSON表示为None+reason='known_empty_map_no_finite_boundary'，
已知空地collision0保持；未知Interact的obstacle collision仍None/N/A。
按域汇报，不把N/A平均成零。

## 仅准备的正式命令

```bash
python docs/style_improvement_20261003/orca_official_mixed_training_staging_v2/train_formal_diffusion.py \
  --maps-manifest data/diffusion_amp/orca_maps_official_usable_1000_v1/manifest.json \
  --interact-manifest data/diffusion_amp/orca_interact_usable_1000_v1/manifest.json \
  --output-dir output/diffusion_amp/official_ORCA_maps_interact_EMA_accum_v2 \
  --init-checkpoint output/diffusion_amp/diffusion_training/step_1000.ckpt \
  --updates 40000 --microbatch 8 --accumulation 50 --learning-rate 0.0002 \
  --missing-map-encoding trace_unknown_raster --device cuda
```

**40k正式GPU训练尚未登记或运行。** 后续由root真实GPU小pilot执行fulltrainer main，评估速度/峰值后正式登记；用户已经授权实现和训练，无需再次请求一般训练许可。
不能据真实NumPy或CPU模型验证声称该新formal main已完整运行。

## 实际CPU训练状态验证

源审查后，使用随机初始化的默认五状态NN、真实两域语料，执行microbatch2×accum2的完整更新。
保存后再执行下一更新，与重新加载raw/EMA/Adam及全部RNG后的下一更新逐值一致。
原生产PaperDiffusionPlanner加载`model`得到相同EMA权重；EMA与训练raw分别保存。
六个错mode、累积配置、EMA配置、未完成microstep、EMA计数和缺失raw状态的负例均被拒绝。

1999→2000的EMA边界仅用极小module构造计数，验证整state与整数buffer的copy cast；
没有执行1999/2000次神经网络训练。checkpoint仅用于CPU测试，不是训练结果。
session45778/7aa771→c0de32实际exit0，9.483秒，CUDA隐藏、4线程；
未读取旧权重、SDK、GPU、DDPM采样、物理评估或sealed34。
参见[实际CPU验证](actual_formal_CPU_proof.json)、[真实闭合](actual_formal_CPU_closedreceipt.json)。
此验证实际effective batch为4；effective400、40k更新和formal main仍未执行。

## 已登记、尚未启动的4更新GPU pilot

[登记的精确argv与来源](GPU_pilot4_registered_commands_v1.json)绑定formal五文件、
复用NN/conditioning/helper、当前185源epoch、实际两域QC和已记录的4b6初始化身份。
准备未重新读取4b6权重内容，未重跑数据QC或185全量源核验。

优先microbatch400×accum1、4次optimizer更新、effective400、CUDA fraction .85，
eval_batches1/eval_interval4/checkpoint_interval4；明确使用pilot-only skip-sampling。
完成时训练窗口暴露1600，尚不产生ADE/FDE或闭环物理结论。
400×1实际进程因OOM闭合失败后，root才可选独立目录的200×2；该配置同样失败后才可选100×4。
各候选fresh初始化、禁止resume，不并行占GPU；需现有AMP owner先实际闭合。
实际退出码/耗时/峰值仍为null；CUDA fraction不是显存成功保证。
4次更新均处于EMA warmup，不能据此宣称2000步后的EMA效果。

## 方法与预算准确范围

论文TRACE/PACE A.1.3写40k/LR2e-4；固定公开JSON写B400/100k。
这里40k+B400是root选的公开信息组合候选，不能称论文明确同时给B400×40k。
原TRACE未来transition6/history8与当前Nav五状态改写不同；AMASS初始4b6也不是作者TRACE权重。
参见[EMA/预算来源核查](../trace_training_budget_source_review_v1/README.md)、
[真实raw模型两模式CPU基线](../maps_interact_mixed_training_staging_v1/actual_CPU_model_smoke.json)。
新候选效果仍需独立open-loop与完整闭环物理检验。
