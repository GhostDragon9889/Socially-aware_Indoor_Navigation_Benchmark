# 固定 AMASS2848：BABEL dense 序列跨度约定

新增 [`extend_babel_dense_sequence_semantics.py`](../../../scripts/extend_babel_dense_sequence_semantics.py)，对原只读 train semantic overlay 添加 **独立、显式的 dataset assumption 层**。原 manifest、2848条动作及顺序、kinematic proxy、原 overlay、专家 prior 和正在评价的S2均未修改；没有加载动作数组、模型或SDK。

## 依据与保守范围

[BABEL 官方说明](https://babel.is.tue.mpg.de/data.html)对 dense 数据中仅有 sequence label 的序列采用整段动作跨度约定，并区分 dense 与 extra 多标注者标注。这里把该约定与已有显式 frame 时间分开记录，仅接受：

1. 原 exact raw-source join 匹配 **`train.json`**；不从 `extra_train.json` 回退。
2. `frame_ann` 明确为null、`seq_ann.mul_act`严格为false且恰有一个 label。
3. 原 raw／proc文本有效、BABEL duration为有限正数；若序列label自带时间字段，其值须与0／duration一致。
4. 原 label／evidence ref完全对应；原 category／null不补写、不投票。

这是比官方 dense 序列约定更保守的实现：实际有39个 dense 无frame父动作标记多动作，保留为不适用。只有一个文字label不等于明确单动作，不能将`mul_act=true`当成false。

区间使用原BABEL `[0,dur]`；投影到原转换记录的 first／last sample time，**不将BABEL时长拉伸到AMASS原长，也不修改半开frame stop**。它是数据集的跨度约定，不是新测得的动作起止时间、单标注者绝对真值或新 frame annotation。

## 实际全量结果

最终6项CPU测试 `656f77`／0，Ruff `3fe5ab`／0；实际构建 `fa970a`／0。
完整ZIP和manifest SHA前后相同，只打开ZIP中的dense `train.json` member；其15,101,932B／SHA `5da8185d…`与原登记相同。
每个已匹配dense record与ZIP原记录完全相等；source key复用原封存join函数，保持完整数据集／受试者／文件路径，不用basename模糊匹配。

| 父动作状态 | 数量／1853 |
| --- | ---: |
| 有显式 frame annotation，不应用约定 | 513 |
| 无frame、明确单动作dense序列，适用约定 | 304 |
| 无frame但多动作／非明确单动作，保留未知 | 39 |
| 无dense train exact match | 997 |

“无dense match”包括仅匹配extra或完全没有train标注，不能称为所有997都没有任何语义证据；原overlay仍保留extra。

- 原2848 clip／1853父动作均保留、顺序和ID完全一致。
- **399条 clip**来自上述304父动作，全部实际sample时间范围处于其BABEL约定跨度内。
- 这399条都不在原 **844条显式timed interval union全clip覆盖**中；两种证据分别存储，未将399条计为显式测量。
- 其余 **1605条**未获全clip显式时间并集或本次保守跨度约定；仍可具有partial／sequence／extra语义证据，不能全部标记为无标注。

原eligible父动作category统计中，walk215、run11、jog3、sideways movement10、stumble11、dance7；还有其他类别。类别可重叠、部分是动作修饰语，**这些是304个父动作的原文本category计数，不是2848 clip类别数、专家draw频率或Nav作者筛选ID**。不据此删除动作、重标proxy或改变训练权重。

## 失败与修正保留

初版5tests `c2b09b`／0、Ruff `5d1321`／0，但实际构建 `a5a09d`／1，因直接将BABEL包装路径与raw source key相比而拒绝。真实例子为 `CMU/CMU/45/45_01_poses.npz` 对 `CMU/45/45_01_poses.npz`。
最终复用原封存 `join_train_semantics.py::feature_key`，只去掉确切BABEL包装前缀，并新增相同basename不同subject与非法路径测试；未放宽成模糊匹配。初版源码、测试和失败日志全部保留。

最终源码、实际输出、输入身份及外部退出见 `execution.json`。输出 `actual_fixed2848_v2.json` 保留所有父动作状态、原标签／ref、各clip原sample跨度、原显式覆盖标志与单独的assumption覆盖标志。

## 复现与后续使用

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/extend_babel_dense_sequence_semantics.py \
  --output output/diffusion_amp/new_BABEL_dense_sequence_convention.json
```

输出文件必须不存在。该补充用于审查既定真实AMASS的动作语义支持；后续任何条件采样／风格奖励实验应明确区分显式frame标注、dense跨度约定和未知项，并独立登记。它没有恢复作者2848动作清单或证明论文效果。
