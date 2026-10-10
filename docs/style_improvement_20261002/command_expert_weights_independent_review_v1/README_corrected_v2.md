# 专家正样本软权重候选：独立 CPU 复核（文字计数更正）

本页仅更正前版正文的候选文件计数27→26；实际JSON早已逐项核验26个文件，131项通过及全部数学/来源结论不变。前版封存字节保留。

**未发现当前数学prototype的具体阻塞。** 131项独立实物门禁、37项专项CPU测试及Ruff通过；在临时副本去掉positive权重会使解析梯度测试失败。生产和正在训练的931隔离source均保持不变，候选尚未接入trainer、checkpoint或GPU。

## 已核对的事实

- 候选26个冻结文件逐项SHA/bytes核对；prototype固定SHA `3889cfcb3841abbf7b5a7ca8a0860d200d9f9610a24c67d5e3bd18f0eea32a09`。实际CPU证据的prototype/reader来源一致，公开的30个真实pre-action NPZ与rawtask数组SHA逐个匹配，4actor合计120请求，不是post-step重建。
- 120个请求descriptor与冻结实物逐位相等；512个固定专家descriptor重算权重逐位相等。当前实物无饱和界限，独立dense核计算及closed-form `raw/mean(raw)`解与分块二分的FP32输出逐位相同。实际权重min `.5399966`、mean `1.0000000015`、max `1.4888176`，loss权重ESS `485.17/512`。
- 最初已实证的subnormal bracket溢出（finite `[1e-308,1]`曾返回mean2）已按候选raw固定[.5,2]最小硬拒绝，末端也核有限性、bounds和mean。不会由clip后直接除mean造成越界。
- 未消耗CPU RNG或启动CUDA/SDK。候选实物引用的旧成熟D checkpoint本次独立完整重SHA，确与既有 `eeac0377…` 声明相同；其实际body/DOF顺序等于PACER canonical，57有效动作一致。候选CPU reader自身引用的是前次封存checkpoint记录，本独立实物复核补了当前byte-binding。

## 方法与接线边界

`prototype.py:73` 的请求速度是未来0.5秒catch-up位移/时间；`prototype.py:123` 的专家速度是历史t−9至t+1共10个间隔的位移，旋到t时刻body heading。两种窗口不等同，已明确说明。路径段方向变化率与body yaw rate分开，不能从20维未来XY推出body目标yaw。

`prototype.py:167` 只汇总全batch请求与专家的核亲和；`prototype.py:223` legacy直接使用原BCE算式，`prototype.py:232` enabled仅给positive BCE逐例权重。D仍只读history/action2019，没有actor的task/context输入，不能得到逐actor条件风格识别。最终权重与ESS只是已采样batch的loss质量分布，不是新的bank采样概率。当前120低速请求下running权重反而降低，不能宣称自动补齐动作多样性或改善摆臂。

后续若正式集成，仍需要单独验证：同一次真实expert抽样取得descriptor、selection后mean1/bounds、negative/GP/replay/reg数学及调用次数不变、enabled配置/来源的严格保存与恢复、legacy逐位路径和GPU transport。当前原型及CPU证明未执行这些接线，未产生训练/导航质量改善。未改变2848数据或固定风格参考，未读34测试。

[131项独立实物检查](actual_review.json) · [原候选说明](../staged_command_expert_weights_v1/README.md) · [临时副本mutation证据](positive_weight_mutation.json)
