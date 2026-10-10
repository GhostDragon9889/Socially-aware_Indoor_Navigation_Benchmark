# 地形 RMS 保护：训练器候选集成核验

状态：**docs 隔离候选66项CPU测试通过、0跳过；13项旧默认/冷导入独立证明通过。** 没有改生产、启动Kit/GPU、训练新模型或宣称导航改善。

## 已接入

- TrainingConfig新增`task_terrain_statistics_mode`，旧缺字段为`legacy`；未知/bool模式、非canonical357/1044、禁用观测归一化或非clip5明确失败。
- protected + from-scratch 在资源读取和Kit之前冷拒绝；冷导入不加载Torch/SDK。
- 训练器构造未绑定policy；act/value/update/save拒绝未绑定使用。重复暖启动检查在所有模型/统计copy之前；加载公开对象后也先核固定公开SHA，才进行copy。
- 仅在原 self/task/AMP moments更新后调用suffix hook。轨迹20维、task共享count、其他moments和PPO/D数学不变。
- save核calls/update_count、共享count/真实transitions、参考SHA和suffix逐位关系；仅protected新增policy key。load调用shared严格恢复，不覆盖受损moments、不读取外部统计NPZ。

## 实际CPU证据

66项包含shared policy33项、候选集成19项与旧基础训练14项。实际完整119755239B公开PACER no_shape执行warmstart，reference_count=61931520001与cap后warmcount=200000分开记录；重复暖启动失败前后模型/moments逐位不变。完整保护bundle保存后，root staged actor-only strictloader实际执行，raw/applied actions与trainer逐位相同，调用者CPU RNG不变，最终仅保留actor/task_encoder。

优化器更新/连续恢复使用**明确合成观测与rollout**及小网络数值fixture；真实执行Torch PPO/D与Adam，不是SDK实测。恢复后下一更新的model/Adam/moments/scalars逐位一致。missing/extra/calls/source/count/moments/mode破坏均拒绝。实际public warmstart和完整推理用例没有跳过。

独立读入before trainer源码，在仅移除声明新hooks后**完整AST与before相等**。旧默认真实CPU优化器更新的模型、PPO/D metrics、两Adam、四normalizer及CPU RNG消耗逐位一致。新默认字段会改变config metadata，因此不声称整个checkpoint文件SHA不变。

首轮66项中因临时overlay缺data资源有1条旧public test跳过；该日志/XML已保留。仅在/tmp overlay新增data→workspace/data资源链接后完整重跑66通过/0跳过；新增protected实际public用例首轮已经执行通过。没有将缺资源误报CUDA测试。

## 尚待实测

shared policy支持严格校验map_location CUDA后小Tensor归CPU；这仍须root实际fresh4更新、恢复再4的GPU pilot确认。没有修改world/runner/RNG恢复顺序或现训练sourcebundle。旧真实三权重的部署兼容可由root独立随后复核，不将本组CPU用例扩大为GPU或导航效果验收。

[66项完整结果](combined_integration_tests_complete.xml) · [13项before/default证明](dependencies_before_default_actual_CPU_proof.json) · [owner冻结](dependencies_source_freeze.json)
