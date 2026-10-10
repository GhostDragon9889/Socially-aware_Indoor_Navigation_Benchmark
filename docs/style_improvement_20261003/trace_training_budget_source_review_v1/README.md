# TRACE EMA、优化器与训练量：窄来源核查

**只读来源研究；不改已CPU测试stage/185/q935，不加载模型或启动GPU。**
Nav公开文本未披露其定制Diffusion的EMA配置；以下是TRACE作者公开实现差异，不能据此断言Nav私有实现是否启用EMA。

## 已证实差异

| 项目 | 固定594b作者代码 | 当前五状态生产/混合stage |
| --- | --- | --- |
| 优化器 | Adam，LR2e-4（algos.py:254–259） | Adam，CLI默认LR1e-4 |
| LR schedule | configure_optimizers只返回Adam；JSON epoch_schedule=[] | 固定LR，无scheduler |
| 梯度裁剪 | 作者Trainer未显式配置clip；不推测其外部PL版本默认 | train_step明确全梯度clip norm1.0 |
| EMA | True，decay.995，every1，start2000 | 没有EMA模型/更新/计数 |
| EMA warmup | step<2000完整load_state_dict(raw) | 不适用 |
| EMA时刻 | training_step入口、forward/backward/optimizer之前 | 不适用 |
| EMA内容 | helper遍历整个state_dict，含buffers；非仅parameters | checkpoint只有raw model |
| 验证 | 同时raw/EMA loss与采样指标，选checkpoint监测EMA loss | 原始模型loss/plan |
| 导航/推理 | forward/get_action/set_guidance/clear_guidance选EMA policy | Planner加载artifact["model"]并使用raw |
| checkpoint | EMA是nn.Module属性；作者Lightning ModelCheckpoint | raw model/Adam/RNG/config/dataset元数据，无EMA状态 |

EMA公式是ema←.995×ema+.005×raw。作者helper同样处理整数buffer，copy_按目标dtype写回；
不能擅自把“只均值parameters、所有buffer直接复制”说成作者原函数。
该公式在普通step的**开头**作用于此前raw状态；实现为optimizer之后更新会相差一个更新时刻，
应单独声明选择，不能把两种次序混写。

EMA policy以nn.Module属性注册，结合作者Lightning保存/加载路径，推断其weights进入checkpoint；
本次没有反序列化作者checkpoint验证实际key。cur_train_step是普通Python计数从0初始化，
未见专用on_save/on_load hook；不推断作者resume后的warmup计数一定完整。
本地新实现应显式保存EMA schedule/计数以及raw/EMA状态，恢复前比对，
不沿用这项未证实假设。

引用：
[作者algos.py:91](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/algos/algos.py#L91)（构造）、
[146–174](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/algos/algos.py#L146)（更新时刻）、
[210–259](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/algos/algos.py#L210)（验证/优化器）、
[269](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/algos/algos.py#L269)（action）；
[EMA helper600–613](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/models/trace_helpers.py#L600)。
作者[train.py:90,104,139](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/scripts/train.py#L90)保存、
[policy_composers.py:42](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/evaluation/policy_composers.py#L42)加载；
本次web raw固定链接cache miss，实际依据已本地克隆的同commit primary sources，source_records.json逐字节核对git blob。

## 40k不是单独可比较的训练量

TRACE/PACE论文附录A.1.3（印刷页13）给40k iterations与Adam/LR2e-4。
固定作者JSON training给batch400、num_steps100000；不是40k。
公开文本与公开config预算不同，正式计划需选定并同时列samples，不称两者同一预算。

| optimizer更新 | 有效batch | 总window exposure |
| --- | ---: | ---: |
| 当前B8×40k | 8 | 320,000 |
| 论文40k + 作者configB400组合候选 | 400 | 16,000,000 |
| 原公开JSON100k×400 | 400 | 40,000,000 |

第二行是明确组合的**本地候选预算**；不是论文明确披露“B400×40k”的证明。
统计应另外保留microbatch、accumulation、optimizerupdates与实际window数，
不能把50次micro forward/backward算50次EMA/optimizer更新。

## RTX4080 16GB的可执行方案：先root pilot，再正式登记

建议在官方Maps/Interact usable源ready后，root独占GPU小pilot以microbatch8起步，
测真实allocated/reserved峰值与每秒microbatches，再测16是否可用。
不得用本次CPU7.3秒smoke或旧GPU数据外推新数据的显存/完整训练速度。

正式候选保留FP32/Adam/LR2e-4，选择：

- microbatch8×accum50 → effective400；或microbatch16×accum25 → effective400。
- 每个microbatch原loss mean除accum，再backward；累计满400窗口后检查finite，
  只做一次总梯度clip/Adam。不能直接调用当前train_step50次，它每次zero/clip/step。
- EMA在一次optimizer-update对应的首个microforward前按显式source次序更新；
  warmup2000与every1都按optimizer更新计数，40k仅40000次。
- 总训练成本是2,000,000次micro8或1,000,000次micro16的forward/backward；
  需根据pilot吞吐估算时间，不给未经测量ETA。
- MapEncoder有ResNet/decoder BatchNorm：梯度累积**不是**batch400的BN统计。
  每次BN见micro8/16，且dropout/RNG调用分段；即使loss梯度加权正确也不声称逐值作者等价。
  暂不新改BN、AMPprecision或引入其它优化器，以免多变量混合。

当前parser的cuda_memory_fraction上限.15也是独立实验的旧资源限制；
不要把它当16GB完整可用内存或擅自改正在绑定的stage。root新pilot选择资源上限并登记后再适配。

## 最小后续实现缺口（本轮未写trainer扩展）

1. 新独立stage的累计梯度更新边界；记录micro/有效batch/真实samples计数。
2. EMA raw/EMA双状态、作者schedule与明确update时刻，完整resume身份/计数。
3. 验证raw和EMA各自原指标；推理明确选EMA，不能让旧Planner静默继续读raw model。
4. EMA推理导出单独artifact/模型SHA；保留raw+Adam的训练checkpoint用于resume。
5. officialMaps loader与新phase接线完成后，root真实fulltrainer main pilot验证该入口。

当前stage的两模式CPU一步smoke仍有效，只验证当前raw训练接口，
不覆盖future EMA/accum、正式GPU预算或论文效果。

附：维度表述须区分来源：TRACE历史encoder输入8维（含扩展yaw/尺寸/availability）；
公开Diffuser state-and-action transition是4 observation +2 action =6维，output2 actions。
本地Nav改写未来输出5维[x,y,u,v,theta]。不能把TRACE所有张量统称8维。

