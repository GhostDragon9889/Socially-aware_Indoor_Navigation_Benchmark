# 正式Diffusion40k：命令与未来CPU终态审计准备

40k optimizer updates已获用户授权；这里没有执行正式main、NN、GPU或读取pilot文件。
Root实际观察400×1因map encoder OOM退出1，200×2的4更新小试退出0。
因此正式候选固定 **microbatch200×accum2，effective400，40k更新，16M窗口，80k microsteps**。
唯一BN metadata修正c344已获root逆整文件byte审查通过；旧4ea／旧注册计划仍保留。

## 源码与命令

[精确正式argv／审计计划](commands_and_audit_plan.json)绑定冻结pilot registration ee4e，
复用其5个formal source、NN42abd、原conditioning/helpers及185源epoch。
不复制或重新冻结整个源树。[未来CPU reader源码](audit_terminal.py)供root审阅，尚未运行。
Ruff source check实际退出0（408a02）；没有新的NN smoke或测试套件。

独立正式目录：`output/diffusion_amp/official_ORCA_maps_interact_EMA40000_micro200_accum2_v1`。
从原4b6/step1000五状态权重**fresh初始化**，`planner.trained_steps=0`，新Adam、fresh训练RNG；
继承其架构／归一化和model权重，不继承原optimizer或pilot step4。
这仍是本地AMASS初始化和Nav五状态适配，不是作者TRACE权重／transition6模型。

## 4更新小试与40k主循环的确定差异

| 参数／行为 | 实测200×2小试 | 正式准备 |
| --- | --- | --- |
| optimizer updates | 4 | 40000 |
| eval_batches | 1 | 4 |
| eval_interval | 4 | 1000 |
| checkpoint_interval | 4 | 1000 |
| 最后DDPM采样 | pilot skip-sampling | raw／EMA各16窗口／域 |
| output | pilot独立目录 | 新正式独立目录 |

同一个c344 `main`调用同一个8daf `optimizer_update`，没有4／40000的预算特殊分支。
相同200×2、Adam2e-4/no clip、unknown.5和条件dropout、五状态NN／52帧clean-loss、
whole-state EMA.995/every1/start2000、真实全union、seed20261001／fraction.85。
第一更新后和最后更新仍按同一条件额外验证；每1000更新保存last，最后独立copy step40000（nlink1）。
验证fork Torch RNG、保存恢复域generator；保持原数学，不改变训练sampler规则。
最后16窗口／域的采样是小范围open-loop诊断，不是全100 scene／完整闭环benchmark。

## 根决定如何记录真实硬件选择

准备JSON仅保存root已观察的信息，未假装已读取sealed pilot artifacts：

- 400：33525/c88759→5a99cf，实际exit1，第一training_loss map encoder OOM。
- 200×2：40507/ee42f4→5a4d53，实际exit0，4更新／1600窗口，step4 SHA1807e2…ee99。
- root报告训练update记录的peak allocated **8762191360 B**、reserved **11523850240 B**。
- main报告elapsed约14.5059秒；update3→4约.99秒。整体耗时不能除4冒充steady更新速度，
  .99秒单区间也不是40k正式ETA；main的timer不含前面的语料／模型setup。

Root在正式启动前写新的actual决定，绑定sealed owner/result/log记录、选中完整argv及c344／注册计划SHA。
实际峰值须注明取自哪个记录，不能把Torch allocator峰值叫整个机器／整个进程峰值。
200×2 BN见200例，真实effective400并不把BN变成400；run_config布尔True正确。
snapshot和result的training_recipe分别明确microbatch=200／accumulation=2／effective_batch=400／BN_scope。
100×4没有被选择，不再未经实际OOM改变正式配方。

## 必须完整保存的终态字段

- 原部署`model`=EMA，`training_raw_model`独立raw；Adam全state／paramgroups。
- step=optimizer_updates40000、microsteps80000、training_windows16000000、未完成microstep0。
- EMA recipe／before_update_events40000／last_pre_update_index39999；warmup与BN buffers按原整state规则。
- 配方（mode／micro／accum／LR／clip／EMA／BN）、两manifest／三个split窗口identity与原4b6初始化身份。
- 三个union sampler RNG，以及原Torch／CUDA／NumPy／Python RNG。
- snapshot独立inode／nlink1／readonly／真实SHA；真实root closed owner与完成training_result。
- raw／EMA分域validation/test loss与16窗口open-loop采样，Interact地图指标仍N/A。

## CPU终态审计的真实范围

未来仅在root实际GPU退出0并封存后调用reader：先绑定实际owner exactargv、source、result和snapshot SHA。
原`validate_resume`核config／配方／corpus／计数／整EMA与finite状态；
原Planner只strict load EMA与raw，原Adam只load完整state并核moment形状，不forward／step。
用isolated CPU对象解码CPU全局RNG，列出三个saved union下一逻辑indices。

**原restore_rng在CPU面对saved CUDA会拒绝。此reader不调用它、不剥CUDA字段，
只保留并验证saved CUDA bytes结构，不能称fullGPUrestore／GPU bitwise resume。**
原2×2 CPU proof仍是既有恢复接口证据；root可独立做实际GPU `--resume sealed step4 --updates1` API检查，
精确准备argv见JSON，预计终态step5，不能声称已与第二次GPU下一更新做bitwise对照。
该step5仅API小试，也不用于40k fresh初始化。

实际中断恢复必须原`validate_resume`＋`restore_training`：raw+raw Adam、单独EMA、所有saved RNG，
并在可见CUDA上实际恢复其保存状态；不能改micro／accum／mode／LR／EMA／数据。
CLI `--updates`是**额外**更新数：从sealed stepN继续40k总预算，应使用40000−N，不能再加40000。
当前reader只接受fresh start0的完整40k结果；若实际发生中断，root另登记真实segment lineage后仅适配这项I/O条件。
训练预算40k+B400是论文40k与公开config B400组合的本地候选，不宣称作者明确给这一配对或论文效果达标。
