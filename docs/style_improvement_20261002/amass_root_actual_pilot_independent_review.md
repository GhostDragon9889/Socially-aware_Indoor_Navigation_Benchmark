# AMASS根轨迹课程：真实4+4更新保存与恢复审计

## 结果

真实GPU训练首段4次更新、从不可变step4恢复后再4次更新均完成。CPU独立审计166项通过：[机器报告](amass_root_actual_pilot_independent_audit_v2.json)、[复现脚本](audit_amass_root_actual_checkpoint.py)、[日志](amass_root_actual_pilot_independent_audit_v2.log)。审计器未启动Kit/GPU、未读取34条封存测试捕获、未修改业务源码或checkpoint。

| 实际执行段 | 新增transition | 控制步/actor | 跌倒reset | clip末帧截断 | episode cap | 本段最低pelvis(m) |
|---|---:|---:|---:|---:|---:|---:|
| fresh→step4 | 32,768 | 128 | 1 | 227 | 0 | 0.428104 |
| step4→step8 | 32,768 | 128 | 0 | 330 | 0 | 0.677508 |
| 合计 | 65,536 | 256 | 1 | 557 | 0 | — |

256个actor、30Hz控制、120Hz物理，192m地面、48m出生offset，AMP采用state-action discriminator；训练rollout使用实际AMASS根XY和明确的linear_velocity尾部，不执行Diffusion规划。runner结果的fall/reset/minimum-height按本次进程段统计，课程checkpoint计数是累计值，已分别核对，不能把恢复段结果当成从零累计结果。

## 工件身份与严格加载

- [step4真实结果](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/amp_style_amass_root_curriculum_pilot_v1/result_step4.json)：checkpoint 387,458,101B，SHA `69bd8c76b9ff61f29281401359bb339b111c3fe24414399f9dcbbd034a60ea19`。
- [step8真实结果](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/amp_style_amass_root_curriculum_pilot_v1/result_step8.json)：checkpoint 652,093,749B，SHA `42154ff23b8d992af00ad5f0b0843b04948a7275072c9cf87954d73cf49474a9`。
- 冻结[prelaunch清单](amass_root_linear_training_source_v1.json)中的124份源码与tar成员逐一核对：精确路径集合、仅普通相对路径、无路径穿越、bytes及SHA全部相等。archive SHA `ed16c4cdbfd533a9fe2bfeb50c3f717cdbb0f033462dadd8b4acf62a0adaf56a`；六份业务源码与此前[25项独立CPU审计](amass_root_curriculum_independent_integration.json)完全相等。

两份checkpoint均由`torch.load(weights_only=True,map_location='cpu')`读取；当前实际`PaperAMPTrainer.load(...restore_rng=False)`以strict模型加载接受完整payload。模型、两个optimizer、四个normalizer、RNG、runner物理/逻辑状态以及完整policy replay buffer的tensor均finite。auditor没有放宽外部checkpoint加载或恢复CUDA RNG。

保存config与结果JSON逐值相等。checkpoint中`action_mask`是tuple，JSON是list，因此检查采用标准JSON序列化投影；最初直接比较产生的auditor失败日志保留为[tuple投影失败记录](amass_root_actual_pilot_auditor_tuple_projection_failure.log)，不是业务配置差异。先前164项报告保留；最终v2新增整个policy-replay/完整bundle finite检查。

## 恢复契约

对全部256个实际保存actor调用最终业务`load_state_dict`及`validate_runner_state`，验证原始motion/f0、年龄、episode IDs、剩余真实帧、末端速度、固定出生平移、151点native计划与本地plan age。step4有3个刚reset actor，step8有4个；这些actor均正确标记过期计划，下一控制步须重新规划，未触发先前发现的age0旧计划漏洞。

共享corpus RNG与课程保存RNG逐位相等。state4→state8训练config和模型架构、runner合同及全部课程source配置一致；control clock从128继续到256，累计actorsteps从32,768继续到65,536，计数未清零。恢复argv明确指向不可变`step_4.pt`。AMP历史全部属于各actor当前episode，无跨episode混合标签。

累计reset为814次：初始化256 + 首段228 + 恢复段330。557次源末端截断、1次真实跌倒；episode cap为0。计划尾部是显式工程延伸：累计step4有64,956/126,250=51.4503%未来10Hz任务knots超出真实源，step8为137,851/257,050=53.6281%；全部属于linear extrapolation，hold计数为0。该比例不代表执行了clip以外的物理状态，也不能标为真实AMASS未来。

## 范围

审计证明实际训练工件完整、source绑定一致、逻辑恢复/随机状态合同有效；真实执行结果证明恢复后又完成了32,768个物理transition。没有序列化PhysX solver/contact缓存，故不宣称两进程物理逐位重放等价。这是执行与保存恢复pilot，不能据此判断长训练收敛、风格提升或严格论文复现。
