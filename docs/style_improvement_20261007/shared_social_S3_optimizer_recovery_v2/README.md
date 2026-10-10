# S3-R：12000次保存点的训练恢复

## 恢复范围

原S3日志停在12454，训练／监控句柄不可用，实际退出码未知。
最后完整保存点为12000：恢复模型、两个Adam、四组RMS、完整判别器回放缓存，
以及Python／NumPy／Torch CPU／CUDA训练随机数状态。
物理世界、接触缓存、课程私有随机数、源采样器及历史未保存，因此重新建立。
本次是明确登记的物理重置后优化器续训，不是原S3连续执行或完整物理恢复。

保留原12000次有效训练曝光，重算12001–12500，共500次／4096000转换。
原日志中未保存的454次及单次smoke不进入有效训练谱系；原始日志／权重／失败记录全部保留。
完成后的有效预算为12500次／102400000转换，但尚不能证明质量达标。

固定真实AMASS2848、seed7、256人／64组、T32、原课程、完整六轮PPO、
奖励、Pop-Art、物理／控制／清理保持；只有显式物理重置和保存节奏发生变化。
每25次保存权重，每125次保存完整rollout，终态预选`step_012500.pt`。
恢复smoke权重没有用于正式训练；两次都从原`step_012000.pt`加载。

## 已执行的验证

- 原12000保存点的实际CPU加载退出0：129个张量、1313个标量字段逐值恢复，含两组Adam／四RMS／回放／待恢复RNG。
- 初版SDK smoke因未展开默认动作掩码而被配置比较拒绝：实际退出1、更新0、清理成功，证据见[v1](../shared_social_S3_optimizer_recovery_v1/smoke_actual_execution.json)。
- 修正版按原Social配置的固定57维关节掩码展开默认值，保存／请求配置严格相同；8项测试及5个subtest、Ruff通过。
- 修正版256人实际执行12001次更新，退出0、清理成功；[外部退出记录](smoke_actual_execution.json)及[CPU审计](smoke_CPU_audit.json)通过。
- 初次正式owner被另一IsaacSim进程的独占检查拦下，没有开始训练；原[失败终态](completion.json)保留。

## 正式运行与终态

正式恢复在2026-10-07 21:50（Asia/Shanghai）启动，23:46:12完成训练和CPU审计。
[终态退出记录](owner_capacity_v3/completion.json)：训练、终态CPU审计实际退出均0，
500次恢复更新全部完成，训练入口耗时6925.390秒（约1小时55分25秒），清理成功。
[外部启动记录](owner_capacity_v3/actual_launch.json)保留首个实际更新12001的历史观测。
[独立OS owner](owner_capacity_v3/owner_plan.json)脱离聊天连接运行，使用`Popen.wait()`记录实际训练退出码，
再执行完整CPU终态审计并保护输出。训练或审计失败会停止后续并保留数据。

另一IsaacSim 6.1进程保持运行，GPU独占不成立；owner记录实际共存进程和显存。
启动要求至少10240MiB空闲显存，不足时等待；没有终止外部进程。

终态见[owner状态](owner_capacity_v3/owner_state.json)及[完整CPU审计](owner_capacity_v3/training_CPU_audit.json)。
[外部冷核验](owner_capacity_v3/external_verification.json)实际退出0，16来源／4输入保持；
最终checkpoint651040366字节，SHA256为`97ba4c6475817e1bbde4aa56377a45191058a0725cad48fcbf3a9b769accb675`。
最终rollout1018179842字节，SHA256为`22145c21031f8c33ec75f90ac2fc2fcce84de2e59d18bbed7a7de03efdfc47cd`。
20份权重、4份rollout及其余数据共28份只读；202张量有限性／历史／计数审计通过。
本段6000个PPO minibatch、1000个D minibatch、16000控制步／64000物理子步、2048000专家窗口已实测。
有效总曝光12500更新／102400000转换／51200000专家窗口；原12000权重、12454旧日志、AMASS manifest及bank哈希保持。
两进程退出、有效预算和完整终态审计均已通过，本次恢复训练完成。
原cell581的完成证书不会被补造；原S3连续执行、论文效果、默认模型晋升均不成立。

```bash
# 冷来源预检：不导入Torch、不启动SDK
env PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/recover_shared_social_training.py \
  --registration docs/style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/registration.json \
  --stage training --check-sources
```

输出为`output/diffusion_amp/shared_social_budget12500_recovery_training_v2`。
该登记只允许这次全新输出，不能重复启动或覆盖已经开始的运行。
训练成功和CPU一致性不等于论文效果。
[S3-R独立跟踪／风格](../../style_improvement_20261008/shared_social_S3R_evaluation_v2/README.md)
已经完成：22条／11982帧，14/22成功、零跌倒，位置0.276747m／旋转0.749612rad。
45D MMD²=.0230231，K50/K100覆盖=.58/.52；比S2距离降低、覆盖缩小，不能声称全部风格指标改善。
原200秒导航保存数据后实际退出-11，原post退出1；GDB复查正常退出0，未定位故障根因。
[独立导航续跑](../../style_improvement_20261008/shared_social_S3R_navigation_retry_v1/README.md)
已于2026-10-08 01:34:43完成，正式导航200s与原CPU数组审计实际退出均0；
固定2848、训练模型和已成功的跟踪／风格输出保持。72份资源严格比较已通过；路线停滞／风格覆盖下降尚未解决。

```bash
# 当前日志进度、保存点及由OS owner实际观察的子进程退出；不从日志推断退出0
python3 scripts/shared_social_recovery_status.py \
  --plan docs/style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/owner_capacity_v3/owner_plan.json
```

保存点的`provenance.initialization`保留原12000次训练的全新初始化祖先记录；
本次恢复阶段以`provenance.training_recovery`为准，不能用祖先字段判断本段未加载旧训练状态。
