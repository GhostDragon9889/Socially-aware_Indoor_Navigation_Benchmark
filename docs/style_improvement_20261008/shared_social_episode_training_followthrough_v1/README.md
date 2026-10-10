# 训练恢复与本轮收尾

## 已完成的恢复训练

[S3-R恢复](../../style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/README.md)
已从已保存的12000更新重算500次更新，训练及CPU审计实际退出均0。
最终权重：`output/diffusion_amp/shared_social_budget12500_recovery_training_v2/step_012500.pt`。
两项Adam、四项RMS、回放及训练RNG已恢复；物理世界重建，旧未保存454次更新回退。
这次恢复及后续独立评价已完成，原S3未知退出码未改写。

## 当前运行

2026-10-08 03:34:51（Asia/Shanghai）：新的episode候选172／2000次更新，
1,409,024／16,384,000物理转换，训练owner473700与SDK487451身份和父子关系核对通过。
全部完整日志行连续且有限。当前尚无候选检查点；固定计划每500更新保存一次。
本轮从新模型训练，与上述12500更新的恢复结果分别保留。

见[实际宿主观察](initial_host_observation.json)、[工具与检查记录](actual_execution.json)，
以及[训练执行说明](../shared_social_episode_after_recovery_v4/README.md)。
最近速度推算的预计时间随负载变化，训练收尾与独立评估完成前不认证终态。

## 自动收尾与查询

正式owner已脱离交互会话，训练结束后自动CPU审计、完整validation22及风格评价、
200秒导航与原始数组审计。独立CPU观察器521166随后执行预登记的同预算S2对比，
参见[比较说明](../shared_social_episode_comparison_preparation_v1/README.md)。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/shared_social_episode_status.py
```

该只读查询适配本轮入口；检查完整日志、进程命令、父子关系和owner启动时钟，
6项CPU检查／7个subtest、Ruff和实际宿主查询均退出0。
查询源码、测试与证据位于正式57来源／7输入之外，原计划来源复核实际退出0。

正式owner出现非零退出时保留失败记录并停止后续步骤；检查点存在不会被当作完成或自动恢复资格。
当前训练正常推进，应继续本次运行。论文效果和默认模型状态由实际独立评价决定。
