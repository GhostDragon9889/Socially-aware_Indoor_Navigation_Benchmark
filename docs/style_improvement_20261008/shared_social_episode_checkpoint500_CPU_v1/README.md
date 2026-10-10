# episode训练第500次检查点：实际 CPU 恢复完整性核验

本轮正式训练保存首个 `step_000500.pt` 后继续推进。
本批在完成至少第501次日志更新后，将原子发布的第500次文件复制到独立只读目录，
核验原文件前后哈希及副本字节保持，未修改训练文件或重启训练。

调用原显式 Pop-Art／SocialAMP 严格加载器，指定 CPU 和 `restore_rng=False`。
构造并加载完整网络与学习状态后，将加载值与保存状态逐值比较；实际退出 0，Ruff 通过。
没有进行前向推理、优化器更新、SDK启动或模型选择。

| 检查项 | 实际结果 |
| --- | --- |
| 保存更新／转换 | 500／4,096,000 |
| 完整模型 | 原严格 schema／来源／形状验证，加载后逐值一致 |
| 两套 Adam | 参数组、步数及一／二阶矩保持 |
| self／task／value RMS count | 各4,096,000 |
| AMP逐帧 RMS count | 61,440,000 |
| 回放数据 | 65,536条，历史及实际动作逐值一致 |
| RNG | 原 schema 验证；未恢复当前进程全局RNG，CUDA字节仅结构验证 |
| CUDA／SDK | 均未初始化／启动 |
| 原始权重／独立副本 | 各651,041,646B，SHA相同 |

**本批证明保存的学习状态可由原加载器读取，不证明完整物理恢复。**
该检查点没有保存物理 World、episode历史或课程私有状态；
如以后从它恢复训练，需要显式重建这些部分并声明差异。
当前活跃训练继续执行，未因该检查启动第二个训练任务。

第500次是中间检查点，不能代替预选的第2000次终态、真实训练退出、终态CPU审计或独立风格／物理评价。
默认权重保持，本批没有论文达标证据。

## 复现

选择尚不存在的独立输出目录；原训练源文件保持。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/inspect_shared_social_intermediate_checkpoint.py \
  --update 500 --output /tmp/new_episode_checkpoint500_inspection
```

证据：[实际CPU执行](actual_CPU_execution.json)、[完整核验报告](result.json)、
[只读权重副本](step_000500.pt)、[来源副本](source_snapshot_index.json)。
