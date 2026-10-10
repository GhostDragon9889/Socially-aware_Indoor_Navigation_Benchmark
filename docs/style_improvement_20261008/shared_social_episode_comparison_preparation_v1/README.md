# episode候选：完成后的同预算独立对比

## 执行状态

2026-10-08 03:12:40（Asia/Shanghai）启动独立CPU观察进程521166。
它等待[正式训练与评估owner](../shared_social_episode_after_recovery_v4/owner_state.json)完成，
以实际`/proc`命令和启动时钟核对owner473700。
[观察状态](comparison_owner_state.json)仍是`waiting_for_actual_episode_completion`；比较尚未执行。

正式owner依次执行8人预检、CPU审计、256人预检、CPU审计、2000更新训练、CPU审计和独立评估。
七阶段都实际退出0并通过来源准入后，本观察进程自动调用预登记的CPU比较；
子进程退出码由`Popen.wait()`记录，最后写出`comparison_completion.json`。
进程失败或来源变化时保存失败记录。已有输出目录不能重复启动。

## 固定比较条件

[登记](registration.json)预选候选`step_002000.pt`和已完成的S2终态，
两者均从新模型训练2000更新／16,384,000物理转换。
保持真实AMASS 2848清单、完整validation22／11982控制步、冻结风格参考与原数值评分代码。
跌倒、失败及其后续捕获帧全部保留，test34不参与模型选择。
90D速度桶只有样本数与参考索引都相同时才用于方向性比较，其余为unknown。

候选包含episode固定程序参考、末端缓冲／私有RNG曝光与显式CPU／CUDA时钟舍入改变，
因此不能将差异归因于单一变量。当前评估是本地validation22与N4／200秒，
作者八场景与私有协议尚未恢复，不能由该比较认证论文完整效果。

评估父入口有两项显式来源扩展，原子评估入口保留原训练准入。
比较分别严格验证完整父准入和原子准入，只有两个预声明扩展不同且全部核心字段相同才接受。
私有比较视图不会修改原始结果或评分数值。

## 已执行检查与查询

6项比较CPU检查、11个subtest和2项进程身份检查实际退出0；Ruff通过。
登记生成与冷来源预检实际退出0，见[执行记录](CPU_preparation_actual_execution.json)。
11份实现来源及观察器／测试来源快照保持；尚无候选终态比较或效果改善结论。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/shared_social_episode_status.py
```

查询在宿主环境中核对进程，并读取完整、连续、有限的日志行；不加载模型或启动SDK。
训练收尾、数值审计、独立评估和比较全部通过前，默认权重保持。
