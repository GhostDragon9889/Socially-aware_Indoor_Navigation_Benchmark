# 完成导航记录的运动与脚部支撑诊断

入口：[`diagnose_shared_social_locomotion.py`](../../../scripts/diagnose_shared_social_locomotion.py)。仅 NumPy/stdlib；不加载权重，不启动 SDK/GPU。读取已经实际正常退出的 S0/S1 四人200秒导航及其独立CPU审计，并逐字节核对 result/arrays/回执。S1同时核对五个成功child、pipeline及owner的本方测量身份。

## 实际结果

两侧均保留全部24,000 actor-controls、96,000 actor-substeps／800 actor-seconds，包括跌倒、接触、reset和停滞。没有按成功筛选或拟合时间／坐标。

| 本地诊断 | S0 | S1 |
| --- | ---: | ---: |
| 跌倒终止 | 956 | 5 |
| 同control root→执行参考距离均值，m | .248155 | .083154 |
| 同control root→原始DP目标距离均值，m | .144609 | .053367 |
| native末子步pelvis XY速度均值，m/s | .913903 | .184023 |
| native末子步pelvis XY速度中位数，m/s | .930497 | .005760 |
| 执行参考XY速度均值，m/s | .972276 | .253320 |
| 执行参考XY速度中位数，m/s | 1.034638 | .102451 |
| 同episode有效位置间隔数 | 23,040 | 23,991 |
| 上述间隔内速度<.1m/s的间隔数 | 131 | 19,303 |
| 双支撑占全部actor-substeps | 8.673958% | 90.435417% |
| 左／右原始接触onset数 | 1,158／686 | 188／191 |
| 四人最长连续双支撑，s | .883／.542／.483／.800 | 160.042／148.717／153.050／175.625 |
| 执行参考速度≥.5m/s的actor-controls | 20,494 | 3,677 |
| 其中native速度<.1m/s的actor-controls | 103 | 19 |

S1有效位置间隔的80.46%低速，四人存在约149–176秒的连续双支撑。低跌倒与距离降低不能证明步态完整改善。另一方面，**在各自速度≥.5m/s的执行命令内**，native低速比例仅为S0 .5026%／S1 .5167%；两侧命令分母不同。因此目前不能归因“有大量移动命令却完全不执行”，还应检查DP原始预测、参考适配和反馈历史的联合行为。这是描述性关联，没有因果定位。

## 时序及测量定义

- 原native ground轴为 `[control, substep, actor, body/channel]`，不是actor/substep互换；每control四个120Hz子步、30Hz控制。完整整数时钟、valid_count、episode/reset/terminal逐项核对。第四子步的Pelvis pose与physical root精确相同，quaternion为unit XYZW。
- 根据已冻结的历史CPU reader重建全部6000 control目标：最新已发生的plan，`age=control−plan_start+1`，仅1..15。raw51@10Hz线性插值，reference151@30Hz与refV直接取age。target/refV最大误差0，raw最大误差约6.36e−7m。每phase各1600 actor-controls。
- root→执行参考距离与root→原始DP目标距离单独报告。`reference_deviation_m` 是执行参考减raw目标的范数，不是root跟踪误差。它们都**不是论文未公开的全局Path Error实现**。
- 从实际FP32 native floor法向力矢量重算各foot norm严格>1N，再把每侧Ankle/Toe的foot flag做any；与全部已保存foot/side flag精确一致。无debounce、无按跌倒删除帧。这些onset只是阈值跨越，不能称为步数／cadence或接触点滑移。
- 支撑时长停止于episode边界或记录结束；跨reset的S0 956／S1 5个actor区间不产生接触onset/release，但全部接触状态仍保留。每actor首样本没有前态。
- 前一位置间隔仅同episode且非reset有效，其余保持undefined；同control原生速度、距离和接触仍全部保留。

## 实际执行与复现

S0 CPU实际退出0，tool chunk `906e14`；S1实际退出0，`964a7c`。
独立只读复核实际退出0，`050d4b`。四项最终测试实际退出0，`ef12c5`；Ruff实际退出0，`df638b`。测试加入跨reset `False→True`／`True→False`，独立review发现原三项未覆盖该变异；实现无需修改。
初次pytest被环境ROS插件缺少lark阻断（`2b9ba8`，exit1），随后显式禁用无关插件自动加载。初次Ruff发现unused import（`b62c5a`，exit1），已修正。失败及实际成功分别记录于 `execution.json`。

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/diagnose_shared_social_locomotion.py \
  --navigation-receipt output/diffusion_amp/shared_social_popart_terminal2000_evaluation_v1/navigation200s_execution.json \
  --audit-owner-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_execution.json \
  --output /tmp/S1_locomotion_new.json
```

输出必须新路径。两份实际JSON、源码快照、测试／Ruff日志和来源哈希见同目录；源和输入在读取前后复核。S2仍由既定owner负责训练及独立评价，本阶段没有修改22项guarded来源、训练／默认权重或评估协议。
