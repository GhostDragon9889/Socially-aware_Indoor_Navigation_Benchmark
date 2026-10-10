# Episode固定参考：独立混合课程与World入口CPU准备

**真实2848数据接口、World准入和66项CPU检查已通过；SDK模拟、训练和动作效果尚未验证。**

## 候选改变

[此前全量重锚定诊断](../shared_social_procedural_anchoring_CPU_v1/README.md)发现，
原程序分支每15控制步从当前根部重新生成路径，会在声明的静止反事实中持续保留高轨迹奖励。
这提供了独立训练对照的理由，尚未证明实际策略失败原因。

新[混合课程](../../../CrowdSim/humanoid/training/episode_mixed_trajectories.py)
让程序分支只在episode重置时生成参考，随后用绝对episode年龄查询同一路径。
周期重规划只创建该参考的当前视图，不从当前物理根部重新起步。
AMASS分支仍采用原future31、固定半数actor分配、真实来源绑定和专家窗口规则；没有合并future81候选。

正式物理episode上限仍为300控制步（10秒），参考覆盖10秒加5秒未来查询。
8人短检查的物理上限仍为4步，参考缓冲上限向上对齐到6步以匹配0.1秒路径节点。
该改变也移除了旧局部5秒参考的末端保持，并改变程序路径长度及后续私有RNG曝光；
这是明确登记的本地参考机制实验，不能称单一标量消融或作者训练分布的精确复制。

[独立训练入口](../../../scripts/train_shared_social_episode_reference.py)
使用调用方私有包加载原训练实现，只在World创建前替换尚未使用的课程对象。
原World的四处来源／类型／配置准入改动见[source_derivation.json](source_derivation.json)，
逆向恢复原字节；物理、PD、奖励、控制、重置和清理代码保持原实现。
入口来源差异见[entry_source_derivation.json](entry_source_derivation.json)。

## 实际检查

[真实数据报告](actual_entry_probe.json)和[实际工具退出](tool_execution.json)：

- 固定2848 bank的全部来源ID、片段数、帧率及清单绑定检查通过。
- 8人和256人两种规模执行实际私有入口、原PD依赖及World类型／来源／配置准入。
- 两种规模的真实历史、next参考、reset来源／RNG、原AMASS原生轨迹与未来目标逐值保持。
- 每种规模各4096个专家来源窗口逐值保持。这是CPU来源接口检查，不是实际判别器训练曝光。
- 首轮原生151目标及程序路径前51节点逐值保持；256人第15步查询固定参考，声明的live-root位移不重抽路径。
- 混合参考状态恢复逐值一致；原bank抽样RNG由调用方管理，另行恢复，不伪称纳入课程状态。
- 41项直接实现来源、4项输入和23项在训S3来源guard保持。全局RNG未消费，CUDA、NN、SDK未初始化。

最终66项检查包含本候选课程9项／入口18项、已有episode参考9项及原混合课程30项；Ruff通过。
真实接口工具会话65657实际退出0。探针只执行World构造器的准入语句，未创建物理World；
保存的根部位移是声明的CPU反事实，不能当作SDK捕获或闭环效果。

首轮课程测试误把外部bank RNG当作课程状态，首轮真实探针重建时误用seed999；
严格状态准入分别拒绝。纠正测试中的RNG恢复及注册seed7后通过，生产状态检查保持。
[首轮检查](initial_checks.json)、[探针失败](initial_private_probe_failure.json)和当时源码均保留。

## 后续执行

[登记计划](execution_plan.json)固定顺序：8人SDK短检查 → 256人SDK规模检查 → 全新2000更新训练。
每阶段必须观察实际退出0并完成CPU审计，下一阶段才可开始；选定终态为`step_002000.pt`。
三个配置是原短检查／S2配置的逐字副本，训练不继承短检查权重。
当前三个阶段均未执行，入口要求S3训练和整条后续评价链完成的外部证书，GPU任务并发上限为1。

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/train_shared_social_episode_reference.py --check-profile \
  --config docs/style_improvement_20261006/shared_social_episode_reference_runtime_preparation_v1/training.json
```

三项配置预检均实际退出0；`--check-ready`因尚无完整S3证书实际退出1，没有启动SDK。
训练后需完整validation22、冻结45D／90D风格、四人200秒物理／路线和同AMP的DP／SFM配对评价。
原生多地图／N3–15／多seed必须为这个新终态另行登记权重绑定，不能填入已有S3权重位置。
本批不能证明动作风格、路线或物理表现改善，也没有达到或超越论文效果的结论。
