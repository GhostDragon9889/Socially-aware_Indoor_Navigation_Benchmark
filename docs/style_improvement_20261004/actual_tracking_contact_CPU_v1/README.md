# 已执行轨迹与真实接触的 CPU 诊断

实际CPU会话 **17342**，启动块 `88d171`、完成块 `8f48e9`，观察exit0；NumPy诊断耗时 **1.57447秒**。[完整结果](actual_diagnostic.json)、[全部72,000行](all_actor_control_rows.csv)、[真实工具与封存记录](actual_CPU_tool_receipt.json)。这是旧封存仿真的新诊断，没有训练或运行新物理仿真。

## 实际结论

| 组别 | 参考root误差均值／P95（m） | 参考代理unsafe pair-controls | 实际root代理unsafe | 真实peer pair-controls | 参考代理安全时仍接触 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLD63 S1 | .11504／.27968 | 128 | 372 | 0 | 0 |
| NEWb563 S1 | .14190／.29865 | 625 | 794 | 70 | 24 |
| NEWb563 S10 | .14138／.31901 | 576 | 1069 | 279 | 118 |

S10参考代理unsafe减少，但实际root代理unsafe及接触增加；118／279个contact pair-controls对应末端参考代理safe。接触actor-controls的跟踪均值S1=.23289m、S10=.43088m，P95=.56526／.98446m；全帧均值几乎相同，不能称全局跟踪显著恶化。接触也可能改变root位置，以上是观察关系，不是因果。

反例与收益完整保留：OLD的root代理unsafe372却peer0；S10 root代理unsafe且没有contact790，静态root unsafe52而static0。S10到达18次、idle4688个合法actor-interval，S1为9次／7207；不只挑物理负数。三组全部0／1／3跌倒、4／5／7重置首控制和接触onset0／3／13保留。每组24,000 actor、36,000 unordered-pair分母；onset持续controls总数准确回到0／70／279。

原计划到每个已执行目标的相位复核：raw最大差9.54e-7m，reference XY／velocity均0；这排除了本诊断中的目标索引错位，但不能定位物理失败原因。独立源码及实际JSON审阅均Required0，root另外实际核验CSV全72,000个唯一actor/control记录。

## 输入和方法

输入是已完成并封存的 OLD63 S1、NEWb563 S1、NEWb563 S10 三组四人、200 秒记录。脚本先核对原比较记录、真实 GPU/CPU closed0 owner、PASS raw audit 以及整份 NPZ 的只读文件字节身份，再读取所需 NumPy 数组。不会重跑原物理审计、加载模型、启动 SDK 或读取 34 条保留测试动作。

诊断保留每组全部 6,000 × 4 个控制帧：

- 比较实际 post-step root XY 与同控制相位 raw / adapted target 的误差，报告完整、接触、非接触、跌倒、重置及静止分布。
- 对三种位置分别计算六组无序行人对净空（半径 0.6 米）和真实 fixture YAML 的 AABB 净空（半径 0.8 米），与原 PhysX 接触标记形成计数表。
- 输出完整 72,000 行 actor CSV，以及每个真实接触 onset 前后 15 个控制步的小窗口。实体 episode/reset 字段直接来自物理档案；不由 raw plan t0 推断。

位置及目标是控制步末端状态；接触标记汇总控制步内四个物理子步的最大法向力，严格大于 1 N。它不是接触子步的精确位置；peer/static 逐子步力序列和原 body-pair force matrix 未单独存档。代理净空严格小于零才记为 unsafe。AABB＋边界和仅 AABB 两类静态代理分别报告，边界代理并非额外物理 collider。代理与接触的关联不能证明碰撞、跌倒成因或论文效果。

onset 是无序对（或静态 actor）记录标记 False→True，或首帧/实体 episode 边界的 True；不是首次物理碰撞，首帧以前的状态不可推断。窗口可以重叠并在档案边界截短，窗口行数不是独立全时分母。

实际执行命令如下；已有输出时脚本拒绝重复运行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/actual_tracking_contact_CPU_v1/diagnose_tracking_contact.py
```

脚本拒绝覆盖 `actual_diagnostic.json` 或 `all_actor_control_rows.csv`。结果已由root在真实工具闭合后登记并封存；本次没有加载权重、Torch、NN、SDK、GPU或34条保留测试输出。事件窗口另存本无序对最大力，明确区分每个actor对所有伙伴的最大力。
