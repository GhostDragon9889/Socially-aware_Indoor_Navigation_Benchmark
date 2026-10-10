# 原生N3–15 raw审计：CPU准备与历史数值回归

2026-10-05。本记录只发布CPU接口、实际子进程退出与历史数值保持证据。
没有启动新的SDK任务，没有新增原生人体捕获或模型质量结果。

## 已执行

- `run_CPU_preparation_v2.py`外部实际退出0：launch `c37572`，terminal `2972c4`。
- 四项源码检查：N3/N15 × 30/6000控制步，显式DP/SFM；均退出0，无Torch/SDK。
- 最终62项CPU测试通过，2.40秒；Ruff退出0。覆盖人数、时长、数组轴、全部24身体、
  原生子步、计划首锚点、时钟、实际driver参数ABI、场景/provider及释放声明的拒绝路径。
- 条件突变：首个实际退出校验的`is int`改为`is not int`；实际21失败、41通过，退出1。
  finally恢复原字节；之后全部62项通过。突变版本单独保留，不作为生产代码。
- 历史S1的DP/SFM两份真实6000步N4捕获：base除schema外每个字段、全部body pose、
  response/support字段完全相同；joint仅显式更名人数标签。输入档案前后SHA不变。
  历史入口是cuboid fixture；未调用新的原生场景、初态或SDK准入，不能计为N3–15实测。
- S3训练23项来源guard不变；未改训练数据、采样、参数或默认权重。

## 保留的失败及修正

v1外部实际退出1：launch `c91083`，terminal `0b4919`。
六个CPU子进程均通过，但读取历史结果时产生`KeyError: native_dynbench`。
该字段属于部署入口参数，不存在于底层driver的`arguments`。

生产reader改为检查实际输出的native grid、provider及live stage/floor/source记录。
新增测试使用已封存preflight的真实driver参数列表，确认ABI不包含该字段。
v2历史回归则显式检查旧N4入口和cuboid scene schema，并拒绝任何native结果字段；
两种准入各自明确，不使用默认mode flag。v1注册、源码、测试、突变和日志均保留。

## 入口和曝光单位

生产入口为`../../../scripts/audit_shared_social_native.py`，只接受显式N3–15、
30或6000控制步、DP/SFM、scene/SHA及外部实际退出receipt。
新receipt schema为`shared_social.actual_native_navigation_execution.v1`；需绑定
`actual_exit_code`（严格整数0）、`terminal_tool_chunk`、`actors`、`control_steps`、
`planner`、`scene_configuration`、`result`及三份`raw_arrays`身份。
每份身份包含绝对path、bytes、sha256；reader再检查实际初态、权重来源、场景、时钟及释放。
声明校验本身不独立观察SDK退出，外部观察者仍必须保留真实工具退出。

对C个控制步、N人，分别报告C×N actor-controls、C×4全局physics-substeps、
C×4×N actor-substeps及C×N×(N−1)/2无序pair-controls曝光。
4个物理子步、4元数分量和4脚部传感器不随人数变化。失败、接触及reset帧全部保留。
6个来源固定内核共43项literal替换，完整逆变换通过；指标公式不变。

## 仍需验证

实际原生SDK smoke、两臂完整World初态、200秒多人数/多seed评价尚未执行。
原生独立风格reader、多seed入口及作者八场景/指标协议仍待补齐。
CPU合成数组仅用于拒绝路径和数学回归；没有作者分母或达到/超越论文的结论。

`root_observed_CPU_execution.json`保存根代理观察到的两次外部退出；
`CPU_evidence_seal.json`保存只读文件索引；最终封存工具退出单独记录。
