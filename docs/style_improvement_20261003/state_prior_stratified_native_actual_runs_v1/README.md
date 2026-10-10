# 专家分层 AMP：原生 Warehouse 实际评价

固定 q AMP 权重 `98a49058…/step_2000.pt`、旧 Diffusion `63e7…` 和生产185源码，guide20／alpha0／S1，使用现有 Warehouse 路线、同一物理参数和200秒协议。此前四人 fixture 完整零跌倒／零接触已通过，原生场景继续独立评价。

| 阶段 | 实际状态 | 跌倒／静态接触／行人接触 | 路线点／完整循环 |
| --- | --- | --- | --- |
| native4 | GPU与原始CPU审计均 exit0，全部6000控制步完成 | 0／0 actor-controls／0 pair-controls | 38／8 |
| native15 | GPU、原始CPU及90D／45D指标均实际 exit0，全部6000控制步完成 | 0／0 actor-controls／0 pair-controls | 146／30 |

## 四人实际记录

- [GPU owner](native4_actual_launch.json)：实际root session6142 exit0，completion `21a238`，SHA `2cef2eca…`。
- [CPU owner](native4_CPU_actual_launch.json)：实际root session80445 exit0，completion `f917ea`，SHA `f74d7de9…`。
- [完整原始审计](native4_raw/audit.json)：200秒，6000控制步、24000共享物理子步、96000 actor-substeps、384000足部观测；全部失败及接触记录保留。
- 各行人路线到达 `[10,9,9,10]`，完整循环 `[2,2,2,2]`；time-limit reset为0。
- 四人零跌倒／零静态接触／零行人接触只用于后续十五人物理阶段准入；没有MMD或路线到达数量的筛选条件。

## 十五人实际记录

- [GPU owner](native15_actual_launch.json)：实际root session42375 exit0，completion `b12d2e`，SHA `57ff53f1…`。
- [CPU owner](native15_CPU_actual_launch.json)：实际root session20512 exit0，completion `f9993d`，SHA `9c448fbb…`。
- [完整原始审计](native15_raw/audit.json)：200秒，6000控制步、90000 actor-controls、24000共享物理子步、360000 actor-substeps、1440000足部观测；0跌倒／0time-limit／0静态／0行人接触。
- 路线到达 `[11,6,8,11,7,14,7,12,9,15,6,9,8,16,7]`，完整循环 `[2,1,2,2,1,3,1,3,2,3,1,2,2,4,1]`，分别合计146／30。

## 已闭合的原生动作风格指标

复用[冻结的90D／45D命令](../state_prior_stratified_native_metrics_preparation_v1/commands_native_CPU_metrics_v1.json)，CUDA禁用、四线程、固定c274参考／b25基底，无refit。四人23996及十五人89985全部合法intervals／poses保留，idle／失败／reset／未知速度桶未筛除。实际tool receipts与完整SHA见[四人completion](../state_prior_stratified_native_metrics_actual_runs_v1/native4_CPU_completion_v1.json)和[十五人completion](../state_prior_stratified_native_metrics_actual_runs_v1/native15_CPU_completion_v1.json)。

| q实际结果 | 90D主MMD²／支持桶 | 45D MMD² | K50／K100覆盖 | 到达／循环 |
| --- | ---: | ---: | ---: | ---: |
| native4 | .07734562／3 | .03760272 | .58／.46 | 38／8 |
| native15 | .08017306／4 | .03683265 | .68／.57 | 146／30 |

四人共同3速度桶secondary：default `.06870245`、19f `.05244885`、q `.07734562`，q较差；旧主90有4桶，不直接按主聚合排名。十五人共同4桶secondary：default `.09969936`、19f `.06987130`、q `.08017306`；q低于default、高于19f，default主90有5桶。十五人45 MMD高于default `.02831979`及19f `.02273235`，覆盖也低于其 `.76/.61`及 `.70/.61`。

历史默认native4／15到达57／211次，19f为44／160次；q为38／146次，进展均未超过两参照。完整物理零失败不代表总体风格改善。来源168／174、183与185及初始化不同，没有bitwise物理或单变量因果改善声明。局部协议、作者场景清单及风格指标参数仍未完全对齐，默认权重保持，不晋升。
