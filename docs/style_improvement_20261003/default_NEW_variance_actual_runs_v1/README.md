# 方差引导单变量：完整物理负结果

使用默认AMP `62c4…`、新地图Diffusion `b563…`、S1采样与同一生产185源码，唯一方法变量为 `trace_clean → trace_clean_variance`。路线、随机种子、参考适配器、PD控制及接触测量保持原设置。

完整四人200秒／6,000控制步已完成。GPU root session7932，`09262b`→`cfe683`实际退出0；CPU raw session62078，`4cbaae`→`2d00c5`实际退出0，203项原始数据检查通过。所有产物实际关闭后封存，失败和idle均保留。

| 指标 | 实际结果 |
| --- | ---: |
| 跌倒／time limit | 0／0 |
| 静态接触actor-controls | 0 |
| 行人接触pair-controls | 315 |
| 行人接触actor-controls | 630 |
| 到达路线顶点 | [3,2,1,3]，合计9 |

对照新DP原clean设置的1次跌倒／70行人接触pair-controls／9次到达：方差引导减少了本次跌倒，但增加了行人接触，没有改善到达数。旧默认DP同场景为0跌倒／0行人接触／21次到达。因此该单变量未通过零跌倒／零接触物理门槛，不晋升默认，不启动该设置的native扩展。

“203项通过”只代表数据及测量核对成功，不能称物理质量通过。[GPU实际记录](fixture_new_actual_launch.json) · [CPU实际记录](fixture_new_CPU_actual_launch.json) · [完整raw检查](fixture_new_audit.json) · [登记与唯一变量](../default_NEW_variance_preparation_v1/README.md)。
