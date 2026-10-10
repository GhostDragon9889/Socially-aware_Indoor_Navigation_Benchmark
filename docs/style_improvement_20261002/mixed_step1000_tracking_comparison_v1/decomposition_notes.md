# step1000 误差分解补充

完整22 / 11982帧主分母，reference/clock和初始physical/history全部dtype及bitwise配对通过。原独立payload门禁保持原样，另以terminal_audit_binding.json绑定终点220项审计中的同一checkpoint字节身份与分组实际转换。

| 同帧描述性指标 | negative1000 | mixed |
| --- | --- | --- |
| root XY drift (m) | 0.22834605 | 0.115571744 |
| root wrapped yaw error (rad) | 0.600986825 | 0.53479195 |
| root-centered / heading-aligned body position (m) | 0.119189275 | 0.122855796 |
| heading-relative body rotation (rad) | 0.573433925 | 0.602473568 |
| parent-relative joint rotation (rad) | 0.301823236 | 0.321270913 |
| actual planar speed (m/s) | 0.777697425 | 0.942853686 |

ALL22世界body位置误差变化 -32.331%；世界body旋转误差变化 1.649%。分解仅描述性，不是独立因果贡献百分比。

baseline success-only：11条 / 7191帧。

mixed success-only：19条 / 10841帧。

common-success：相同11条 / 7191帧。各自与共同子集的per-clip/frame两类均值保留在comparison JSON，不替代ALL22或未知论文分母。

累计1000updates，总8192000转换，每组4096000；baseline总8.192M。equal_total_budget_to_baseline=True；each_group_has_baseline_total_exposure=False。

本工具没有运行200秒导航、官方风格重算或读取34test，也没有改变部署默认、权重或业务源码。
