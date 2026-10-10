# step500 误差分解补充

完整22 / 11982帧主分母，reference/clock和初始physical/history全部dtype及bitwise配对通过。原161项milestone门禁保持原样；另以新 `terminal_audit_binding.json` 绑定真实terminal220项审计中的同一step500字节身份与每组2.048M转换，不把step2000总跌倒数冒充step500。

| 同帧描述性指标 | negative1000 | mixed500 |
| --- | --- | --- |
| root XY drift (m) | 0.22834605 | 0.190085809 |
| root wrapped yaw error (rad) | 0.600986825 | 0.626892542 |
| root-centered / heading-aligned body position (m) | 0.119189275 | 0.123738751 |
| heading-relative body rotation (rad) | 0.573433925 | 0.590980765 |
| parent-relative joint rotation (rad) | 0.301823236 | 0.321445558 |
| actual planar speed (m/s) | 0.777697425 | 0.832356879 |

ALL22世界body位置误差变化 -9.726%；世界body旋转误差变化 7.369%。位置改进伴随root平移误差降低，heading-relative姿态/旋转并未同步改进；这些同帧分解是描述性证据，不是独立因果百分比。

secondary success-only：baseline11条7191帧，mixed16条9879帧；common-success为同11条7191帧。各自与共同子集的per-clip/frame两类均值保留在comparison JSON，不能替代ALL22或未知论文分母。所有baseline成功条目在mixed500中仍成功。

本阶段总4.096M转换（每组2.048M），baseline总8.192M。未进行200秒导航或真实新风格评估，未改变默认部署选择。
