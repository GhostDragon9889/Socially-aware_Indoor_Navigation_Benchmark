# Independent full22 paired matrix audit

**实际CPU PASS**。无需Torch/SciPy/SDK；独立normalized XYZW→matrix以及trace/skew atan2的SO(3) geodesic从raw NPZ重算，与原primary quaternion-dot arccos及冻结local decomposition一致；四模型每个11,982 metric frames，全部failed-inclusive。

69-import源admission由主比较记录；本审计重新核对result/capture全SHA、连续reference_frame/time与原初始reset mask。三个baseline/mixed2000、mixed1000、negative1000共66对clip，完整reference/clock与初始pose/velocity/history逐元素相同。没有time/pelvis/yaw拟合；own-root/heading去除仅在单独命名的diagnostics，不替换primary。

最大matrix-vs-primary残差：state_action2.16e−13、mixed2000 1.28e−13、mixed1000 1.26e−13、negative3.07e−13。固定刚体变换、quat antipodal及π rotation解析门禁通过。

[全部JSON](audit.json)、[执行日志](execution.log)、[正式风格比较](../full22_step2000_comparison_v1/README.md)。新功能只是一份只读审计，不改变任何capture、weights、kernel或业务代码。
