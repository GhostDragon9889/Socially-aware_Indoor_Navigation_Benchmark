# 实际30秒 alpha=.25 手臂混合：同长度冻结对照

两份900步/30秒/4人全部保留。CPU评分/原生采集完整性通过；短时效果不替代200秒稳定性，不提升默认控制器。

|指标|pure baseline|alpha=.25|
|---|---:|---:|
|fall|0|0|
|路线顶点|2|3|
|inclusive MMD²|0.0718073804|0.0747329079|
|速度W1 m/s|0.207179082|0.135578491|
|实际差分速度mean m/s|0.709771319|0.778372113|
|idle总actor秒 / 120秒|5.03333333|2.76666667|

主MMD略退，.1–.5m/s bin .032663878→.107676361变差；<.1m/s bin .160563718→.117759450改善。unsupported bins保持unknown。速度分布接近参考、路线2→3不等于风格整体改善。

每份3600 actor-frame保留；合法same-episode导数3596、加速度triplet3592，没有fall/reset帧过滤。c274参考与82ee/b09原90D/sigma1/速度bins/min30/max256不改。

## 原生支持、接触与足部代理

原测量auditor函数未改，alpha真实231检查通过，baseline复用此前实际900步验收并重新绑定result/NPZ完整SHA。均无静态/peer接触；这只是此次30秒，不是普遍collision-free。

双支持 32.437500%→27.062500%；无足支持 1.111111%→0.736111%。支持条件足部原点切向速度mean 0.099431→0.101957m/s。它包含滚动/旋转，不能当contact-point slip。COM真实重构最大误差约1e-6米；AMASS真实COM/contact参考未知。

## 实验与来源边界

baseline62c4…与donor28aed…，仅24个Thorax/Shoulder/Elbow/Wrist归一化raw mean坐标按alpha混合，再统一clamp/57mask，单次物理PD解码。未执行critic/D/训练transition。跨不同观测下不能声称其余raw坐标逐位相同。

源146 baseline与源154 alpha采用各自实际预启动archive，完整控制For AST相同但world/contact/simulator/可选native等文件不同。native branch未启用；不拿后来的provider编辑或current tree替代历史，不宣称完整运行时GPU import map。

停留定义和plan phase沿用原函数；只将可选90/105/120秒诊断snapshot改为10/20/30秒，避免越界，不改指标。yaw/wrist仅描述性，没有逐帧身体旋转真值。

证据：[比较JSON](comparison.json)、[冻结原评分](frozen_style/evaluation.json)、[原生验收](native_ground_comparison.json)、[停留/phase](stagnation.json)、[yaw/wrist](descriptive_yaw_wrist.json)、[参考](frozen_style/heldout_reference.json)。CPU-only；未读取sealed34、改业务/weights或启动Kit/GPU。
