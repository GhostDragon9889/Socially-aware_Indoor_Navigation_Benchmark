# 官方生成器Maps原始数据：实际CPU核查

1000场景、2000个原始文件全部读取并保存SHA，预登记Random(0)分割800／100／100未变。实际CPU进程25657已观察exit0（completion470fa5），耗时3.9498秒；见[完整QC](actual_QC.json)和[真实闭合记录](actual_CPU_closure.json)。原始文件由root随后独立核对并[封存](../orca_maps_official_raw_actual_runs_v1/raw_root_seal.json)，本reader未修改它们。

- 5640名行人，569640个保存的agent-frames；行人1–10、障碍0–20、真实半径0.4m、初速0、属性范围／全部状态和polygon数值有限，polygon结构有效。
- 45个空地图全部保留；956种geometry，重复的空geometry跨split存在合法。scene／raw身份按预登记划分，没有按geometry排除场景。
- 真实初始状态单独保存。trackTime首帧0.0166666675s、末帧10.0000190735s，首99段名义0.1s、最后段5／60s，累计时钟误差最大0.000049337s；不是严格均匀10Hz，也不能把首帧改标t0。
- 初始圆盘／障碍重叠均0。保存的状态中有114次无序pair圆盘重叠观测、25次障碍圆盘重叠观测（gap＜−1e−4m）；最坏间隙−0.003169m／−0.151476m。全部保留。这是保存采样的纯几何诊断，不是Isaac接触，不证明60Hz完整路径。
- 官方`USE_BOUNDARY=False`，保存状态最大坐标绝对值19.1261m合法，不能套用旧localMaps的有界／非空／4–15人／唯一geometry条件拒样本。

这批资源是固定官方TRACE生成器及参数、当地seed20261005新生成的数据，不是作者原始数据。格式／数值QC通过不等于无碰撞专家轨迹；未读取权重、模型、SDK、GPU或封存34条测试输出。

后续[独立Maps转换源](../orca_maps_official_usable_staging_v1/convert_official_maps.py)直接复用已执行Interact的真实t0＋时间戳插值及neighbor-only窗口。Maps的map presence始终True，包含真实空地图；不要求窗口可见障碍，不删重叠、empty或zero-window场景。
