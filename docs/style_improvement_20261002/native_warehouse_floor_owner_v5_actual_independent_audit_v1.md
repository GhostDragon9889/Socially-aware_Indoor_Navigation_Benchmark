# Warehouse v5：实际原生地图独立审核

**120项 CPU 检查通过；场景资源准入成立。** [机器报告](native_warehouse_floor_owner_v5_actual_independent_audit_v1.json)、[可复现只读审核器](audit_native_warehouse_map_v5.py)、[日志](native_warehouse_floor_owner_v5_actual_independent_audit_v1.log)。本审核器没有启动Kit/GPU。

原生251×370二维buffer共92,870格：occupied4=3,343、knownfree5=56,239、unknown6=33,288。固定官方native排布与全部occupied／knownfree世界点逐格匹配；API“free”名单中的32,786个unknown和未列出的502个unknown保持未知。

全部56,239条floor ray各有且仅有一条 `/__PaperMapProbeReferenceGround` 原生命中；完整CSR路径／normal／position／distance与nearest support一致。没有foreign或未声明共面支撑。原Plane与11个decals的collisionEnabled在实际导出session中均false，另一个只读USD子进程从原source＋runtime session复读确认。全部原source layers重新完整hash一致。

同一56,239个canonical5格已完整查询0.1m XY×地板上0.15..1.8m的native fullheight band。708条exact authored obstacle命中对应520格；55,719格无band命中且具floor支持。raw2D与API列表保留，known5被高障碍命中降occupied，原6不升级free；最终ROS图还保留原CPU projection unknown。所有3张map从raw独立重建逐像素一致。

CPU PhysX zero-gravity快照只提交两次120Hz step（1/60s），submittedcounter保持2、timeline查询前后0；不声称已经观察到native engine-time。无dynamic/animated移动，原origin净空且唯一地板支持。

此结果是有限generated reference ground＋static snapshot＋native2D与显式完整高度工程补充。它不是原USD自带地板、作者八场景地图、行人全身footprint或导航质量验收。v4此前漏检共面Plane而误报资源PASS的[退回证据](native_warehouse_fullheight_v4_floor_ownership_independent_audit_v1.json)仍保留，不因v5成功而更改。
