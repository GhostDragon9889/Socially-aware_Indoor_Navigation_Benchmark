# mixed step1000：实际四行人200秒独立审核

**2,259项 CPU 检查通过**，确认来源、完整捕获和时序；效果未验收。13次跌倒／15顶点，旧negative guide20的0次跌倒／21顶点保留。

[完整JSON](mixed_step1000_navigation_ground_actual_verification_v1.json)、[测量说明](mixed_step1000_navigation_ground_actual_verification_v1.md)、[日志](mixed_step1000_navigation_ground_actual_verification_v1.log)。

|项目|旧negative guide20|mixed step1000|
|---|---:|---:|
|控制帧／actor帧|6000／24000|6000／24000|
|actor物理子步|96000|96000|
|跌倒／time-limit|0／0|13／0|
|路线顶点|[2,8,6,5]|[6, 2, 5, 2]|
|静态接触actor帧|0|1|
|同行人接触actor帧|0|20（10无序pair帧）|
|双足支持actor子步比例|48.0156%|47.8292%|

24000个120Hz提交步连续，全部6000控制帧、13个失败终点与17个reset标记（4初始＋13后续）保留；无删帧或补零。424次联合规划／1696条actor计划逐帧核对raw51与executed151、replan/reset/route-arrival强制时钟与post-step目标。按actor／episode完成1个完整四顶点循环，不合并零碎顶点。

trueCOM独立重构最大误差9.96e-07m，总质量63.313949kg。脚力1N支持、native foot paths／floor filter、末子步pose／episode与physical原4个接触字段全部核验。foot-origin速度是代理，AMASS真实COM／接触点滑移未知。历史nativeprobe false保留。

全部跌倒时刻（秒）：

- actor3：11.233333秒，episode1；terminal static_contact=false。
- actor2：12.933333秒，episode1；terminal static_contact=false。
- actor3：22.333333秒，episode2；terminal static_contact=false。
- actor1：45.666667秒，episode1；terminal static_contact=false。
- actor3：46.733333秒，episode3；terminal static_contact=false。
- actor0：59.666667秒，episode1；terminal static_contact=false。
- actor3：62.233333秒，episode4；terminal static_contact=true。
- actor3：85.000000秒，episode5；terminal static_contact=false。
- actor3：108.300000秒，episode6；terminal static_contact=false。
- actor0：121.400000秒，episode2；terminal static_contact=false。
- actor3：127.066667秒，episode7；terminal static_contact=false。
- actor0：132.633333秒，episode3；terminal static_contact=false。
- actor3：197.466667秒，episode8；terminal static_contact=false。

138来源manifest／archive、预登记参数、terminal220模型准入完整绑定；step1000 SHA76708bfa…全文件重算一致。实际argv逐键逐类型与预登记相同，场景／live physics与旧基线相同。课程／模型／预算不同；共享seed不保证相同物理轨迹，不作单因素归因。本审计未读取封存34测试或重算姿态MMD。

[12:25:21UTC当前138字节核验](mixed_navigation138_current_identity_after_step1000_v1.json)全部匹配；实际报告使用immutable archive而非后续current tree。
