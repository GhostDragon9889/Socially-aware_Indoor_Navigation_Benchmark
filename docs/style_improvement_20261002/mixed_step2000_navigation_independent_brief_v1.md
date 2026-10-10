# mixed step2000：实际四行人200秒完整记录审核

**2,259项独立CPU检查通过**；这里确认采集／来源／时序完整性。效果尚未验收：11次跌倒、19个顶点，旧negative guide20基线0次跌倒／21顶点继续保留。

[完整机器报告](mixed_step2000_navigation_ground_actual_verification_v1.json)、[自动生成测量说明](mixed_step2000_navigation_ground_actual_verification_v1.md)、[执行日志](mixed_step2000_navigation_ground_actual_verification_v1.log)。

|项目|旧negative guide20|mixed step2000|
|---|---:|---:|
|完整控制帧／actor帧|6000／24000|6000／24000|
|actor物理子步|96000|96000|
|跌倒／time-limit|0／0|11／0|
|路线顶点|[2, 8, 6, 5]|[5,4,4,6]|
|静态接触actor帧|0|130|
|同行人接触actor帧|0|0|
|双足支持actor子步比例|48.0156%|39.3094%|

完整24000个120Hz提交步连续，每控制4子步；记录15个reset标记＝4初始＋11后续。全部11个terminal对应真实跌倒；无删帧／补零。共424次joint规划／1696条actor计划，全部每15步及reset／route-arrival强制规划、raw51／executed151最近计划与post-step相位、速度／deviation／端点残差逐帧核验。按actor／episode实际完成3个完整四顶点循环，不合并不同actor的零碎进度。

真实COM独立重构最大坐标误差1.01e−6m，总质量63.313949kg；native ankle/toe路径、地板唯一filter、1N rawforce支持、末子步body／Pelvis／episode与physical原始四接触字段一致。foot-origin速度仍是代理，AMASS参考trueCOM与摩擦／接触点滑移未知。历史probe false保留；actual_native_capture_validated=true仅针对本完整记录。

全部跌倒时刻（秒）：

- actor3：16.100000秒，episode1；terminal当帧static_contact=false。
- actor2：25.766667秒，episode1；terminal当帧static_contact=true。
- actor2：49.566667秒，episode2；terminal当帧static_contact=false。
- actor2：60.400000秒，episode3；terminal当帧static_contact=false。
- actor1：73.500000秒，episode1；terminal当帧static_contact=true。
- actor1：92.133333秒，episode2；terminal当帧static_contact=true。
- actor3：125.233333秒，episode2；terminal当帧static_contact=true。
- actor0：147.033333秒，episode1；terminal当帧static_contact=true。
- actor1：164.333333秒，episode3；terminal当帧static_contact=false。
- actor0：172.500000秒，episode2；terminal当帧static_contact=true。
- actor2：182.866667秒，episode4；terminal当帧static_contact=true。

138来源manifest285e4126…／tar a06eae3d…、三候选注册参数及实际220项终点admission全部绑定；step2000模型SHA28aed874…全文件重算一致。actual runtime参数逐键逐类型等于预注册，场景/live physics与旧基线一致，仅profile／模型／output参数不同。模型训练课程与预算不同，共享seed不保证同一物理轨迹，不能作单因素归因。原34测试与姿态MMD协议未由本审核读取／重算。
