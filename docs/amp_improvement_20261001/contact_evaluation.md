# 真实行人接触记录的 CPU 评估合同

状态：**66 项测试通过（28 项新增接触回归，38 项原评估／CLI 回归），Ruff 通过。** [contact_evaluation_tests.xml](contact_evaluation_tests.xml)。没有启动 Kit／GPU。

`rollout_metrics` 与 CPU CLI 可选读取 `static_contact[T,B]`、`peer_contact[T,B,B]`、`static_contact_max_force_n[T,B]`、`peer_contact_max_force_n[T,B,B]`，runner 以 `physical_contact_measurement` 声明合同。NPZ 出现接触字段时，四项必须齐全；旧 NPZ 不含这些字段仍保持 actual contacts 为未知。

合同来自 PhysX exact-path `get_contact_force_matrix` 的法向力、单位 N，排除地面支撑与自身 articulation。每控制帧保存所有物理子步中的最大 body-pair 力幅度，并以严格 `force > threshold_n` 定义接触。当前默认阈值1N、4×120Hz 物理子步／30Hz 控制；有 metadata 时严格验证4子步及这两个 dt，不以数组形状伪证明实际 SDK 采样。布尔 mask 必须 genuine bool，力必须 finite、非负浮点；shape、阈值一致性、自碰撞对角线均 fail closed。实际 SDK 读数与采样正确性由 root 的真实 probe 提供，本 CPU 检查不自行证明 physics。

`actual_contacts` 分开报告：

- **static actor-frame**：某 actor 任意 body 触碰声明静态 collider 的帧数，以及连续事件和受影响已观测 actor episode；不是 pelvis 点占据。
- **peer actor-frame**：某 actor 任意 body 触碰其他 actor 的帧数；同一 pair 同时涉及两人时计两个 actor-frame。
- **unordered pair-frame**：上三角 `i<j` 的接触帧数，一对行人每帧只计一次。两个方向的 contact view 用 OR／force max 合并，方向布尔不一致 entries 单列。
- **contiguous pair incidences**：任一参与者发生 episode／reset 边界、出现无接触帧或缺失有效记录时切断事件；与 actor 的“持续触碰任意人”统计不同。

最大力是 body-pair 幅度的最大值，不是各 body 力的总和。单 actor 没有可观察 pair，pair 帧分母为0、比例与最大 pair 力为 `None`；没有证据不返回伪零。

若旧 `map_collision_mask`／`agent_collision_mask` 没有提供，可从上述真实 body contacts 填充对应旧顶层字段，并标清 PhysX 来源与非作者协议。已有显式旧字段不被覆盖；真实统计仍保留在 `actual_contacts`。root-point cuboid 与 XY-circle 几何代理继续单独报告，不能替代 body contacts。论文作者的帧／episode／序列聚合协议未公开，本地计数不声称等同作者碰撞率。

CPU 回归包含手工可数的3人5帧事件、reset 跨帧拆分、无序 pair 不重复、非对称方向、无效帧缺口、严格1N边界、单人空分母、旧代理保留、22类非法合同／数据与真实 NPZ 字段读取接口。所有接触测试数据为合成接口输入，不能误当 SDK 实测。
