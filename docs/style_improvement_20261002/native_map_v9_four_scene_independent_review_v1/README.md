# 四场景 V9 真实产物独立核验

**4,441 项只读 CPU 检查通过**。核验读取已完成的原生 SDK 产物，不重复 provider 准入或配置生成，不启动 Kit/GPU。最终报告为 [actual_artifact_review_complete.json](actual_artifact_review_complete.json)。早期 4,400 项报告与 reader API 错误记录保留，后者只是独立工具误用 `Sdf.Layer.GetInfo`，未形成场景失败结论。

| 场景 | 源文件层数 | known5 全量 floor/height 查询 | API 缺失 known5 | 高度障碍格 | 近地非地面障碍格 | 原生下边界 padding XY |
|---|---:|---:|---:|---:|---:|---|
| Warehouse | 53 | 56,239 | 0 | 520 | 0 | 0,0 |
| Office | 543 | 223,776 | 3,532 | 6,094 | 1,289 | 0,1 |
| Hospital | 293 | 112,006 | 0 | 1,931 | 366 | 1,1 |
| Full_Warehouse | 1,851 | 155,300 | 0 | 890 | 12 | 1,1 |

合计 **547,321 个 known5 格**均保留实际全高度查询和地面 all-hit CSR，全部恰有一个声明地面的支持命中。所有额外近地支持路径必须属于原源几何证明的静态非地面物体，并降为占据格；没有未声明/重复地面支持。

原始 native buffer 的 4/5/6、API 占据/自由坐标和固定官方布局逐格重算。Office 的 3,532 个 API 漏列没有被标成 API 完整，采用明确官方中心公式重建并逐格原生验证；API 自由列表中的 label6 仍为未知。原生 padding 只影响工程导出世界裁剪，原 buffer/原 API 列表完整保留。导出图的每个像素由原生标签、height/near-ground CSR 和原 projection unknown 掩码重算，没有 unknown 升级。

**原源 RAM 与全部 2,740 个文件层 SHA/bytes 通过**：匿名 root 只包含元数据及唯一绝对原 USD 子层；session 只允许声明的原地面碰撞禁用、去实例、原 PhysicsScene 禁用，以及拥有的有限地面/查询 PhysicsScene。导出层使用当前 bytes 新匿名解析，query root RAM SHA 与导出精确相等，保留原 FPS24、defaultPrim、Z-up、米制。Office/Hospital 原源 `-1..0` 时间范围没有被写入修改。

实际 context/cache/root/session 身份一致；attach counter0→0，没有新 App/Context 或手动二次 attach。每场快照两次真实提交步后 counter2，全部查询前后 counter2、timeline0，独立读取初始与完整高度查询 clock JSON 互证。CPU 审计没有再次观测实时 Kit 内部时钟。

训练与导航 source174 的冻结归档逐成员重算；本次 CPU 审查前后当前 174 个文件也全部匹配。该观察是源码证据，不是完整运行 import ledger。生产探针为 `e47cd51d…`，最终报告 SHA `2559ba737be52561314074dc104750a2f30e2a854a257ee8f03af0c56f6999c6`。

门禁范围是固定零重力静态快照、显式有限 reference 地面、0.1m 单格 XY 和 0.15..1.8m 高度范围。**不构成全身净空、四场闭环行人、论文八场景或人感知导航质量通过**。历史 V5/V7/V8 失败与版本证据均未重写。
