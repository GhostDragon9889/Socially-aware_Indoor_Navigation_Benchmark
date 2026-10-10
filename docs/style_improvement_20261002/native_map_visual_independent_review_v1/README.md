# Render 与 V9 V2 独立 CPU 审查

本目录保留合并前的只读审查证据。审查没有启动 Kit、GPU，没有修改生产或候选源码。随后合法合并造成当前树变化，不会改变这些历史证据。

## Render 候选

冻结候选 `2a5a34f6…` 的 **31 项 CPU 检查通过**：实际 Warehouse 53 个文件层 SHA 不变，1,442 个非 Render Xformable 在 default/t0 共 2,884 次矩阵比较完全相同，SDK 原有 42 个属性不变。重建使用实际 SDK 根层文本，不构成实时 Kit 渲染证明。

只有源 `/Render` 的空 metadata over 与匿名 SDK Render 命名空间冲突可准入；源以最弱根子层组合，session 保持较强。原物理场景禁用、其他根路径冲突、碰撞禁用及有限可视地面门禁保留。`prepare_native_visual_scene`、`verify_native_visual_sources` 与旧实现 AST 相同，composition 的声明改动以外 AST 相同。

详见 [visual_review.json](visual_review.json) 和 [verify_visual_candidate.py](verify_visual_candidate.py)。后续实际回放可见性必须另外验收。

## V9 缓存缺陷与修复

V1 使用 `Sdf.Layer.FindOrOpen` 读取导出的 runtime root/session。独立反例先缓存真实 Warehouse session，再同路径写入有效 USD 位移属性并重新登记 SHA/bytes；旧门禁仍接受缓存对象。这项失败完整保留于 [v9_owned_export_cache_review.json](v9_owned_export_cache_review.json)。`v9_verifier_reviewed_initial.py` 是审查时的中间源码快照，不表示已经验证更早的 header-only 版本。

V2 `d21020e1…` 从当前 SHA/size 绑定的 bytes 创建新匿名层并 `ImportFromString`，使门禁检查实际导出内容。独立 **12 项行为检查通过**，**5 个反例均拒绝**：同路径缓存重写、根层新增空间 prim、导出与 query RAM 不一致、伪造原源 RAM 声明、无效导出 bytes。真实 Warehouse 候选的 56,239 格完整准入通过；这是读取已有实际产物的 CPU 核验，没有新增原生执行。

详见 [v9_v2_review.json](v9_v2_review.json) 和 [verify_v9_v2.py](verify_v9_v2.py)。原源文件层的 `FindOrOpen` 有意用于原源 RAM 证据，导出物使用新匿名对象。

## 版本边界与发行身份

旧 V5/V7/V8 provider/grid dispatch 的 AST 不变。原 buffer、坐标、CSR、crop、floor/full-height-band 和 query-clock 数学 AST 不变，V9 增加原源与匿名 root/session 的所有权门禁。独立 **88 项**有界八路径、before/after SHA、归档成员、保护路径和作者 JUnit 身份检查通过；作者最终结果为 270 tests、0 failure/error/skip，独立审查未重复运行该套测试。

最终 patch：`f10a0d49a3deaf7bace48fa97f6ff06a3e9a6c5d837c385faa6231e4f011849a`。

最终 manifest：`62cee030f798e9644ed7237f39611f8c17b0f6771cee657bb7e91cda8bab359d`。

作者归档：`6ab5a79ba1ebe9524e11f422ee164a885af4c85830a08823b53eada5d1f47378`。

详见 [v9_raw_math_AST_review.json](v9_raw_math_AST_review.json) 和 [v9_v2_final_artifact_review.json](v9_v2_final_artifact_review.json)。本报告不能替代另外三个场景的新原生执行，也不表示论文八场景、行人风格或导航质量通过。
