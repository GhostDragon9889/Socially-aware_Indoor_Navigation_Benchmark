# 四场景实际 V9 资源审计

实际输出：`output/scene_maps/native_reference_four_scenes_anonymous_root_v9`。固定 producer SHA 为 `e47cd51d9f40cd0ead27595283fefab95871ea229c7029c5622b57391a01ad82`。本报告只运行 CPU 读者，没有启动 SDK、GPU 或修改生产源码。

四场景都通过完整消费端准入：原始 native4/5/6、官方固定布局、API与工程补列、全量地面/全高度 CSR、低障碍保守降占据、世界裁剪、unknown保留、导出图、源文件/层 SHA、原始源 RAM 和匿名运行根/会话所有权均重算通过。数组的 dtype、shape 和 C-order 字节 SHA 均保存在 `report.json`。

| 场景 | 全量 known5 格 | 全高度障碍命中格 | 低上表面障碍格 | API 缺失 known5 |
|---|---:|---:|---:|---:|
| Office | 223,776 | 6,094 | 1,289 | 3,532 |
| Warehouse | 56,239 | 520 | 0 | 0 |
| Full_Warehouse | 155,300 | 890 | 12 | 0 |
| Hospital | 112,006 | 1,931 | 366 | 0 |

共 547,321 个 known5 格全部做了真实原生查询。Office 的 3,532 格明确来自官方 raw5 buffer centers，API row=-1；没有宣称 API 完整，也没有将 unknown6 变成 free。障碍两列可能重叠，不把两列相加作为唯一占据格数。

四场景实际查询均为 submitted counter=2、timeline=0，并保存查询前后未推进证据。计数器是提交步计数；没有伪造 native engine timestamp。匿名 root 的元数据变化允许留在自有层，原始 file-backed source RAM 与文件层保持不变。

`source174_binding.json` 将全部 174 个选择源码的当前字节和归档成员逐项绑定到 epoch V7 manifest `f9c007ec503236790766e768967a8b745b0dcde1a4a96dfd0b58986763ce403f`。这是选定源码集，不声称完整动态导入账本。旧 V7/V8 的失败产物未覆盖。

## 保守出生与路线配置

`../configurations4_v1/` 中四场景 N4 都已生成，原生地图绑定且所有初始/循环 A* 路线通过。

`../configurations3_to15_v1/report.json` 记录全部 52 次请求：48 份配置通过，4 次明确拒绝。48 份成功配置共检查 2,058 段初始/循环 A* 路线。约束固定为路线 clearance .8 m、出生点 clearance ≥3 m、root separation ≥1.6 m；unknown/outside 都阻塞，确定性 maximin 无随机兜底。

| 场景 | 成功人数配置 | 最大成功人数时最小 root separation |
|---|---|---:|
| Office | 3–15 | 1.780449 m |
| Warehouse | 3–15 | 4.459821 m |
| Full_Warehouse | 3–15 | 6.303174 m |
| Hospital | 3–11 | 1.612452 m |

Hospital 的 N12/13/14/15 maximin 最小间距分别为 1.562050/1.500000/1.252996/1.118034 m，小于 1.6 m，未发布配置、未降低约束、未重采样。这仅说明该固定贪心配置算法失败，不证明其他排列数学上不可能。

这些配置是 CPU 可达性证据，不是人形身体碰撞证明或闭环行走成功。四个本地场景也不能冒充作者八场景协议。
