# 四个DynBench场景：离线占据地图原型

已按occupancy-map技能生成真实CPU USD几何投影，未启动Kit/GPU，未改外部Assets、SDK或共享场景配置。[完整清单与SHA](maps_projection_inventory.json)；脚本：`scripts/generate_pedestrian_maps.py`。

## 结果

四场景Z-up、metersPerUnit=1；选择dominant地板0m与其世界坐标support bbox，分辨率0.1m，底左origin为实际负坐标，PNG row0为最大Y。

| 场景 | 图像尺寸 | origin XY m | 障碍格 | free格 | unknown格 | 状态 |
|---|---:|---|---:|---:|---:|---|
| Full_Warehouse | 372×551 | [-28.6, -24.0] | 45233 | 0 | 159739 | 缺模型，禁止导航free |
| Warehouse | 251×370 | [-12.5, -18.5] | 27314 | 0 | 65556 | 缺模型，禁止导航free |
| Office | 311×882 | [-24.6, -27.6] | 75644 | 156025 | 42633 | 局部几何完整、待PhysX |
| Hospital | 762×423 | [-48.800000000000004, -4.9] | 50294 | 141415 | 130617 | 局部几何完整、待PhysX |

每个`data/diffusion_amp/maps/<scene>/`目录包含`occupancy.yaml/png`、`projection_preview.png`、完整composition列表、world bounds/collider schema/输出hash的`provenance.json`和stderr日志。它们是本轮重建资源，并非论文作者地图，也未自动接入原4份配置。

Office完整worldBBox包含约1063×677m的装饰城市，不能默默把这些范围全视为可行走free。本脚本保存完整世界BBox，同时默认使用真实地板support union的bbox；域外灰色unknown。可显式`--extent-mode world`，但有max-cells门槛。障碍用地板上0.15..1.8m高度段内几何的世界AABB投影；原地板/上方屋顶不投成障碍，地板边缘用cell center支持判定，避免将padding误写为free。

## 实际缺失的几何

- Warehouse：`../Assets/DynBench/Props/KLT_Bin/small_KLT_visual_collision.usd`，138个 authored reference sites。
- Full_Warehouse：`../Assets/DynBench/Props/Forklift/forklift.usd`，`/Root/forklift`一个引用。

两项缺失geometry没有可信footprint，故这两张occupancy将所有原free候选标为unknown，只保留已知障碍；预览图单独展示几何free候选，但不能供导航当作有效free。没有用替代盒子、虚构模型或同一空图填坑。递归composition未发现remote arcs；遇到remote/invalid layer将直接阻断，stage生成有独立subprocess timeout。

## 离线验证与限制

6项实际USD CPU回归通过（0.30s），Ruff PASS；覆盖厘米→米、负世界坐标、Y-up右手转换、高度过滤、世界/网格中心往返与Y翻转、missing变unknown、PNG/YAML/provenance和源文件只读。Office小样已人工查看房间结构，Hospital全图已查看。

这不是PhysX占据地图：USD26.8的AABB投影不能证明Kit USD25.11/PhysX cooking正确，旋转障碍、非矩形地板、门洞、多层和动画需另复核。作者八场景/seed/200s协议没有因此补齐。

## GPU空闲后的正确omap调用

已查本地Sim6真实`_omap.pyi`与头文件/官方tests。`Generator(physx,stage_id)`；`update_settings(cell_size,occupied_value,unoccupied_value,unknown_value)`；`set_transform(free_origin,min_offset,max_offset)`，bounds是相对于origin的offset。skill示例的z_min/z_max/occupancy_threshold并不是此版本settings参数，不能照抄。

实际测试明确：不启动timeline会返回空buffer。后续在新的单实例Kit进程加载场景、PhysicsScene、play且至少update一次后生成，核对`get_dimensions()`的width/height/depth与buffer非空，并与已知occupied/free world positions验证行列/Yflip及地图边界。每个完整原型provenance已提供最大projection clearance的候选free-origin和relative offsets，但尚未验证该点的PhysX碰撞状态。实际UI另委托render-based generator，不能用UI产出证明core PhysX `_omap`。
