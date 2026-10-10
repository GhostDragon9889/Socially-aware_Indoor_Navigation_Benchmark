# 当前场景和占据地图清单

本次只读核对当前workspace、相邻Assets与原NavIsaaclab2.0参考路径；没有启动SDK，没有修改共享场景。逐项路径、尺寸、SHA256、metadata与图像尺寸见 [JSON](paper_scene_resource_inventory.json)。

## 现有场景配置

| 配置 | USD | 对应占据地图 |
|---|---|---|
| bidirectional_open.yaml | 存在、非LFS指针 | 存在 |
| dynabench_full_warehouse.yaml | 存在、非LFS指针 | 缺失 |
| dynabench_hospital.yaml | 存在、非LFS指针 | 缺失 |
| dynabench_office.yaml | 存在、非LFS指针 | 缺失 |
| dynabench_warehouse.yaml | 存在、非LFS指针 | 缺失 |
| pedestrian_open.yaml | 内建空地 | 存在 |
| warehouse.yaml | 存在、非LFS指针 | 存在 |

仓库七份配置不代表七个独立物理场景：warehouse与bidirectional_open复用同一个多货架USD和newmap，pedestrian_open为本轮新增空地。DynBench另有Jetracer根USD；六个本地根USD均存在且不是LFS指针，但不能据此认定作者的八场景评估协议已经补齐。这里只验证根文件，没有重新验证完整递归组合与每个碰撞体。

## 仍缺的四份地图

- `/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Full_Warehouse/map/occupancy.yaml`
- `/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Hospital/map/occupancy.yaml`
- `/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Office/map/occupancy.yaml`
- `/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Warehouse/map/occupancy.yaml`

当前Assets中没有发现其他occupancy metadata；原参考配置使用的`/data/DynBench`根目录也不存在。现有`newmap.yaml`引用World1.png，395×603、0.05m像素、origin=[-10.325000381469726,-12.174999809265136,0]；两种warehouse行为共用。World0.png存在但没有独立YAML注册。新增pedestrian_open为301×301、0.05m像素的空地，不提供DynBench四场景的障碍几何。原参考的两张World图与当前文件SHA一致。

两份官方HDR现已通过安装manifest中的原USD引用路径解析；这修复了天空纹理依赖，不能代替上述地图或作者八场景清单。

## 当前真实训练的覆盖范围

AMPWorld训练使用无障碍平地、隔离物理环境、空邻居mask、全free raster。Shared evaluation可加入其他当前行人的状态历史，但训练语料中没有作者人群交互/障碍轨迹，不能声称避障与人群交互已经学到或复现TableII。完成作者八场景、每场3–15人200s的精确实验仍需要场景ID/配置/种子/聚合定义；可另行建立有来源的重建场景协议，并明确与作者协议的区别。
