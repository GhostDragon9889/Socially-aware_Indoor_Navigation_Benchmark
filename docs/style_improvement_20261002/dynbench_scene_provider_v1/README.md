# DynBench 独立 SceneProvider：已准备，尚未接入

新增 [dynbench_scene.py](../../../CrowdSim/humanoid/runtime/dynbench_scene.py) 和 [专项测试](../../../tests/test_dynbench_scene_provider.py)。没有修改 source135 中的 AMP world / runner / settings / trainer / benchmark，也没有修改已冻结的五份薄入口文件。没有复制控制循环或使用 MaskedMimic fallback。

本轮仅 CPU。当前没有通过的 native full-grid map：四份 `maps_resource_complete` 是 CPU 投影；SDK-free v6 预检也不能当作地图的原生验证。v6 记录的 probe script 字节与当前脚本不同，新 Provider 正确拒绝加载该旧来源，不改写它的 SHA。root 运行的 Warehouse 原生 pilot 还因 buffer/world-position 数量不一致失败，不能据此准入地图。

## 明确接口

```python
from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider

provider = DynBenchSceneProvider.from_preflight(
    selected_current_preflight, expected_sha256=selected_preflight_sha,
)
# 拟接入的 simulator factory 必须在创建 SimulationContext 之前调用：
sim_cfg = provider.prepare_simulation_cfg(sim_cfg)
# 一次 Kit App 已启动后，新的 factory/_get_scene_cfg hook 可调用：
scene_cfg = provider.prepare_scene_cfg(base_scene_cfg)
# 等真实 native full-grid probe 完成后，明确绑定其来源：
provider.admit_native_map(selected_native_result, expected_sha256=native_sha)
provider.bind_native_query(actual_scene_query, physics_step_count=actual_counter)
```

当前冻结的 `IsaacLabSimulator.__init__` 先创建 SimulationContext，后调用 `_get_scene_cfg`；冻结 AMPWorld 没有上面的 pre-create hook。**仅在 `_get_scene_cfg` 注入 USD 无法保证启用 PhysX scene query。** 因此本轮不擅自接线；root 后续须单独新增 factory/subclass/driver，并实际验证一次 App、一次 PhysicsScene 与同一 native view，不能绕过此前 source freeze。

| 接口 | 具体行为 | 尚未证明的部分 |
| --- | --- | --- |
| `from_preflight` | 完整 hash 绑定 JSON、USD 全部 layer、投影文件、floor recipe/overlay 与 probe script；检查静态、Z-up、米单位、原始 collider exact paths | 不执行 Kit cooking，不准入 CPU 投影 |
| `prepare_simulation_cfg` | 返回配置副本，明确 `enable_scene_query_support=True` | 当前 AMPWorld 未调用；不能在 context 创建后补救 |
| `prepare_scene_cfg` | lazy SDK import；复制 base cfg，移除生成 Terrain mesh，保留 robot；contact sensor 只将旧 terrain filter 换成 exact reference floor；禁用复制 physics，新增一个全局 source spawner | 尚未在真实 InteractiveScene 执行 |
| lazy spawner | 原始 USD 作为 session sublayer，保留 `/Root/...` 世界坐标；源 PhysicsScene 在 session 停用；唯一有限 reference plane；原 floor colliders 全部禁用；不保存原 USD | 新 provider 的 native floor cooking / root 冲突 / materials 仍需实测 |
| `admit_native_map` | 只接受 real probe schema、full-grid 支撑、同 source/preflight/script、static snapshot、zero foreign/occupied-free overlap、query 时钟不推进；验证实际 PNG/YAML/支持 rays 字节与 exact floor，unknown 不变成 free | gate 是资源快照，不是导航路线、footprint 或八场景质量通过 |
| `ground_height_at` | actual native raycast，只接受 exact floor，要求 floor support、向上法向和高度范围 | 无 native API timestamp，只有调用前后已提交 step counter 不变 |
| `height_at` | 同一 raycast 集合，取 exact floor/static 水平表面的最高值；显式排除 robot/未知 collider | 不用视觉 bbox；不将墙/机器人当作地面；射线 band 上限 1.8 m 是已声明工程假设 |
| `raster` | `[B,5]` 世界状态 → TRACE `[B,2,224,224]`；native 实际原点、ROS row flip、negative XY 保留，205及地图外均 `.5/.5` | 需要先 native map admission；不把 unknown 默认写成 walkable |

## 有限地面与查询语义

唯一 floor path 为 `/__PaperMapProbeReferenceGround`，平面由原始 map bounds 加 0.2 m margin（允许明确 0.1–1 m）生成，两个 +Z winding 的 triangle，static mesh `approximation=none`，contact offset 0.002 m、rest offset 0，摩擦 1/1、restitution 0、combine average。geometry/schema/material/deinstance 逻辑来自已审阅的 `scripts/probe_paper_scene_maps.py`，没有 import 该脚本或启动 App。此 plane 是 **reference_flat_terrain 工程地面**，不是“原始 USD 自带 floor collider”，也不推断 holes/stairs/其他楼层。

exact floor 与 static collider 两集合必须互斥；原 floor colliders 完整禁用，不将所有 floor-like 视觉 mesh 一律赋予碰撞。height/ground 用 actual query 分流，坐标越出有限 plane 明确失败；不把负 XY clamp 至 0。Torch 输入返回相同 device/dtype 的 height tensor，NumPy 输入返回 NumPy。当前逐点 CPU query 未优化，32×32×多 actor 的性能仍需真实测量，不承诺部署速度。

## 验证范围

31 项 CPU tests：cold import 无 Torch/NumPy/pxr/SDK、真实 finite geometry、floor/static 互斥、source bytes 改变拒绝、pre-create hook copy、lazy cfg 注入、default no map admission、partial native/clock/floor/unknown promotion 坏记录拒绝、raster unknown/outside、exact floor vs wall vs robot 分流、查询推进计数拒绝、四份本地历史 preflight 实物 SHA 门禁。31 PASS、Ruff PASS。native test JSON 和 native query 对象是显式 synthetic contract fixture；没有生成或冒称真实 PhysX map。

后续真实门禁：新的 source-matched preflight → bounded native pilot → full-grid static native map → 新 provider 单 App/scene/floor/query integration probe → AMP inference + Diffusion 同时挂接 → 对实际地面支撑/静态/peer/COM/导航计数核查。任一阶段失败均保留原始产物；不得以 blank map、CPU projection、visual bbox 或另一 floor owner 降级。
