# 四场景 PhysX 地图探针：CPU 准备与边界

[独立脚本](../../scripts/probe_paper_scene_maps.py)、[完整证据](paper_scene_maps_probe_readiness_v1.json)。**33项CPU回归与Ruff通过，四场景真实源USD的SDK-free预检通过；尚未执行Kit或原生PhysX。**

## 地面来源

原Nav代码在提供Terrain时生成 `/World/ground`，见 `/project/EXP/NavIsaaclab2.0/protomotions/simulator/isaaclab/utils/scene.py:226`；CrowdSim patch保留并可移XY，见该仓库 `CrowdSim/world/builder.py:164`。因此视觉floor无CollisionAPI或disabled，不能直接推论原参考项目会跌穿。

| 场景 | 选定视觉floor | 原USD enabled floor collider | 结构floor候选 | 未解决顶面重叠对 |
|---|---:|---:|---:|---:|
| Office | 99 | 0 | 84 | 4 |
| Warehouse | 26 | 2 | 24 | 4 |
| Full_Warehouse | 69 | 15 | 54 | 2 |
| Hospital | 60 | 38 | 60 | 36 |

`--floor-owner none` 默认仅用原USD；floor支持必须实际查询，缺口原样保留。早期Office硬拒绝日志仍在 `paper_scene_maps_preflight_v1/`；只读清单在 [floor collision inventory](paper_scene_floor_collision_readonly_inventory_v1.json)。

`--floor-collision-overlay confirmed_visual_meshes` 是另一个显式工程替代方案，排除decal/carpet、检查真实faces与静态父层，并因上述部分顶面重叠而拒绝四场景schema admission。证据保留 [v4](paper_scene_maps_overlay_preflight_v4/manifest.json)，不继续修剪/布尔几何，也不伪造原作者物理配置。

`--floor-owner reference_flat_terrain` 是已完成的可排队方案：在session层生成一个**有限**静态双三角形ground，覆盖声明map XY bounds加0.2m margin，高度取已声明floor（当前四场景均0）；friction1/rest0/average、contactOffset0.002/restOffset0。禁用选定视觉floor的原enabled碰撞，避免双地板。Hospital的38个instance roots只在session层deinstance后编辑原floor属性；原USD及依赖层内容SHA均不变。它遵循参考项目生成Terrain的地面归属，但不是原USD自带地面、私有论文scene配置或任何孔洞/台阶复原。原投影205未知区域不会因有限平面而变为free。

完整四场景真实CPU recipe/schema/hash结果：[reference-ground v6](paper_scene_maps_reference_ground_preflight_v6/manifest.json)。Office/Warehouse/Full_Warehouse/Hospital分别543/53/1851/293依赖层重哈希；无enabled非kinematic rigid body或时间采样xform。独立源文件冻结SHA见JSON。

## 原生执行合同

实际签名以安装的 `_omap.pyi`、`MapGenerator.h`及官方test为准：`update_settings(.1,4,5,6)`、`set_transform(origin,relative_min,relative_max)`，Generator绑定实际PhysX和stage_id。omap依赖启用的碰撞几何及不占据的起点。[NVIDIA官方文档](https://docs.isaacsim.omniverse.nvidia.com/latest/digital_twin/ext_isaacsim_asset_generator_occupancy_map.html)。

单个App逐场景加载原USD，session配置CPU PhysX、零重力、120Hz、显式scene query。原authored PhysicsScene仅session inactive，后续runtime新scene路径唯一。原生`raycast_all`读取hit.position/normal/collision/rigid_body并只接受精确floor whitelist的向上近floor命中；`overlap_box`使用XYZW identity。不采用closest ceiling、不生成投影fallback、不保存原scene。

保存原生buffer、occupied/free世界位置、实际bounds/dimensions后再校验；只按返回世界中心重建ROS north-up，拒绝labels/count/duplicate/conflicting cell/roundtrip问题，不假定native reshape方向。当前full-ROI门禁保守要求actual bounds/dimensions等于声明请求；官方允许实际范围缩小，出现这种情况会保留raw并拒绝本次full-ROI资格，不误称API必定完整覆盖。seed7固定world点保存所有CPU projection/native label与band overlap差异。

每个native free cell的floor rays可limited或full。`--floor-probe-limit 0`才是全量，有限抽样永不通过scene-resource gate。全部支持、起点自由且有地板、时钟未在query期间推进、静态/源SHA/映射/抽样free-overlap检查通过，才标 `scene_resource_gate_passed`。**navigation_quality_validated和navigation_ready仍为false**；没有机器人/行人clearance、路径或论文八场景效果验收。

## 排队命令（仅主代理在GPU空闲后执行）

默认preflight不启动Kit：

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/probe_paper_scene_maps.py \
  --floor-owner reference_flat_terrain --output /tmp/scene-map-preflight-new
```

先limited pilot，输出目录必须全新：

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/probe_paper_scene_maps.py \
  --execute-physx --floor-owner reference_flat_terrain --floor-probe-limit 256 \
  --output output/scene_maps/native_reference_pilot_v1
```

完整验证另一个全新目录、改 `--floor-probe-limit 0`。limited模式预期scene-resource admission为false；以result/manifest JSON为权威，不能仅凭SDK close后的process exit0认为成功。原 `maps_resource_complete` 不覆盖。
