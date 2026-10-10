# 四场景资源接入后的新 CPU 地图

父进程已创建并验证 Forklift/KLT 两个外部相对链接。原四个场景重新递归检查后，composition 与 authored 纹理/MDL 路径闭包均完整，没有 missing、remote URI 或 invalid/LFS 资源。[完整机器清单与文件 SHA](maps_projection_resource_complete_inventory.json)；[地图总览](maps_projection_resource_complete_overview.png)。本 agent 未启动 Kit/GPU、未改原 USD/业务配置。

新输出目录：`data/diffusion_amp/maps_resource_complete/{Full_Warehouse,Warehouse,Office,Hospital}/`，包含实际 `occupancy.yaml/png`、`projection_preview.png`、完整 composition、provenance、visual dependency audit 与 stderr 日志。旧 `data/diffusion_amp/maps/` 和旧 `docs/maps_projection_inventory.json` 的记录 SHA 已逐一核实保留，没有覆盖历史快照。

## 当前地图

四图均 Z-up、metersPerUnit=1、地板 0m、分辨率 0.1m；bounds、负坐标 origin、尺寸和 YAML 门槛与原版相同。PNG row0=max world Y，column0=min X；网格中心与世界坐标往返均通过。

| 场景 | 图像尺寸 XY | origin XY m | occupied | free | unknown |
|---|---:|---|---:|---:|---:|
| Full_Warehouse | 372×551 | [-28.6, -24.0] | 45,569 | 152,956 | 6,447 |
| Warehouse | 251×370 | [-12.5, -18.5] | 27,314 | 59,521 | 6,035 |
| Office | 311×882 | [-24.6, -27.6] | 75,644 | 156,025 | 42,633 |
| Hospital | 762×423 | [-48.8, -4.9] | 50,294 | 141,415 | 130,617 |

两份 warehouse 原 free=0，现在实际资源补全后均有 free。Warehouse 投影几何从 371 增至 785（138 个 KLT 引用各三 geometry），新增 KLT 在本次高度段投影中未增加 occupied 格；它们已经实际加载。Full_Warehouse 的叉车使 occupied 增加 336 格。Office/Hospital 的 occupancy PNG、preview PNG、YAML 与旧版逐字节同 SHA，场景 root USD 的 SHA 四份都保持原样。

## 当前离线依赖检查

| 场景 | 本地 composition layers | authored asset records 扫描 | 缺纹理/MDL | remote / invalid |
|---|---:|---:|---:|---:|
| Full_Warehouse | 1,851 | 7,737 | 0 | 0 / 0 |
| Warehouse | 53 | 358 | 0 | 0 / 0 |
| Office | 543 | 2,350 | 0 | 0 / 0 |
| Hospital | 293 | 1,209 | 0 | 0 / 0 |

扫描包括 USD `asset`/`asset[]` 默认值与 time samples，实际 MDL 静态 imports/resources 和已安装 shader source closure；没有按材质名字造路径或近似模型。已有 scene textures 只查路径/非空/LFS，未全部解码，也未执行 MDL compiler/RTX。新接入两模型的全部六张 PNG 此前已真实解码并核对 SHA；Forklift normal 来源为原 Full_Warehouse 本地包，remote 下载错误证据仍完整保留，不能宣称其为官方 6.0 同一字节。

## 待执行的真实验收

这是 conservative USD world-AABB CPU 投影：地板上 0.15..1.8m 高度段、dominant floor support union，域外 unknown。旋转/非凸障碍、门洞、地板非矩形支撑、多层、动画，以及 Kit USD25.11/PhysX cooking 仍需单实例 Kit 验证。新图状态均为 `prototype_ready_for_physx_review`，没有 navigation-ready/PhysX 已通过的声明。

GPU 空闲后应加载实际 collider、timeline play/update，再用真实 `_omap` API 与 free/occupied world points核对。provenance 已给候选 free origin 和相对 min/max offsets，候选尚未证明 PhysX free。原场景配置没有自动重指到新图；四个现有场景及当前资源不能代替论文作者八场景清单/地图版本/seed/200s评估。

## 可重复命令与源码

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/generate_pedestrian_maps.py \
  --output data/diffusion_amp/maps_resource_complete --resolution 0.1 --timeout 60
```

本次没有使用 `--allow-incomplete`。各 stage 在限时 60 秒的独立 CPU 子进程完成。辅助只读视觉路径扫描源码：`docs/maps_projection_resource_complete_texture_audit.py`（Ruff PASS）；每场景同样有限时 60 秒的独立进程。完整坐标断言、旧输出 SHA、四场景 USD source SHA、Office/Hospital 图片/YAML 同 SHA 的验证结果位于 `data/diffusion_amp/maps_resource_complete/historical_comparison.json`。
