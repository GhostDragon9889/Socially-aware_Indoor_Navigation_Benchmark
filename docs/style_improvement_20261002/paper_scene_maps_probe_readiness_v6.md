# 原生地图修正 v6：CPU 准入

**82 项 CPU 回归、Ruff、四个场景 standalone USD 预检和四份真实旧 native raw 布局核验通过；v6 原生执行待 root 排队。** [机器记录及完整 argv](paper_scene_maps_probe_readiness_v6.json)、[实物 CPU proof](paper_scene_maps_mapping_v6_actual_raw_CPU_proof.json)。

固定 installed `compute_coordinates` 的方向继续使用，不搜索最佳轴排列。全部 occupied API 点须完整覆盖 raw4；每个 free API 点须唯一、中心坐标往返成立且标签为 5 或 6，unknown6 永不提升为 free。

Office 的 API 缺少 raw5 中 3,532 点（native 列 137／295／300／305，各 883 点）。默认 `api_verified` 仍拒绝；显式 `official_buffer_centers` 按固定 min／dim／0.1m 公式生成缺失中心，记录 API row=-1、完整缺失 indices／world positions，保持 `known5_API_complete:false`。这些几何中心不代表原生支持证明；全部 raw5（包括缺 API 点）必须再逐格执行完整 0.15–1.8m native overlap 和 native floor ray 才可准入。

Office native 下界 Y 多一格；Hospital／Full_Warehouse 下界 X/Y 各多一格。只接受已验证的整数 0／1 格下界 padding，上界和分辨率必须一致。先完整查询 native grid，保存原生数组／完整 native PNG，再按世界坐标裁到原 projection bounds。所有导出格中心已 CPU 全量校验，未修改作者投影文件，未把未知格补成自由格。

v5 单地面门禁、all-hit CSR、foreign／coplanar 支持拒绝和查询时钟不前进门禁保留。v4 Warehouse 错误 PASS 保留并明确被独立审计拒绝；v5 Warehouse 120 项独立实际检查保留。v6 source 改变，四场景须新输出重跑，不继承 v5 成功标记。有限采样永不称完整地图准入。

## root 串行原生入口

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/probe_paper_scene_maps.py \
  --output output/scene_maps/native_reference_four_scenes_mapping_v6 \
  --scenes Office Warehouse Full_Warehouse Hospital \
  --floor-owner reference_flat_terrain \
  --known5-position-source official_buffer_centers \
  --floor-probe-limit 0 --comparison-points 256 --snapshot-steps 2 --execute-physx
```

输出必须新建。`scene_resource_gate_passed` 仅说明静态场景资源检测；有限 reference ground 是工程参考地面，完整高度查询是原生 2D API 的显式补充，不代表作者私有配置、人体 clearance、八场景论文复现或实际导航质量验收。
