# 四场景原生地图探针 v7 CPU 封存

当前状态：**91 项 CPU 回归、Ruff、四场景 standalone USD preflight 通过；v7 原生执行尚未进行**。不启动 Kit/GPU，不代表导航效果验收。

## 修复与边界

- 将查询时间线在 SDK reset 前后显式设为 0，保留原 USD authored time range；先持久化实际时钟，再严格核验非负计数/时间、查询不推进物理。Office/Hospital 原层 startTimeCode=-1 是失败原因的源码推断，v6 没有保存失败时刻的实际时间，不伪称已测量。
- source 已知、enabled、static 且几何验证不是地板的低障碍上表面，保留全部 ray hit CSR，单列分类。命中该表面的 known5 单元强制 occupied，即使上表面低于 0.15m；不是将它作为额外地面。真实 v6 raw 回归覆盖 Office 1289、Hospital 366、Full_Warehouse 12 行。
- 每格仍必须恰好一个声明有限地面的有效命中。原 Plane/平地 collider 必须禁用；foreign、动态、未验证几何或未声明近地支持面继续拒绝。
- 官方固定 buffer 布局、Office 缺失3532个 API known5 的显式工程中心、native lower padding、原 projection 世界裁剪、全量 floor/fullheight、raw6 不升级均保留。

## 来源

当前探针 SHA256：`a538bf92ffb64b536a7005329f38c080bd1b8818114df58d9880d5506f0665d5`。归档：`paper_scene_maps_clock_obstacles_source_v7.tar.gz`。全部来源、历史 v6 failed results/raw SHA、四场景 preflight 和测试日志见 [readiness JSON](paper_scene_maps_probe_readiness_v7.json)。v5 Warehouse 的独立120项审核和 v6 失败证据没有覆盖。旧 v6 readiness argv 的 base-uv 路径已在新 metadata 纠正，实际执行命令必须使用未 resolve 的 IsaacLab venv。

## 后续原生命令（由 root 在训练 GPU 空闲后执行）

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/scripts/probe_paper_scene_maps.py --projection-dir /project/EXP/SangforAndSaut/ProtoMotions/data/diffusion_amp/maps_resource_complete --output /project/EXP/SangforAndSaut/ProtoMotions/output/scene_maps/native_reference_four_scenes_clock_obstacles_v7 --scenes Office Warehouse Full_Warehouse Hospital --floor-owner reference_flat_terrain --known5-position-source official_buffer_centers --floor-probe-limit 0 --comparison-points 256 --snapshot-steps 2 --execute-physx
```

full0 要求所有 native known5 单元完成 0.15..1.8m 原生 overlap 与 exact floor ray；limited256 永远不准入。该产物是 native2D + explicit full-height 工程补充和 reference flat Terrain，不是作者原图、不说明孔洞/台阶适配或八场景人类导航质量。
