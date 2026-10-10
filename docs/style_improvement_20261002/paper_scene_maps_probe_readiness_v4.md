# 原生地图探针：完整高度补充 v4

59项CPU测试与Ruff通过；源码与测试独立归档：[v4机器记录](paper_scene_maps_probe_readiness_v4.json)。未启动Kit/GPU；原生Warehouse v1失败、raw2D、API列表和v3历史身份全部保留。没有改训练135／导航138来源。

## 新合同

二维API返回z=.7..8，不能冒称.15..1.8整体净空。因此增加显式工程refinement：每个canonical buffer5世界点查询0.1m×0.1m、地板上方.15..1.8m的native overlap_box。半尺寸(.05,.05,.825)、中心z=ground+.975，XYZW单位旋转；只有exact authored enabled collision paths可分类，selected floor exactpaths排除，foreign路径保存并拒绝分类／资源准入。

`--floor-probe-limit 0`对**所有**canonical known5 cell同时进行floor ray及完整高度查询。全部CSR hit offsets／collision与rigid-body路径／分类、mask／原生free行身份保存到独立NPZ。submitted-physics counter和timeline before/after单独保存；任何推进即拒绝。256点limited模式只能提供有限probe，不能全图准入，未查询known5降unknown。

保留raw native2D png／buffer及API列表。5中检测到已知障碍的cell降occupied；最终knownfree仅full-height-clear且floor-supported；未测／无floor／foreign未分类均unknown；原6永不升级free。生成height_band_refined.png和final occupancy.png；reference-ground mode继续保留原投影205。native全部floor支持、完整height、static/source/clock以及origin位于refinedfree共同门禁，floorgrid单独不能通过。

该产物是“native2D＋explicitfullheight”的工程地图，不是纯omap2D完整高度API行为、作者八场景地图或行人footprint／全身净空验收；navigation_ready始终false。CPU59项含真实raw未知保存、错误范围、known5高障碍降级、floor排除、foreign分类拒绝和微小clock推进拒绝；fake查询与synthetic分类明确不当原生实测。首次新增metadata dict闭合错误导致collection失败，已修正且v1日志保留，v2为59项最终通过证据。

## root 排队原生完整执行

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
../IsaacLab-Protomotions/.venv/bin/python scripts/probe_paper_scene_maps.py \
--scenes Warehouse --floor-owner reference_flat_terrain --floor-probe-limit 0 \
--execute-physx --output /tmp/native_warehouse_fullheight_fresh
```

实际执行重新preflight并绑定本v4字节。仅CPU物理scene的SDK进程也由root串行启动，本agent未执行。SDK app.close可能导致进程exit0，因此每scene result状态及fullgate为准。
