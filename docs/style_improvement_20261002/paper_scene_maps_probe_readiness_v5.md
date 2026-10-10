# 原生地图探针 v5：地板所有权门禁

**65项 CPU 回归与 Ruff 通过；Warehouse SDK-free预检通过。尚未执行 v5原生进程。** [完整冻结记录](paper_scene_maps_probe_readiness_v5.json)包含源码／测试／独立archive SHA。脚本 SHA：`16f845af4d315cd9328d7e1ebdf68317b64192947b5f49df07b3942b4f2f9c0d`。

## v4实际缺陷与保留证据

Warehouse v4已真实取得全部56,239个known5的floor和全高度查询（520个高障碍命中），但origin全命中记录还包含未声明共面 `/Root/GroundPlane/CollisionPlane`，z≈8.94e−8m。原native结果的 `scene_resource_gate_passed=true` 漏掉所有权检查，**不能用于single-floor资源准入**。[独立退回报告](native_warehouse_fullheight_v4_floor_ownership_independent_audit_v1.json)保留原JSON/hash；原v1/v4失败／原始查询／readiness均未覆盖。v4只有全量hit数量、未存每条完整路径，不能把origin的extra路径推断为每条射线命中。

## 修正范围

全enabled collider类型库存记录真实类型。Plane依据实际变换后的axis／world normal／高度识别近地面，不能使用缩小的visual extent或名称猜测；Mesh采用真实face证据。显式reference-ground mode只在session禁用原启用近地面的静态Plane／Mesh和旧selected-floor collider，保持屋顶／墙体与原USD文件不变；animated/dynamic平面拒绝替换。Warehouse实际预检识别并禁用原Plane及11个decals，共12条；源composition/hash未变。

每个floor ray保存全部hit CSR：offsets、collision／rigid-body exactpath、世界position／normal、distance、classification及rowmask。若未声明路径存在upward normal≥.9且离ground≤.08m的支撑面，或任意foreign ray路径，即使选定reference ground也命中，仍拒绝floor ownership。高处屋顶／台面单独标为known_nonfloor_surface，不当作地板支持。实际enabled flat-ground库存必须等于declared floor paths；reference mode要求唯一owner。失败数据先保存，门禁不能补零／丢弃extra hits。

v4已冻结的native布局、raw4/5/6及unknown保留、全高度0.15..1.8m和全量floor/refinement门禁继续存在。新 `floor_ownership` 合同以及全部 `floor_support_rays.npz` CSR必须有效。limited采样仍不资源准入；地图仍不是作者八场景／人体footprint／导航质量验证。

## root 原生执行（fresh目录）

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
../IsaacLab-Protomotions/.venv/bin/python scripts/probe_paper_scene_maps.py \
--scenes Warehouse --floor-owner reference_flat_terrain --floor-probe-limit 0 \
--execute-physx --output output/scene_maps/native_reference_warehouse_floor_owner_v5
```

SDK-free准备路径：`/tmp/native_warehouse_floor_owner_v5_final_preflight/Warehouse/preflight.json`，其script SHA与本冻结一致。原生执行重新生成preflight，不复用旧v4许可。SDK app.close可能使失败进程exit0，必须审查scene result以及全部raw合同。本agent只执行CPU测试和USD预检，无Kit／GPU；训练135／导航138字节未修改。
