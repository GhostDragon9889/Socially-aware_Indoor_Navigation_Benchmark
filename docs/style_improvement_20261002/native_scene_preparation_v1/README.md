# 可重复的原生行人场景配置生成

新入口：[prepare_native_pedestrian_scene.py](../../../scripts/prepare_native_pedestrian_scene.py)。它只运行 CPU 地图准入、选点和 A*；不启动 App/SDK，不创建世界，不运行模型，也不修改地图、原始 USD 或已有配置。

## 实际 v5 Warehouse 复现

最终记录在 [minimum_separation_gate_v2/actual_Warehouse_v5_reproduction.json](minimum_separation_gate_v2/actual_Warehouse_v5_reproduction.json)。早期未设显式最小出生间距的生成记录保留在本目录，不作为最终门禁。

|人数|路点偏移|初始与闭环 A* 段数|实际最小根位置间距|与原配置比较|
|---:|---|---:|---:|---|
|4|1,2,3,4|20|10.0955435713 m|出生点、路线的 dtype/shape/bytes 相同；解析后的完整配置相同|
|15|1,4,8,12|75|4.4598206242 m|出生点、路线的 dtype/shape/bytes 相同；解析后的完整配置相同|
|3|1,2,3|12|13.3154046127 m|新增本地三顶点协议；没有原配置相同的声明|

原来的两个 `configs/diffusion_amp/native_warehouse*_clearance08_v1.yaml` 没有改写。新文件键顺序可以不同，报告没有声称文件哈希相同。

## 命令

在仓库根目录运行，输出必须是不存在的新路径：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python scripts/prepare_native_pedestrian_scene.py \
  --preflight output/scene_maps/native_reference_warehouse_floor_owner_v5/Warehouse/preflight.json \
  --native-map output/scene_maps/native_reference_warehouse_floor_owner_v5/Warehouse/result.json \
  --probe-script-snapshot docs/style_improvement_20261002/native_runtime_epoch_v5/probe_paper_scene_maps.py \
  --actors 4 --route-shifts 1 2 3 4 \
  --clearance-m 0.8 --spawn-clearance-m 3.0 --minimum-spawn-separation-m 1.6 \
  --output-config output/prepared_scenes/warehouse4.yaml \
  --proof output/prepared_scenes/warehouse4_CPU_proof.json
```

15 人使用 `--actors 15 --route-shifts 1 4 8 12`。默认偏移为：N=3 使用 `[1,2,3]`；N=4..15 使用 `[1,N//4+1,N//2+1,3*N//4+1]`。自定义偏移允许 2..N 项，模 N 后各不相同，只有最后一项允许指向自己的出生点。N<3 明确拒绝，不改变既有手工配置的 1..15 人合同。

未来四场景 v7 结果真正准入后，可把 `--preflight/--native-map` 换成相应场景的实际文件，并指定已保存的 exact v7 probe 源码。没有实际结果或门禁失败的 v6 文件不能生成可用场景。

## 选点与验证合同

1. 重新验证源 USD/layers、probe 身份、原始 native buffers、单地面、全高度查询和未知标签；配置只绑定实际通过的资源。
2. 阻挡包括 occupied、unknown 和地图域外。对 free cells 的距离变换减去两个半格对角线，使用与 `NativeGridScene` 相同的全格保守净空。
3. 选取 4 邻接最大连通分量；并列时选 ROS 行优先的第一个分量。出生点须具有至少 3 m 保守净空。首先选净空最大的第一个格子，之后按到已选点的最小距离取第一个 maximin argmax。不使用 RNG。
4. 默认实际根位置两两间距至少 1.6 m；最后距离不足则失败，不换分量、缩小阈值或重采样。证明同时记录要求和实测值。这个几何约束不等于全身碰撞验证。
5. 对每个人验证出生点到第一路点、各路点之间和末点回第一点的全部 A* 段。没有路径或退化段直接失败。
6. 输出只有既有 `crowdsim.native_pedestrian_scene.v1` 冷配置字段；单独 CPU proof 保存 SHA、选点、间距、路径长度/哈希和源信息。两个输出均不覆盖已有文件。

最终门禁：32 项 CPU 测试通过，Ruff 通过，见 [CPU_gate.json](CPU_gate.json)。测试覆盖未知/域外阻挡、断路、候选点不足、很多像素却出生点太近、非退化偏移和禁止覆盖。生成的是本地场景协议，不能称论文八场景协议或物理导航通过。
