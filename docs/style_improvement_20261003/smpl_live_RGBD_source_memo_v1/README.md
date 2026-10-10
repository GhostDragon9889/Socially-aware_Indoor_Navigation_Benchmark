# SMPL 实时外观与 RGB-D：最小接线依据

2026-10-04。此轮只读源码和官方资料，并新增一个可选捕获适配及纯 CPU 接口测试；没有运行 Kit、模型、CUDA 或渲染，也没有修改原 185 文件。**真实 RGB-D 同步、标定与物理不推进仍待实际捕获核验。**

## 已有外观，不需要另建骨架渲染器

论文 §III-B 将 SMPL 骨架与连续蒙皮对应，用于 RGB-D；没有公布可核对的作者纹理/相机身份。[NavIsaacLab §III-B](https://arxiv.org/html/2606.26265v1#S3.SS2)。本地已有 neutral SMPL、6,890 顶点、13,776 面、faceVarying UV 和两张 albedo。历史完整 900 controls × 4 人蒙皮核验的关节位置最大差为 **0.23931 mm**；这证明历史姿势回放接线，不能替代新实时同步验证。[历史实际审计](../../style_improvement_20261002/recorded_smpl_skin_audit_v1/README.md)。

现有 [train_paper_amp.py](../../../scripts/train_paper_amp.py:716) 已在 `--capture` 分支创建同一个 `ProtoMotionsHumanMeshAdapter`，明确 `shape_std=0`，并在实际 `world.step` 后、reset 前更新网格。根位置和原生 XYZW 根四元数来自真实机器人状态，关节按语义名映射；转换采用 `v_world = R_root C(v_LBS − J_pelvis) + p_root`，`C = Rz(π/2)Rx(π/2)`。[实现](../../../CrowdSim/tools/smpl_mesh_visualizer.py:174)。

| 问题 | 当前可复用做法 / 官方约束 |
| --- | --- |
| 动态表面 | 复用 `Mesh.GetPointsAttr().Set(Vt.Vec3fArray...)`。USD points 是局部坐标；当前 overlay 的父 Xform 为 identity，故写入世界顶点合法。不要再放到已变换 Robot 下造成二次变换。[OpenUSD PointBased](https://openusd.org/release/api/class_usd_geom_point_based.html) |
| UsdSkel 替代 | 动画需要有序 joint-local translation/rotation/scale、绑定与 skin weights；SMPL 还具有 pose corrective blend shapes。只写关节变换不能替代当前完整 LBS，暂不重建资产。[UsdSkel schemas](https://openusd.org/release/api/_usd_skel__schemas.html) |
| 隐藏碰撞体 | 现有 adapter 只对已覆盖的 Robot 调 `UsdGeom.Imageable.MakeInvisible()`；不改 `PhysicsCollisionAPI` 或 `physics:collisionEnabled`，不使用 `SetActive(False)`。可见性与碰撞是独立属性。[Imageable](https://openusd.org/release/api/class_usd_geom_imageable.html)、[CollisionAPI](https://openusd.org/release/api/class_usd_physics_collision_a_p_i.html) |
| UV / 材料 | 保留现有 faceVarying UV、`UsdPreviewSurface` 与 `UsdUVTexture` 绑定；不再翻转 V，不逐帧创建材质。已有纹理不是作者相同资产。[USD PreviewSurface](https://openusd.org/dev/spec_usdpreviewsurface.html) |

## RGB-D 的名称、单位与坐标

继续使用原 `_capture_setup` 创建的一台 camera 和一个 render product；`rgb` 与新增 `distance_to_image_plane` **attach 同一个 product**。Replicator 的后者返回二维 `float32`、单位为米、轴向图像平面深度；`distance_to_camera` 才是相机中心欧氏距离。其文档规定无命中时值为 0；本地输出同时保留 +inf 并用显式有效掩码统计。[Replicator 1.13.30 Annotators](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.13.30/source/extensions/omni.replicator.core/docs/API.html)。

USD camera 局部轴为 +X 向右、+Y 向上、沿 −Z 看；因此正轴向深度不是世界 Z，也不是脚距地面。若以后导出点云，需要真实相机 world transform 的逆以及真实 focal length、aperture、分辨率派生 K；本适配不伪造 K 或 timestamps。[OpenUSD Camera](https://openusd.org/release/api/class_usd_geom_camera.html)。Isaac Sim 6 的 `RtxCamera` / `CameraSensor` 也提供相同深度名称，但本次已有 Replicator product 足够，不迁移相机框架。[Isaac Sim 6 camera](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_camera.html)。

## 单 App、暂停边界与新增薄适配

`SimulationApp.update()` 会更新整套系统；不能当作天然无物理副作用的调用。[Isaac Sim 6 SimulationApp](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.simulation_app/docs/index.html)。保留原 [simulate.py](../../../CrowdSim/humanoid/simulate.py:60) 的有界 RGB helper：刷新 physics view、暂设 `/app/player/playSimulations=False`、首次 12 / 后续 3 次原 App update、finally 恢复设置。新增 helper **不再调用 App update / world.step / sim.step / orchestrator.step**，不创建第二 App、Context 或 stage。IsaacLab 的 singleton context 也应复用；当前安装 3.0 的 renderer 行为以本机源码为准。[IsaacLab SimulationContext](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html)。

新增 [capture_rgbd.py](../../../CrowdSim/humanoid/application/capture_rgbd.py) 的 `capture_helpers(output, interval=..., depth=..., camera_eye=..., camera_target=...)`：

- 临时替换原 setup/frame，返回原三元组和原 RGB；setup 恰一次，可成对覆盖 camera eye/target。
- 原 RGB helper 完成同一次有界更新后读取 depth；按单次 evaluation 的 `control_step = frame_index × interval + 1`，exclusive 写 `depth/frame_{control_step−1:06d}.npy` 和逐帧 flush 的 `index.jsonl`。
- 保存原米值为 float32；0/+inf 保留；负值、NaN、−inf、尺寸不匹配拒绝。index 有有效比例/范围，`actual_synchronization_verified=False`。
- context 退出恢复原函数，并尝试 detach depth；SDK fast exit 可能绕过 Python finally，不能声称一定完成 detach。

**同 product 与读取顺序只保证接线，不能证明 renderer 数据新鲜、RGB-D 同帧或 K 正确。** 后续独立短 capture 应保存真实源 control/episode、相机与 renderer 元数据，并核验原 capture 前后状态和 physics counter；物理、策略和动作质量仍使用原结果。

实际纯 CPU fake seam 测试为 9 passed / 0.14 s，工具 chunk `744ac0`、exit0；Ruff chunk `f69999`、exit0。覆盖同 product、无新增渲染调用、帧编号、原米值、无效深度、exclusive 文件、warmup 与异常恢复；这些是单元接口证据，**不是实际图像证据**。[小记录](actual_CPU_seam_checks.json)。

性能需实际测量：现实现每帧执行 LBS 并复制顶点到 CPU，再写 USD points，可能产生 GPU 同步；拓扑、UV、材质应只创建一次。4 人顶点位置 payload 约 330,720 B/帧，15 人约 1,240,200 B/帧，仅为数组体积，不是显存、FPS 或总体吞吐保证。优先复用现有单镜头和低频捕获，由 root 在正式训练之后串行执行。
