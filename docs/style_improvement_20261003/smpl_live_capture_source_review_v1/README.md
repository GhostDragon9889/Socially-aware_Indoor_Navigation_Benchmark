# 当前 AMP live skin 与 RGB capture：源码补正

2026-10-04。仅检查当前源码及已有小型 JSON 记录，没有导入 Torch/项目模块、反序列化模型、读取整份 capture、启动 SDK/GPU 或执行渲染。生产185、训练和旧冻结记录均未修改。

## 结论与历史补正

**现有 bounded AMP evaluation 已接好实时 SMPL 纹理覆盖层；无需新增 renderer 或 skin adapter。** [历史外观报告](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/smpl_visual_appearance_source_review_v1/README.md:9) 对 `AMPWorld` 本身不建网格的说明成立，但其第60行“要另做 adapter 接线”遗漏了 runner 的 `--capture` 分支。本补正保留历史报告原字节。

当前边界如下：

| 模式 | 当前源码行为 |
|---|---|
| 默认训练、无 `--capture` 的物理评价 | 不在此入口创建 skin overlay |
| bounded `--evaluate-steps … --capture` | 创建 live textured SMPL，读取实际 post-step 状态，捕获 RGB |
| `--capture` 且非 bounded evaluation，或同时 `--physics-only` | 原 parser 明确拒绝 |
| RGB-D | 本路径只装 `rgb` annotator，尚无 depth 输出 |

[configuration.py:186](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/application/configuration.py:186) 明确限定 capture 模式；[train_paper_amp.py:716](/project/EXP/SangforAndSaut/ProtoMotions/scripts/train_paper_amp.py:716) 已调用 `enable_human_mesh(...hide_humanoid=True, shape_std=0.)`、`from_simulator`、`create` 和 `update`。

## 实际状态、更新与 reset 顺序

`world.step(applied_actions)` 在 [第755行](/project/EXP/SangforAndSaut/ProtoMotions/scripts/train_paper_amp.py:755) 完成。physical 数组随后记录该 post-step 状态；captured step 在 [第832行](/project/EXP/SangforAndSaut/ProtoMotions/scripts/train_paper_amp.py:832) 调用 `mesh.update()`，第835行读取 RGB，第841行写 PNG，最后才在第844行 `world.reset(ids)`。因此终止/跌倒帧先被捕获，后重置。源码未发现 reset 顺序或缺失 live update 的阻断。

`create()` 先写入实际有限网格，随后才隐藏物理 Robot 的 USD 可见性；`update()` 失败会恢复骨架可见性并抛错。[adapter.create/update](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/tools/smpl_mesh_visualizer.py:162) 没有改 PD、action、刚体状态或 collider。`SMPLMeshVisualizer.create` 创建 `UsdGeom.Mesh` 与材质，不给皮肤添加刚体/碰撞 API。

[RGB capture](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/simulate.py:59) 暂停 `playSimulations`，执行有界12次首帧/3次后续 app update，再恢复该设置。原 deterministic evaluation 已在捕获前后比较实际 `world.state()`；本次只读检查没有将这个源码诊断机制冒称新的实际运行通过。

## 可直接复用的 API 与坐标合同

| API | 输入与作用 |
|---|---|
| `enable_human_mesh` | 仅设置 model/texture/shape/appearance 环境参数，本身不建 USD |
| `ProtoMotionsHumanMeshAdapter.from_simulator(simulator)` | 使用实际 simulator 的 actor 数、device、资源配置 |
| `adapter.create()` / `adapter.update()` | 从 `_robot.data.joint_pos/root_pos_w/root_quat_w` 读取实际状态，写现有视觉网格 |
| `SMPLRobotPoseMapper(joint_pos, joint_names)` | `[N,69]`；按23个非 Pelvis body 的名称对应XYZ三轴，支持 semantic 与 IsaacLab D6 `:0/:1/:2` 名称，不猜 backend 排序 |
| `SMPLMeshVisualizer.update(body_pose, root_pos, root_quat_wxyz)` | `[N,69]` SMPL局部姿势、`[N,3]` world米坐标、`[N,4]` WXYZ根四元数 |

[adapter.update:184–195](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/tools/smpl_mesh_visualizer.py:184) 显式读取 joint names，并由 simulator 的 native quaternion order 转成 WXYZ。SMPL body 顺序由 [SMPL_JOINT_NAMES](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/tools/smpl_mesh_visualizer.py:25) 定义；[姿势映射](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/tools/smpl_mesh_visualizer.py:239) 为每个 body 查找名字，不按 MJCF/PhysX index 混用。

[compute_world_vertices](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/tools/smpl_mesh_visualizer.py:598) 使用 neutral SMPL LBS，减去模型 Pelvis，施加固定 `yaw90 @ Rx90` 的 SMPL→robot局部坐标变换，然后用实际根 WXYZ旋转、world根位置。capture 显式 `shape_std=0`，保持 neutral 几何。仅有 body position 不能恢复全部 skin rotation；现有实际 DOF 与根状态接口已提供所需信息。

## 现有资源与材质

原资源为 `data/smpl/SMPL_NEUTRAL.pkl`（既有模型 symlink）、`data/smpl/smpl_uv.npz`、`data/smpl/textures/{f_01_alb.002,m_01_alb.002}.png`。本次不重新读取/哈希模型或纹理；既有身份及署名见 [安装记录](/project/EXP/SangforAndSaut/ProtoMotions/docs/resource_completion_20261001/smpl_texture_installation.json) 和 [UV_TEXTURE_ATTRIBUTION.md](/project/EXP/SangforAndSaut/ProtoMotions/data/smpl/UV_TEXTURE_ATTRIBUTION.md)。

[材质创建](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/tools/smpl_mesh_visualizer.py:452) 使用 `/World/CrowdSim/HumanMesh/Materials/person_i`、`UsdPreviewSurface`、`UsdUVTexture` 与 `st` reader。[mesh.create](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/tools/smpl_mesh_visualizer.py:500) 使用 faceVarying UV，不再次翻转V轴；实际网格为 `/World/CrowdSim/HumanMesh/env_i/Body`。

录制回放已经复用同一个 mapper 与 visualizer，在 [replay:327](/project/EXP/SangforAndSaut/ProtoMotions/scripts/replay_paper_pedestrians.py:327) 将存档 XYZW→WXYZ。已有 [900×4 recorded skin audit](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/recorded_smpl_skin_audit_v1/actual_cpu_audit.json:120) 的最大 joint-origin误差为0.23931mm；这是历史已记录姿势的完整检查，不是新的 live capture/RGB-D 或物理质量证明。

`native_visual_scene.compose_native_visual_scene` 是 **render-only replay** 接口，明确禁用场景物理；[其源注释](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/runtime/native_visual_scene.py:151) 要求不可共享 active PhysicsScene。live capture 应沿用原 world 场景，不把这个回放 composer 装进物理世界。

## 剩余最小差异

[simulate._capture_setup:54](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/simulate.py:54) 仅附加 RGB annotator，960×720、focal length20。没有 depth annotator、深度数组或相机标定导出，故这条现有入口只能称 RGB capture。若后续要求 RGB-D，最小工作是同一 render product/同一捕获时刻添加深度与其单位、时间、标定记录；skin接线可直接保留。本次没有实现或执行该扩展，也没有发现需要重写蒙皮、renderer 或 physics/controller 的具体源码缺陷。

当前小型 source binding 见 [source_bindings.json](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/smpl_live_capture_source_review_v1/source_bindings.json)。仅选定文件与185manifest逐byte核对，没有重做全部185源审计或旧suite。
