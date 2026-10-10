# SMPL 可视外观与现有姿势回放：只读核查

2026-10-03。本轮只读源码、资源元数据和已有渲染产物；没有启动 Kit、物理仿真、模型推理或渲染。生产 185 文件、训练源、资源和默认模型均未修改。

## 结论

当前行人已经具备连续 SMPL 皮肤、UV、纹理和原室内 USD 场景的回放路径。已有一次真实完成的 Warehouse 录制姿势回放：450 帧、15 fps、30 s，`mesh_mode=textured`。这足以排除“缺 neutral 模型或皮肤纹理才能显示完整行人”的资源问题。

物理训练入口仍直接使用胶囊/盒体。是否显示连续皮肤取决于入口：`AMPWorld` 没有安装网格覆盖层；`CrowdSim/world/builder.py` 和现有录制回放入口会安装它。这是现有入口接线区别。

论文 §III-B 说明 SMPL 骨架对应 skinned mesh，用于 RGB-D；§VII 将可定制服装和体型列为后续方向。因此不能把新增独立服装网格、鞋或布料仿真当成复现论文已实现方法的必要条件，也不能声称本地纹理等同作者资产。[NavIsaacLab §III-B](https://arxiv.org/html/2606.26265v1#S3.SS2)、[§VII](https://arxiv.org/html/2606.26265v1#S7)。本次依据已缓存原文 `public_resource_refresh_v1/paper_html_v1.txt:599–615,2027`。

## 实际资产与接线

| 层 | 实际路径/入口 | 本次确认 |
|---|---|---|
| 动力学刚体 | [amp_world.py](../../../CrowdSim/humanoid/amp_world.py:132) → [SmplRobotConfig](../../../protomotions/robot_configs/smpl.py:55) | 24 body、69 DOF；XML 有 16 capsule、8 box、没有 mesh asset |
| IsaacLab spawn | [scene.py](../../../protomotions/simulator/isaaclab/utils/scene.py:156) | 从 MJCF 转为 USD，然后 `UsdFileCfg` 加载；材质覆盖为灰色 PreviewSurface |
| 实际生成缓存 | [mjcf_to_usd.py](../../../protomotions/simulator/isaaclab/utils/mjcf_to_usd.py:61) | `~/.cache/protomotions/isaaclab_mjcf_usd/smpl_humanoid_<digest>/smpl_humanoid/smpl_humanoid.usda`；已有 7 个缓存入口，本次没有将其中任何一个冒称当前进程所用缓存 |
| 仓库预置 USD | `protomotions/data/assets/usd/smpl_humanoid.usda` | 当前及 `/project/EXP/NavIsaaclab2.0` 对应文件都是 131 B Git LFS pointer，目标 114,283 B；当前 AMP 使用 MJCF 生成 USD，故此缺 checkout 不阻断当前仿真 |
| 连续表面模型 | `data/smpl/SMPL_NEUTRAL.pkl` | 已存在，是指向本机 SMPL v1.1.0 模型的 symlink，实际 247,186,228 B。本轮没有解包/反序列化或重新哈希模型 |
| UV/纹理 | `data/smpl/smpl_uv.npz`，`data/smpl/textures/{f_01_alb.002,m_01_alb.002}.png` | 已存在；UV 266,068 B，两张 4096×4096 albedo；已有拓扑核查和安装身份记录 |
| 蒙皮与材质 | [smpl_mesh_visualizer.py](../../../CrowdSim/tools/smpl_mesh_visualizer.py:407) | 使用 neutral `smplx.SMPL`、faceVarying UV、UsdPreviewSurface+UsdUVTexture；姿势来自实际关节和根状态。纹理中的衣服图案不是独立布料动力学 |
| 实时覆盖层 | [builder.py](../../../CrowdSim/world/builder.py:307) | `human_mesh=true` 时建网格，随后由原 nav 更新；当前 pedestrian 配置还隐藏物理骨架。该配置使用另一运行入口，不等于 AMPWorld 已启用覆盖层 |
| 录制回放 | [replay_paper_pedestrians.py](../../../scripts/replay_paper_pedestrians.py:217) | 原 30 Hz 物理状态 → 姿势映射 → neutral mesh → 渲染；`shape_std=0` 保持已记录 neutral 物理体型 |

模型 symlink 的真实目标为：

`/project/EXP/SangforAndSaut/arena_sangfor/arena_ws/arena_assets/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`。

24 个本地 body STL 也存在，但当前 MJCF 没有引用这些 mesh。纹理来源是 Meshcapade 官方 `SMPL_texture_samples` 的固定 commit `c87d871c7498fdc98f3cabb7341b5c85c38debaa`；原许可证/署名说明与派生 UV 身份见 [UV_TEXTURE_ATTRIBUTION.md](../../../data/smpl/UV_TEXTURE_ATTRIBUTION.md)、[安装记录](../../resource_completion_20261001/smpl_texture_installation.json)、[拓扑记录](../../resource_completion_20261001/smpl_uv_geometry_verification.json)。参数化 SMPL 模型的原许可证单独保留。

## 原室内 USD 与已完成回放

现有 `native_reference_four_scenes_anonymous_root_v9` preflight 实际覆盖 **Warehouse / Full_Warehouse / Office / Hospital**，四者声明 Z-up、metersPerUnit=1；并不是 Factory，也不能据此称作者八场景匹配。资产入口：

- `/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Warehouse/warehouse.usd`
- `/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Full_Warehouse/full_warehouse.usd`
- `/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Office/office_partnav.usd`
- `/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Hospital/hospital_partnav.usd`

[native_visual_scene.py](../../../CrowdSim/humanoid/runtime/native_visual_scene.py:104) 将已完成 capture 的 scene/preflight/map/原 USD 身份绑定起来，保留原世界坐标与材质，在匿名渲染层中关闭原碰撞、刚体和 PhysicsScene；不重写原 USD。本轮没有打开 USD stage 或重验证全部场景依赖。

可直接查看已有产物，无须再启动 GPU：

- [30 s MP4](../../../output/diffusion_amp/native_warehouse_state_prior_recorded_indoor_30s_v1/pedestrian_motion.mp4)，2,315,858 B，SHA `79ff1637511a0c82ff26345d647cd03f371840ab60621cb251309b40964daf7a`。
- [已查看的 10.033 s 帧](../../../output/diffusion_amp/native_warehouse_state_prior_recorded_indoor_30s_v1/frames/frame_00150.png)：可见连续、贴纹理的行人表面和 Warehouse 原室内背景；固定相机未保证所有四人同时可见。
- [真实 renderer result](../../../output/diffusion_amp/native_warehouse_state_prior_recorded_indoor_30s_v1/result.json) 和 [真实 closed owner](../state_prior_native_actual_runs_v1/recorded_replay_actual_launch.json)；owner 为 exit0/child_exit_observed=True。

该视频重放历史 `native_warehouse_state_prior_clearance08_200s_v1` 的四人 capture，并非正在运行的 state-only 新模型。renderer 记录 physics_steps=0、policy_forward_calls=0、simulation_time_advanced_s=0，渲染成功不能替代源物理报告或当前模型的动作质量评价。只有外观/回放路径已实际执行。

## 最小新增回放方案（准备，未执行）

优先直接使用上面的已有 MP4。若需要另一段可对照短片，可复用同一已封 capture 和原脚本，保持原相机、材质与坐标，仅选择 300 controls / 10 s、每 2 controls 一帧。`source_inventory_and_replay_plan.json` 提供精确 argv；先 `--preflight-only`，后由 root 在 GPU 空闲时串行执行原 renderer。输出必须为新路径，原 30 s 成果保持不动。

当前 replay ABI 需要单一完整 result 的 `state_capture` 和 `root_pos/root_rot/dof_pos/body_pos/episode_ids/reset_mask`，明确 30 Hz、common SMPL body/joint 名称、XYZW quaternion、重置时序、真实 controls 数。完整 22 条 tracking 的独立 NPZ/result 不是此 ABI；若以后展示这些 tracking，需要薄 I/O 适配并保留真实重置/时间语义，不能直接伪造 physical rollout 元数据。对于后续已关闭的 fixture/native capture，源文件通过上述原 ABI 后即可复用现入口。

## 仍存在的边界

1. AMPWorld 的录制/训练入口默认没有 live skin overlay。已有离线回放足以观察真实已记录动作；若要求该入口实时 RGB-D，要另做现成 adapter 的最小接线并验证读取的是实际 post-step 状态，不能用另一配置的显示成功代替此验证。
2. 现有回放固定 neutral 几何并使用两个样本 albedo；没有作者相同皮肤/材质/相机/光照身份，不能宣称作者视觉效果完全匹配。可定制服装/体型属于论文未来方向。
3. 外观不会修正追踪误差、停滞、接触或跌倒。当前 state-only 新模型及正式 Diffusion 的物理指标、作者八场景匹配仍须按实际结果分别判断。

本轮只新增本报告和少量路径/源码身份、已有产物身份及未来 argv 清单；没有新增 launcher、蒙皮数学、物理接口或资产框架。
