# 记录姿势的纯视觉回放

`scripts/replay_paper_pedestrians.py` 将已有 `physical_states.npz` 的 root XYZ、root quaternion XYZW 和 common69维 exp-map DOF 转为 neutral SMPL mesh，仅渲染 USD 场景。它不加载 AMP/Diffusion 权重，不生成新轨迹，不创建物理 humanoid，也不调用物理 step。模式固定为 `recorded_pose_visualization`，`rendering_only=true`；`status=passed` 只表示渲染入口执行完成，不能继承 source 的零跌倒结果成为新闭环验证。

## 执行

使用现有 IsaacSim6/Lab3 专用环境；输出目录必须不存在。先进行不导入 Torch/Kit 的 CPU preflight：

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/replay_paper_pedestrians.py \
  --states output/diffusion_amp/shared_independent_amp_continuous_timed_v1/physical_states.npz \
  --source-result output/diffusion_amp/shared_independent_amp_continuous_timed_v1/result.json \
  --output output/diffusion_amp/recorded_pose_replay_preview \
  --start-step 0 --steps 900 --capture-every-steps 2 --preflight-only
```

去掉 `--preflight-only` 后启动实际渲染。默认1280×960、15FPS、源记录首30秒；`--steps 6000 --capture-every-steps 2` 可回放全部200秒。支持显式 `--camera-eye X Y Z` / `--camera-target X Y Z`。不会覆盖已有输出或 source。

## 合同

- preflight 核对源 result 的实际 capture path、完成状态、30Hz／world米／XYZW／common SMPL 声明；24body顺序与固定 MJCF 相同，源 semantic 或 D6 joint names 必须构成相同69DOF双射。
- 原始记录为 post-control samples，索引0对应1/30秒；source初始 pre-step pose未记录。source窗口、实际捕获索引和逐帧时间单独保存。可视帧顺序连续，按声明FPS写MP4；窗口不能越界，不跨 reset 插值。
- body_pos的Pelvis须等于root_pos；姿势必须有限、root quaternion必须单位长；episode IDs与before-frame reset masks必须一致。逐帧保留 source episode/reset标记。皮肤由当前neutral SMPL和名映射重建，不声称记录了原始mesh顶点或全body renderer逐位等价。
- 直接使用当前 `SMPLRobotPoseMapper` / `SMPLMeshVisualizer`，shape_std=0，appearance seed来自源seed。保留本地licensed SMPL模型，不下载/再分发许可资源。光照、摄像机、地面与source cuboids为可视场景，不构成新的接触验证。
- timeline处于play状态以驱动渲染图，`set_auto_update(False)`固定其时间；`playSimulations=False`，不创建PhysicsScene。RGB annotator attach前关闭Replicator Orchestrator/reference-time gate，激活HydraTexture更新；仅用有界`app.update()`取RGB，不调用Replicator step/async或physics forward。每次app更新后timeline时间必须保持启动值，否则失败且不将时钟回写来隐藏推进。第一帧最多120次app update（至少48次warmup），后续最多24次（至少3次），分别有120/30秒循环期限；单次SDK update的内部耗时不能由Python期限抢占。
- 每张PNG和MP4每帧永久显示 `Recorded physical state replay | source t=...s`，时间来自post-control源索引。源RGB曝光统计在叠字前计算，banner不掩盖空帧/黑帧。
- source NPZ、source result、当前common rig和实际neutral模型保存完整SHA256；启动provenance与截图逐帧统计单独保存。source轨迹结果仍需查看其原物理run报告。

## 当前CPU证据与SDK来源

15个CPU用例通过，包含shape/nonfinite、姿势坐标、D6双射、控制时间、reset边界、源路径绑定、fresh输出、clean进程无Torch/Kit导入，以及fake SDK下有界app update、当前RGB读取、推进时间硬拒绝和永久banner。真实200秒source的900步preflight通过，见 `recorded_replay_tests.xml`、`recorded_replay_preflight.json`。fake SDK仅检验接口编排，不能证明真实RTX产帧；本次子任务没有启动Kit。

实际pilotv1的`output/diffusion_amp/recorded_pose_replay_pilot_v1/result.json`是`failed`，原因`Recorded-pose render fence exceeded120s`，没有产生帧。SDK关闭流程使shell exit0，不能据此把pilot判为成功；该失败证据保留。v2移除该async fence后由主任务实跑：pilot完成6帧／0.4秒，随后 `output/diffusion_amp/recorded_pose_replay_200s_v1` 完成3000帧／200秒／15 FPS／1280×960带纹理回放。

CPU事后核验全部3000个源索引／时间／reset flags／episode IDs、所有timeline0→0记录、完整源/代码/模型/UV/纹理SHA和MP4实际解码帧数通过。首／中／末PNG非空、永久banner源时间正确。完整结果见[recorded_replay_actual_verification.md](recorded_replay_actual_verification.md)和对应JSON；实际视频仍只作为已有物理记录的视觉展示，不是新闭环零跌倒验证。主任务在pilot后增补源码/runtime/texture metadata；`recorded_replay_capture_fix.json`的CPU修复hash是中间快照，实际完整回放使用的最终源码hash以运行result与事后核验为准。

安装SDK `omni.replicator.core-1.13.25+110.1.1` 的 `omni/replicator/core/scripts/annotators.py:1410–1414`读取`Orchestrator/enabled`/`disableAnnotatorGate`决定是否开启reference-time gate，`:1522`使用`render_product.hydra_texture.set_updates_enabled(True)`。`orchestrator.py:963–965`在Orchestrator disabled模式不产生trigger/capture事件；这里读取的是直接RGB annotator，而不是writer调度。`omni.timeline-1.1.0`的`_timeline.pyi:203,221`提供`play`/`set_auto_update`，`:73`提供秒单位时钟读取。当前IsaacLab `isaac_rtx_renderer_utils.py`使用`RealTimePathTracing`；摄像机复用项目`_capture_setup`，RGB更新方式参照`_capture_frame`的有界app循环，纯视觉入口不调用其中`physics_manager.forward()`。SDK证据与v2冻结hash见`recorded_replay_capture_fix.json`。
