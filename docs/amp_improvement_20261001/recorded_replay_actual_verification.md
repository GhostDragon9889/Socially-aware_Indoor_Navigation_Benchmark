# 实际200秒记录姿势回放核验

CPU核验结果：**PASS**。主任务运行的 `output/diffusion_amp/recorded_pose_replay_200s_v1` 生成3000张PNG和H.264 MP4，1280×960、15 FPS、实际解码3000帧、200.000秒。MP4为4,432,454 bytes，SHA256 `d98e9a2f4da99a5b642cabc27993032214279401d62f17abb5234114a77f298a`。实际渲染wall time为580.711秒；本核验进程仅运行CPU/Pillow/NumPy/ffprobe，没有启动Kit、GPU或SMPL计算。

这是 `recorded_pose_visualization`，将已经保存的物理姿势重建成带纹理SMPL视觉帧。回放不产生新策略轨迹，不提供新的物理接触／跌倒验证。`status=passed`表示渲染执行和本项工件核验完成。

## 完整性与源绑定

- 源为 `shared_independent_amp_continuous_timed_v1/physical_states.npz`：6000个30Hz post-control帧、4人、24body/common69DOF。完整SHA256 `693492d700c5daf7bd720676212c8e1c6945dd9a8d545d0f14270fc27cb646e4`；源result完整SHA256 `007175bd7d9faab92a335993f31f90dbc97fc0e27f06309bf5856c0182233f6e`。源result声明的capture path与实物严格绑定。
- 全部3000个回放索引严格为 `0,2,…,5998`，源时间为 `(index+1)/30`，episode IDs/reset flags与NPZ逐项相同。首帧源时间0.033333秒，末帧199.966667秒。视频第0帧呈现的是源第一个post-control状态；source初始pre-step状态和索引5999的最终200秒状态不在这次抽帧中，不跨reset插值。
- 全部3000条timeline前／后记录均为0，`simulation_time_advanced_s=0`、`physics_steps=0`、`policy_forward_calls=0`、未创建PhysicsScene/物理humanoid。首帧48次app update，其余2999帧各3次。全部原始RGB均值在211.349383–211.574021之间，无空帧／黑帧曝光门槛失败。
- 当前4份渲染源码、固定SMPL MJCF、licensed neutral模型、UV以及2份纹理的当前完整SHA256/bytes均与实际启动manifest一致。4个人的纹理分配为女／男／女／男。metadata增补后的实际回放脚本SHA为 `06dffb167d420279c6b39c5193e6f5b24b358488bbd9117af49a71563566a929`；先前`recorded_replay_capture_fix.json`中的`dc221…`只对应CPU修复时的中间冻结快照。
- 实际运行manifest：IsaacSim6.0.0.1、IsaacLab产品3.0.0／包12.0.0、Lab commit `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`、Torch2.11.0+cu128、NumPy2.3.1；mesh计算device为CPU，RTX模式`RealTimePathTracing`。本核验记录manifest，并核验源码/资源实物，不另启SDK验证这些运行版本。

## 首／中／末帧实物

| 视频帧 | 源索引 | 源时间（秒） | banner下方RGB均值 | banner下方RGB标准差 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0.033333 | 211.464972 | 19.073408 |
| 1500 | 3000 | 100.033333 | 211.572939 | 18.990169 |
| 2999 | 5998 | 199.966667 | 211.602663 | 19.013798 |

三张PNG均完整解码，尺寸一致、非单色；永久banner `Recorded physical state replay | source t=...s`与各源时间逐像素重建匹配。排除banner后，首→中／中→末分别有395830／383724个像素变化，并保存了相应源root XYZ。该检查确认工件非空且视觉帧变化，不能凭像素变化单独证明所有mesh顶点与原物理body逐位一致。主任务已另行查看首／中帧，记录四人和fixture可见、中央障碍产生自然遮挡；不承诺每个人在每一帧都无遮挡。

## 源物理证据的边界

源物理run记录200秒0次fall reset、最低pelvis0.633056m、route vertices `[8,6,6,6]`；这些来自**源闭环物理run**，不由回放重新验证。实际源模型根据evaluation arguments解析并计算实物SHA，避免误用继承的training provenance：

- Diffusion：`social_diffusion_training_extended_v2/step_5000.ckpt`，SHA `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`。
- AMP controller：`amp_independent_curriculum_v1/step_1000.pt`，SHA `e0d6ed4a9acb39c581bed92fab722a426bf23c34a3949bc48712cc13bf04c333`。

回放本身不加载以上权重。源仍明确 `author_experiment_reproduced=false`、`convergence_claim=false`。自定义共享fixture、重建模型和样本协议不能替代论文八场景、作者权重或2848个clip清单。

v1真实pilot因120秒reference-time fence超限失败，原失败报告保留。修复后的v2真实pilot完成6帧、0.4秒、纹理模式且timeline0→0；之后完整200秒回放通过。本次未修改业务源码或环境。

完整机器报告：[recorded_replay_actual_verification.json](recorded_replay_actual_verification.json)。可重跑CPU工件核验：

```bash
CUDA_VISIBLE_DEVICES='' ../IsaacLab-Protomotions/.venv/bin/python \
  docs/amp_improvement_20261001/verify_recorded_replay.py \
  --render-dir output/diffusion_amp/recorded_pose_replay_200s_v1 \
  --output /tmp/recorded_replay_actual_verification.json
```

核验使用完整file hashes和ffprobe `-count_frames`实际解码计数，不能使用`python -O`关闭验收断言。
