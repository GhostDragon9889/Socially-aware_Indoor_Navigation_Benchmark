# AMASS／原生 PhysX 关节速度坐标审计（2026-10-02）

## 结论与适用范围

当前 ProtoMotions AMASS 转换器及 v6 curated bank 的局部角速度使用父关节坐标；当前 SMPL PhysX 原生球关节速度使用子关节坐标。直接把二者作为 AMP 的同一组57D速度输入存在真实坐标差异。已经给 **Nav AMP 专用 `AMASSCorpus`** 添加显式适配，默认 `legacy_stored` 保留历史张量，新配置显式选择 `child_joint`。通用 `MotionLib`、RobotState、IsaacLab 后端、原始数据、旧 manifest／checkpoint 均未因此修改。

v7 已实际重新计算为 `child_joint` 并同时声明 bank／split；adapter 检查一致性后原样返回，不再旋转。没有把旧文件改一个标签就当作新数据，也没有把历史训练结果重新标为新坐标训练。此审计解释速度语义，不证明新的风格训练已经收敛。

## 1. 源码链与数学含义

设局部旋转 (R_t) 把子关节向量转换到父关节坐标；四元数为 XYZW，采样间隔为 (\Delta t\)。在相同最短弧取法下：

\[
\omega_{parent}=\operatorname{Log}(R_{t+1}R_t^{-1})/\Delta t,
\qquad
\omega_{child}=\operatorname{Log}(R_t^{-1}R_{t+1})/\Delta t
=R_t^{-1}\omega_{parent}.
\]

这两者都是角速度向量。它们与主值旋转向量的逐分量差分 `diff(dps)/dt` 一般不同。

| 环节 | 当前源码证据 | 实际语义 |
|---|---|---|
| ProtoMotions angular velocity | [pose_lib.py](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/components/pose_lib.py:1400) | `q_next * conjugate(q_current)`；局部旋转输入产生父坐标角速度 |
| AMASS 原始转换 | [convert_amass_to_proto.py](/project/EXP/SangforAndSaut/ProtoMotions/data/scripts/convert_amass_to_proto.py:225) | 对23个非根局部旋转调用上述函数，flatten69 |
| Nav 原仓库函数 | [旧 pose_lib.py](/project/EXP/NavIsaaclab2.0/protomotions/components/pose_lib.py:1313) | 同样使用左乘旋转差 |
| 公共 PACER expert | [motion_lib_smpl.py](/project/EXP/SangforAndSaut/ProtoMotions/third_party/pacer/pacer/utils/motion_lib_smpl.py:44) | `inverse(q_current) * q_next`，子坐标角速度，末帧延续前值 |
| 新 curation 显式两种 frame | [amass_curation.py](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/training/amass_curation.py:240) | `parent_joint`／`child_joint`；v7 实际计算右乘旋转差 |
| IsaacLab simulator | [simulator.py](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/simulator/isaaclab/simulator.py:812) | 直接读取 `robot.data.joint_vel` |
| 当前已安装 Lab | [articulation_data.py](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py:1054) | 来自 `root_view.get_dof_velocities()` |
| COMMON 转换 | [simulator_state.py](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/simulator/base_simulator/simulator_state.py:425) | 只做 DOF 名称重排，没有速度坐标变换 |
| AMP 判别器 | [observations.py](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/amp/observations.py:175) | 除双手／双脚趾外19个关节的57D速度原样进入195D观测 |

PhysX 公共源码的 spherical motion matrix 在子关节局部坐标中建立转轴，随后乘子刚体世界旋转；SDK DOF velocity API 单位为 rad/s。参见 [PhysX joint motion axes](https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/main/physx/source/lowleveldynamics/include/DyArticulationJointCore.h)、[Featherstone velocity reconstruction](https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/main/physx/source/lowleveldynamics/src/DyFeatherstoneArticulation.cpp)、[官方 articulation 文档](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)。公共 main 分支不冒充本机二进制的精确提交；以下同瞬时物理数据验证了本机实际语义。

## 2. 已运行物理探针的 CPU 复核

父代理运行真实32控制步 SMPL tracking pilot；本审计仅在 CPU 读取其33帧捕获，使用32个 post-step 帧。捕获中的 body angular velocity 为世界坐标，body／DOF 已转换到 COMMON 顺序。每个关节计算：

\[
\delta\omega_W=\omega_{child,W}-\omega_{parent,W},\quad
v_{child}=R_{child,W}^{-1}\delta\omega_W,\quad
v_{parent}=R_{parent,W}^{-1}\delta\omega_W.
\]

| 与捕获 native57D 比较 | RMS（rad/s） | 最大绝对误差（rad/s） |
|---|---:|---:|
| 同瞬时 child frame | 1.5418916e-7 | 1.3212348e-6 |
| 同瞬时 parent frame | 0.17432037 | 1.76837272 |

原生速度自身 RMS 为0.65267553 rad/s。该比较使用同一帧的原生刚体角速度，无30Hz有限差分，也没有把120Hz子步时间错位当成坐标误差。

此外 CPU `Usd.Stage.Open` 检查该真实运行资产的 active physx variant：23个关节的 `localRot0`、`localRot1` 全部为 identity。因此此资产的子关节坐标与子刚体坐标一致；这不是任意其他资产的假设。此处未启动 Kit。

完整路径、源文件 SHA、实际关节路径、32片段数值在 [dof_velocity_semantics_audit.json](dof_velocity_semantics_audit.json)。`dof_velocity_semantics_audit_initial.json` 是中间脚本哈希快照，最终证据以上述无后缀报告为准。

## 3. 参考数据与独立 SO(3) 校验

对 v6 的前32个真实片段，stored velocity 与当前父坐标差分实现的最大差为0；与独立子坐标计算的最大差为8.1609020 rad/s。显式 adapter 输出与 SciPy 独立逆旋转计算的最大差为3.3350342e-6 rad/s。片段／frame ID 和全部误差记录保留在 JSON。

一个独立解析例子从旋转向量 `[.4,.6,-.2]` 开始，按子坐标速度 `[.3,-.7,1.1]` 前进1/30秒：右乘旋转差得到该子坐标速度；左乘得到 `[.5934188,-1.1164549,.4374730]`；`diff(exp_map)/dt` 得到 `[.5255802,-.9449232,.8267331]`。据此不能用 DOF position 分量导数代替原生球关节角速度。

## 4. 显式 adapter 契约

入口：[AMASSCorpus](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/amass_corpus.py:94)。

```python
corpus = AMASSCorpus(manifest, split="train", device="cpu",
                     dof_velocity_frame="child_joint")
state = corpus.state_at(indices)
metadata = corpus.provenance()["dof_velocity"]
```

- 允许输出模式 `legacy_stored`／`child_joint`；允许存储标签 `parent_joint`／`child_joint`，不认识的标签、显式 null、entry／bank 冲突直接拒绝。
- 未声明旧数据仅推定为当前 ProtoMotions converter 的 `parent_joint`；provenance 明确标为 assumption。这不能推广为任意外部 bank 的格式保证。
- `legacy_stored` 不旋转；v7 明确 child bank 不旋转。只有显式请求 child 且存储 parent 时，才用选中帧的真实 `lrs[...,1:,:]` 逆旋转69D速度。
- 转换要求实际 `[N,24,4]` local XYZW、有限单位四元数及浮点69D速度；缺少 lrs 只在该显式转换分支拒绝。默认旧读取与明确 child bank 均无需 lrs。
- scalar／向量／多维 frame indices 的所有 leading dimensions、dtype、目标 device 均保留；根速度、世界刚体速度、位置、旋转、DOF position 不变。
- adapter 只换坐标，不改变旧 velocity horizon、平滑或时间估计策略。v7 新数据明确 horizon1／30Hz，与旧参考数值策略存在有记录的区别。

父代理的新训练 profile 显式保存 `expert_dof_velocity_frame`，resume 检查 manifest SHA／bank SHA／split 及此 profile contract。旧 checkpoint 缺少新字段时保持 `legacy_stored`；不会仅因默认值而给历史 checkpoint 自动升级。

## 5. v7 bank 与帧元数据门禁

最终 manifest SHA：`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`。
最终 bank：1,155,817,682 bytes；声明并由资源代理完整校验的 SHA：`46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`。本审计不重复声称重新计算了整个 bank SHA。

CPU 对全部2,848行实际检查通过：1,853个父动作、523,272个30Hz控制帧；其中1,499个片段来自原30Hz，1,349个来自真实50→30Hz重采样。每行 source path、original motion ID、源帧数及源 dt 均与原始 mmap bank 的实际 metadata 对照通过。

- `motion_id`、`original_motion_id`、frames及四种 frame bounds 均为 JSON int；dt／时间字段为 JSON float。bank starts／counts 为 int64，状态／速度及 motion_dt 为 float32，contacts 为 bool。
- `parent_control_frame_start:stop` 是重建30Hz父动作的半开范围，长度等于此片段 frames。
- `parent_source_frame_start:stop` 是原30／50Hz数据的插值依赖半开范围，长度在50Hz时通常不同。它不是片段控制帧数，也不是世界时间步计数。
- 示例50Hz原片段210帧，重建126控制帧：control `[0,126)`，source dependency `[0,210)`；末采样时刻4.1666667s。源 float32 dt `0.019999999552965164` 被记录，插值使用明确验证的 nominal0.02s。
- 源 dt metadata 与 nominal source grid 的最大差1.7384688e-9s；每个父动作最多4段、段间不重叠，无跨片段 padding／重复时间区间。
- entry／bank／所有行及 resampling 的 velocity frame 一致 child_joint；实际抽取512个 frame 的 `dof_vel` 与 bank 原值 `torch.equal=True`，无二次转换。
- train父动作与整个 frozen validation／test 父动作零交叉。该门禁不证明跨数据集人物身份不同，也不证明公开 PACER 预训练从未见过这些人物。

复核器：[verify_v7_frame_metadata.py](verify_v7_frame_metadata.py)；完整机器结果：[v7_frame_metadata_verification.json](v7_frame_metadata_verification.json)、[v7_velocity_adapter_verification.json](v7_velocity_adapter_verification.json)。2,848是公开标签与非重叠分段重建出来的数量，作者原始2,848 ID 清单仍未公开，不能声称 membership 一致。

## 6. 回归与历史可比性

81个 focused CPU 用例通过；包含真实 v6／v7 bank、独立 SciPy SO(3)、默认 bitwise、非 identity 复合旋转、未知／冲突声明、缺少 lrs、scalar／多维索引及 frozen parent leakage gate。详见 [JUnit](amass_velocity_adapter_tests.xml)。业务与审计脚本 Ruff 均通过；本代理没有启动 Kit／CUDA。

旧 style evaluator frozen reference artifact SHA 保持 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`，旧 raw／timed 分数逐项完全相等。新增 adapter metadata 写入独立 report，不改 frozen reference 原对象；style 指标本身来自 body pose／几何导数，未因 DOF velocity 坐标变化伪造历史提升。

复现检查命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python -m pytest -q \
  tests/test_amass_velocity_coordinates.py tests/test_amass_pedestrian_corpus.py \
  tests/test_pedestrian_style_evaluation.py
```

最终业务文件已冻结；后续新训练及新评估必须显式声明 child_joint，以新 checkpoint／manifest 区分旧历史。
