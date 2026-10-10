# 同一 Social AMP 的 SFM 评价接入：CPU 已验证，物理待执行

新增入口 [`evaluate_shared_social_sfm.py`](../../../scripts/evaluate_shared_social_sfm.py) 和桥接 [`social_force_runtime.py`](../../../CrowdSim/humanoid/trajectory/social_force_runtime.py)。此阶段继续实现[论文IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)的同AMP高层对照。作者八场景任务／SFM参数未公开；当前只接入已定义的四人cuboid fixture，不能称为作者完整消融。

## 接入边界

入口在独立module globals内复用当前S1/S2的规范评价入口及冻结物理driver，通过逐字节可逆的有界替换接入SFM。没有修改任何旧入口或22项S2 guarded来源。7项planner/provenance替换与2项计数标签替换均完整逆还原；未改变world.step/reset、AMP admission、actor/RMS/mask、原始目标时序、因果参考、contact/foot captures、计数分母或退出清理helper。

当前API限定一个完整四人共享scene；拒绝拆成两个小batch。current `[4,5]`与history最后态、全部other当前态逐项对应，mask/device/时钟/目标完整核验。SFM只消费当前位置／速度和注册A*的5秒端点；十个waypoint仍传入，但选最后端点并固定整个预测horizon。其它Diffusion优化项明确不应用，未知新增objective直接拒绝。预测输出raw51；原driver继续执行同一causal adapter与AMP。selected_sample=0只是确定性SFM的兼容archive标志，不是Diffusion采样结果。

SFM使用原[公开完整方程与150步同步推进](../sfm_shared_amp_preparation_v1/README.md)。fixture网格由**相同物理cuboid YAML**的XY投影生成，.1m cell与cuboid相交即占据，ROS row0为最大Y。通过[SciPy1.17 metric EDT](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.ndimage.distance_transform_edt.html)计算米制距离；外部occupied padding参与EDT。这个保守网格是本地扩展，不是native地图验收；不支持未知／域外穿越、未注册native scene或额外actor。

本分支不构造Diffusion网络、不读取其tensor、不消费planner RNG。CLI仍要求原显式Diffusion路径/SHA，作为两臂共享资源的未消费绑定；provenance明确标为unused，runtime的diffusion_used_for_rollouts=false、diffusion_plans=0，trajectory_plans计实际SFM生成行。实际AMP仍按原训练成功回执、终态权重及全部RMS严格读取，没有将fixture或旧default权重替代S1。

## 实际CPU与来源证据

| 检查 | 实际结果 | 退出tool chunk |
| --- | --- | --- |
| 最终source derivation | 完整inverse、shared helper AST相同；Torch=false，SDK=false | `94038b`／0 |
| 五项CPU bridge fixtures | 5 passed：cell边界、小cuboid、full-scene当前态、RNG/输入、未知objectives、源漂移与公共物理调用 | `0295b5`／0 |
| 最终Ruff | 四个新文件通过 | `ad3dee`／0 |
| 实际S1 resource preflight | 同AMP终态及formalDP40000绑定通过；未创建输出／加载NN／启动SDK | `45edfc`／0 |
| 原始S1 snapshot→raw→reference | 实际4人post-control1输入，raw `[4,51,2]`→reference `[4,151,2]` | `aaf682`／0 |

最后一项读取已实际正常退出的S1导航及独立CPU审计，复用全24,000 actor-controls／原native force/phase核验。固定选择第一个完整control、全部四人；输入是已保存的真实post-step root XY、XYZW、末120Hz子步native速度及注册waypoints。Torch只做CPU张量和原reference运算，没有加载任何权重、policy/NN/optimizer。输入及CPU RNG保持，CUDA未初始化。最大参考速度1.500000119m/s、加速度2.000001669m/s²，初始overspeed四人均false；F32误差按既有容差核验。原数组、raw及dense输出SHA分别保留在`snapshot_CPU.json`和`snapshot_CPU_arrays.npz`。

这不是live SFM rollout，也不是普通live replan、AMP动作或物理稳定性验证。记录输入来自DP闭环，不能宣称SFM已经驱动出同样行人姿势。

首次source检查因选择不存在的源码end marker退出1（`789437`），没有启动SDK；失败日志和确切entry源码保留。修正为实际main边界后通过；后续完整call检查补足peer/mask/device/时钟和显式eval参数要求，并修正计数标签。所有初次／最终日志及source快照见`execution.json`，没有用最终源码冒充失败时源码。

## 下一实际SDK步骤

S2及其自动全动作／风格／导航流程仍独占GPU。本阶段没有新增GPU任务。
其流程完成后，先运行新输出的30控制步SFM smoke，核对真实进程退出、原生视图释放、全部raw/ground/reference数组及实际planner class origin；然后才执行6000控制步／200s。

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_sfm.py \
  --execution-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_execution.json \
  --diffusion-checkpoint output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt \
  --diffusion-sha256 4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff \
  --control-steps 30 --output output/diffusion_amp/shared_social_S1_sfm_smoke30_v1
```

这条SDK命令当前**未执行**。只读resource检查需额外`--preflight-only`和新output参数；实际preflight不会创建该目录。不要把CPU退出0升格为SDK正常退出。

未来完整两臂应重新配对：固定同AMP checkpoint、seed、原始AMASS清单／scene／actor ID／routes、参考配置、120/30Hz、200s，并核对actual初始rigid-body snapshot和RNG来源。当前historical DP结果不能自动作为新SFM匹配成功证明。预测失败须保留部分数据、实际非零退出及计划／实际曝光，不自动回退、删除failure或重跑挑选最佳。四个native场景及N3–15的接入、八场景协议恢复、多seed/style/route/physics仍未完成；默认权重未晋升。
