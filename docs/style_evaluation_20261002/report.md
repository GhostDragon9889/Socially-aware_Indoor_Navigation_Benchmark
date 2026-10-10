# 按实际运动速度比较的行人姿态质量评估

## 论文指标与本地合同

论文 **Table II是导航可靠性**：Diffusion+AMP的success/fall/map/agent/path分别为88.67、1.12、2.40、7.81、0.153m；条件是八场景、每场景3–15人、200秒。**Table III才是风格/分布评估**：作者K50/K100 coverage为0.92/0.90，entropy为0.800/0.821，cluster distance为0.305/0.285，MMD为0.327。正文说明30Hz、15节点、root中心、heading对齐和尺度归一化，但没有完整节点ID、尺度公式、cluster sampling或kernel参数。不能用本地MMD²0.092与作者MMD0.327直接判定已超过论文。[NavIsaacLab IV-C、Table II/III](https://arxiv.org/html/2606.26265v1#S4.SS3)。本次还检查了本地PDF第7/8页的表格布局。

论文以2,848个按标签筛选的locomotion clips训练，另有闭环reference tracking（mean位置0.218m、rotation0.343rad；mean位置每步>0.5m即失败，成功64.6%）。这个tracking实验需要逐时刻完整参考与rigid-body rotations，不能由共享导航捕获的root/local target误差替代。公开本地Nav参考源码检索没有找到完整Table III evaluator；缺失作者协议继续保留未知。[论文 IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)。

新指标是独立的本地诊断。其依据是动作风格需要包含时间变化，不能仅看单帧pose coverage；PhysDiff作者明确讨论floating、foot sliding和ground penetration，但这里的脚端代理不是PhysDiff的物理投影或其精确评分实现。[AMP作者项目](https://xbpeng.github.io/projects/AMP/)、[PhysDiff官方项目](https://nvlabs.github.io/PhysDiff/)。PACER公开方法使用独立的locomotion训练与程序目标；其约200个训练clip身份未与本地heldout公开对齐，不能保证公开warm actor从未见过评估动作。[PACER附录B.3](https://arxiv.org/html/2304.01893v1#A2.SS3)。

## 新实现与API

- `CrowdSim/humanoid/style_evaluation.py`只依赖NumPy，不导入Torch/Kit。`sequence_style_samples(body_pos, root_quat_xyzw, episode_ids=…, reset_mask=…, body_names=…, spec=…)`接收`[T,B,24,3]`world body origins与XYZW，返回合法interval的实际root速度、姿态/速度特征和标量统计。
- `compare_speed_conditioned(generated, reference)`按固定actual root speed bins `[0,.1,.5,1,1.5,2,3,+inf)`比较。每个metric/bin需双方至少30个有效样本；缺support为`None`。保存每个bin分母、reference mass coverage、无条件距离，以及相同有效bins上的均值，避免把速度混合比例的改变当风格改善。
- scalar分布用精确empirical一维Wasserstein，不拟合histogram。新增90D pose+local-velocity Gaussian MMD²：45个local位置加45个local速度×0.1秒，固定sigma1、包含diagonal的biased estimator，每bin双方等量、最多256个确定性等间隔样本。保存实际sample indices；这是新局部指标，与旧45D pose-only MMD及作者MMD不同。
- local姿态每帧减Pelvis、移除root yaw，使用每个sequence/actor固定median的head-chain加mean-leg-chain长度归一化。它不是完整人的真实身高；固定尺度避免每帧pose-dependent高度把弯腰/蹲姿改变混入导数。差分均为30Hz真实采样，reset跨界的一阶/二阶项均剔除，不平滑或隐藏尖峰。
- 输出pelvis高度、root up、局部姿态速度/加速度、toe-origin近地支撑/双支撑比例及脚端XY速度。脚端proxy要求两个interval端点z≤0.12m且|vz|≤0.2m/s；slide threshold0.1m/s仅作显式诊断，无论文验收阈值。转动脚掌时body origin也可移动，proxy不是已测contact-patch滑动。
- 真实COM只有传入captured `body_com_world`与matching masses才计算。现有运行只有body origins，因此trueCOM/actual foot-ground contact均为`None`；另外明确报告mass-weighted **joint-origin-height proxy**，没有冒充真实质心。static/peer PhysX接触保留原独立评估，不用toe高度代理替换。

## 固定heldout与公平比较

首次真实CPU分析使用原AMASS **validation22 clips**，seed7按clip均匀抽256个连续61帧窗口，共15,360个合法30Hz interval；本组没有短clip被排除。sample identities、局部起止帧、manifest SHA和实际选中body/rootrot数组SHA全部保存到 `frozen_validation_comparison/heldout_reference.json`。窗口可重叠，不能将相邻frame当独立样本来计算IID confidence interval。

同一artifact复跑时，reference整个字典、actual sample SHA及raw/timed各inclusive score逐项完全相同。没有在generated/validation上fitcluster或bandwidth；旧K50/K100 train-only评估继续独立保留。

每个run核对其声明training manifest与heldout original source-file IDs不相交。resources agent正在扩充**原train bank**并保留旧validation/test身份；这个新CLI可复用固定旧artifact检查新训练manifest。当前严格证明的是selected original source不与声明的再训练数据交叉；跨数据集真实人物身份、公开PACER预训练动作重合及未声明数据仍未知。

建议迭代选模型使用validation，最后test单列；34个test clips已经出现在多轮历史pose报告中，不能将后续结果称为完全未看过的blind test。额外按不同motion IDs将validation参考分成两组，输出自然between-clip距离作尺度背景，不自动把它变成通过阈值。

## 两份真实200秒记录的新结果

两份源分别为`shared_independent_amp_raw_v1`与`shared_independent_amp_continuous_timed_v1`，均使用同一AMP1000（SHA `e0d6ed4a…`）和Diff5000（SHA `63e7cad0…`）。改变的是部署时的guidance/route/reference组合；此对照不能证明新增训练已改善style。

| 新本地诊断 | raw endpoint | timed + causal | 解读 |
|---|---:|---:|---|
| 实际root平均速度，m/s | 0.4446 | 0.6621 | 速度分布改变，单独报告 |
| pelvis高度W1，相同5 bins均值，m | 0.03196 | 0.01666 | 更接近固定参考 |
| local pose velocity RMS W1，相同5 bins均值，chain/s | 0.23389 | 0.21654 | 略接近参考，仍有差异 |
| local pose acceleration RMS W1，相同5 bins均值，chain/s² | 2.95902 | 2.87761 | 描述性改善，不能把小加速度直接当自然 |
| toe近地slide proxy W1，相同4 bins均值，m/s | 0.24017 | 0.25817 | 代理项没有改善；第五bin的timed有效支撑样本不足 |
| 双toe近地支撑比例，无条件 | 77.26% | 64.94% | 固定AMASS为24.79%；速度匹配后仍明显偏高 |
| 新90D pose+velocity MMD²，相同5 bins均值 | 0.09075 | 0.09274 | 没有下降；不是旧45D/作者评分 |

在相同0.5–1m/s bin中，姿态velocity W1由0.11757降到0.07873，但两个部署在低速bin仍与参考不同。new MMD²在明确直立子集（up≥.8、pelvis≥.7m）为0.08977/0.09226，未显示全面style改善；完整inclusive统计和保留比例同时报告。参考不同clip组的MMD²均值为0.02199，但仅有4个supported bins，不能与5-bin结果直接作为显著性阈值。

timed方案0falls/26vertices/5cycles是源导航证据，不是style通过。toe滑动proxy的无条件均值上升还包含更快行走、足部转动及支撑判断的影响；应结合速度分层、真实foot-ground contact、同一任务的多个seed、fixed checkpoint和视频判断，不能仅凭proxy的一个数给控制器贴“不自然”标签。

图：[style_comparison.png](style_comparison.png)。完整bin/分母、upright子集、clip calibration和sources：[frozen_validation_comparison/evaluation.json](frozen_validation_comparison/evaluation.json)；相同common bins摘要：[style_comparison.json](style_comparison.json)。

## 后续固定controller评估入口

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python scripts/evaluate_pedestrian_style.py \
  --run new_controller=output/diffusion_amp/NEW_CAPTURE \
  --reference-artifact docs/style_evaluation_20261002/frozen_validation_comparison/heldout_reference.json \
  --output output/diffusion_amp/NEW_STYLE_EVALUATION
```

输出必须为新目录，源body ordering/30Hz/worldXYZ/XYZW/reset contract由已有capture gate校验，selected持久窗口不能越clip；source NPZ/result/model和training/reference manifests保存完整小文件/模型SHA。多GB AMASS package默认只验证stat与声明checksum及真实tensor元数据，实际选中数组另算SHA；仅`--verify-package-hashes`才重新hash整个bank。

18项CPU回归通过，覆盖translation/yaw/scale、foot anchoring vs gliding、hovering不能假造零slide、reset teleport的一阶/二阶排除、真实COM门禁、非法输入、exactW1/identical MMD、speed-mixture分离、missing support、holdout窗口/泄漏以及clean process无Torch/Kit导入。三个实现/测试与plot脚本Ruff通过。没有启动Kit/GPU或改训练runner、CLI、旧guides、环境/权重。

训练stage建议先用同样任务/seed/时长做固定checkpoint对照：导航可靠性单列；style看速度共同support上的位置+时间分布、支撑和height/up，并记录实际contacts和sampling coverage。保留多指标与unknown，不用单一coverage、MMD或“0次跌倒”宣告论文效果。
