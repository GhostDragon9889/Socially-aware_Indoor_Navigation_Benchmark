# AMP + Diffusion 行人动作风格改进

本阶段使用 Isaac Sim 6.0.0.1、Isaac Lab 3.0 开发提交和当前 ProtoMotions 的 SMPL PhysX 刚体。
目标是改善自然步态、转向与动作跟踪，并用保留的动作验证结果。混合 AMASS／导航课程已完成 2,000 轮训练：完整验证成功数从负方向惩罚基线的 11/22 升至 **19/22**，均零跌倒，位置误差由 0.301 降至 0.200 m。但混合模型的实际四人导航仍有跌倒，未替换零跌倒基线；整体动作风格没有全部改善，尚未达到或超过论文效果。

## 1. 2,848 段动作的实际来源

推荐数据入口是 [`manifest_labeled_v7.json`](../data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json)。
它包含 **2,848 段真实、互不重叠的动作片段，来自 1,853 个 AMASS 原始动作文件**。
这不是论文作者未公开的 2,848 个原始动作 ID，也不代表 2,848 次独立采集。

- 699 段采用 CMU 官方动作描述，2,149 段采用有导航含义的文件名；没有以未知数值编号直接推定为行走的动作。
- 每个来源最多四段，保留原始动作 ID、来源、时间及源帧/控制帧边界；没有重复时间片段或补零凑数。
- 1,349 段经过真实 50→30 Hz 根位置插值和局部旋转 SLERP；重算当前中性 SMPL 的 FK 和速度。
- 1,131 段支持至少 8.3 秒的 Diffusion 窗口，其余有效片段至少 3 秒，供 AMP 使用。
- 原验证集 22 条、测试集 34 条动作的来源和 ID 保留。按父动作检查训练与保留集隔离；跨数据集的真人身份不能仅由来源命名证明。
- 风格分类是运动学代理：转向 1,568、行走 720、站立/观察 427、侧移 116、跑步 17。类别数量不是人工逐帧风格标注。

[完整覆盖审计](style_improvement_20261002/amass_style_coverage_v7_read_only_v1.md)进一步区分了整段标签和短窗运动：
17 段跑步代理共 58.4 秒，但真实十帧历史内的快步窗口有 6,063 个，来自 168 个父动作；窗口数量不是实际采样概率。
427 段站立代理中 180 段名称含 push/recovery，存在普通静止与推扰恢复混类风险。
其中仅 103 段满足本次“运动学安静”代理，3 段表现为低根速度但明显足部活动，321 段仍未定。
这些是来源名称及运动学诊断，不是人工语义真值，也不证明 AMASS 记录了外力。
已列出的短跑和原地跑候选尚未通过全部 FK、质量与重复门禁，没有自动加入训练库。

最终训练库 SHA256：`46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`。
数据清单 SHA256：`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`。
完整检查见 [v7 数据验证](amass_curation_20261002/verification_v7.json) 和
[源帧/控制帧边界检查](style_improvement_20261002/v7_frame_metadata_verification.json)。

进一步的逐动作时间标签可使用 [BABEL 官方标注](https://babel.is.tue.mpg.de/data.html)。
本次访问其官方下载入口跳转至登录页，本地没有该标注包，因此没有声称完成 BABEL 时间段筛选。
现有 v7 数据及评估集保持不变；不会从未知动作编号或非官方复制包补造标签。

## 2. 已修正的关节速度坐标

真实 PhysX 探针表明，球形关节 `joint_vel` 是 **子关节坐标系中的相对角速度**。
在 32 个真实控制帧、57 个有效坐标上，它与同一时刻的
`R_child⁻¹ (ω_child_world − ω_parent_world)` 的 RMS 差为 **1.54×10⁻⁷ rad/s**。
使用父坐标系计算时，差为 **0.1743 rad/s**。当前 SMPL 的 23 个关节局部框架旋转也已核对。

新训练库使用 `axis_angle(q_t⁻¹ q_(t+1))/dt`，与公开 PACER 的约定一致。
[`AMASSCorpus`](../CrowdSim/humanoid/amass_corpus.py) 提供显式 `child_joint` 适配：
对已有父坐标速度使用当前局部旋转转换，对已声明的子坐标速度直接读取。
训练配置和 checkpoint 记录此约定，续训检查约定一致。
旧 checkpoint 的 `legacy_stored` 模式保留其历史数据约定，便于审计和复查。

适配范围是 Nav 行人 AMP 的数据接口。当前模拟器读取显式 link-origin 速度，已核对其与 FK 位置约定一致。
详见 [关节速度证据](style_improvement_20261002/dof_velocity_semantics_audit.md) 和
[link/COM 读取检查](style_improvement_20261002/body_link_com_velocity_audit.md)。

## 3. 新训练方案与代码结构

首轮实验配置：[`amp_style_amass_replay.yaml`](../configs/diffusion_amp/amp_style_amass_replay.yaml)。
它的完整跟踪验证出现退步。已完成且改善跟踪的单变量配置是
[`amp_style_amass_backward_penalty.yaml`](../configs/diffusion_amp/amp_style_amass_backward_penalty.yaml)。

| 部分 | 本阶段配置 |
| --- | --- |
| 低层控制 | 独立程序化 XY 轨迹指令 + 真实 SMPL PhysX + PPO/AMP |
| 高层轨迹 | 单独训练的五维 Diffusion；本次 AMP 训练不调用 Diffusion |
| 初始化 | 公开 PACER actor、critic、状态判别器及对应统计量 |
| 判别器 | 10×195 状态历史，BCE/softplus；这是明确的 PACER 适配方案 |
| 判别器回放 | 65,536 条原始策略历史；与当前 rollout 各占负样本的一半 |
| 更新频率 | 每个 rollout 独立进行两次判别器更新 |
| 左右动作约束 | 原始观察镜像后重新归一化，两分支都反向传播；系数 50 |
| 有效动作 | likelihood、entropy、bounds、mirror 使用 57 个实际有效坐标 |
| 统计适应 | 保留公开均值/方差，显式将初始统计计数限制为 200,000 |
| 奖励 | AMP 权重 2、任务权重 0.5；属于本阶段优化配置 |
| 更新保护 | 目标 KL 0.02；保存实际接受的更新次数 |
| 轨迹指令 | 速度目标 0–2 m/s，向量加速度上限 1.5 m/s²；不是物理速度硬限制 |

```text
AMASS 源库 + CMU 描述
  └─ training/amass_curation.py → v7 动作库/来源清单
       └─ amass_corpus.py → 明确的关节速度约定
            ├─ amp_world.py → reset、专家历史和真实 PhysX 步进
            └─ training/trainer.py
                 ├─ PPO、独立 AMP 判别器更新
                 ├─ replay.py：原始策略历史回放
                 └─ symmetry.py：左右镜像约束

独立轨迹指令 → AMP 训练
已训练 Diffusion → 部署参考轨迹 → AMP actor → 真实物理状态
  ├─ motion_tracking.py：完整动作跟踪与失败阈值
  ├─ style_evaluation.py：冻结保留集的速度分段风格比较
  ├─ foot_support.py：四足部位与实际地面的过滤法向力
  └─ foot_capture.py / ground_support_archive.py：120 Hz 力、真实 COM 和时间/episode 归档
```

轨迹/动作层分开训练参考 [Trace and Pace](https://arxiv.org/html/2304.01893v1)。
历史回放与动作先验依据 [AMP](https://xbpeng.github.io/projects/AMP/)。
本配置对 [NavIsaacLab](https://arxiv.org/html/2606.26265v1) 做了明确的风格改进，
不能当作作者原配置或权重。

## 4. 已完成的实际验证

| 验证 | 结果 | 能说明的范围 |
| --- | --- | --- |
| 新训练/镜像/速度/数据/评估专项测试 | 326 项通过 | 此次测试门禁，不是所有仓库测试 |
| 2,848 段数据检查 | FK、速度公式、片段边界与真实 batch 通过 | 数据接口有效，不保证物理可跟踪 |
| 原 319 条动作的短训练及真实续训 | 累计 12,288 状态转换 | 回放、优化器、统计量、RNG 与物理状态续训流程有效 |
| v7、256 环境、4 轮短训练 | 32,768 状态转换，3 次跌倒 | 新数据和训练接口可运行；不是收敛验证 |
| 旧 actor、22 条完整验证动作 | 6/22 成功，1 条跌倒 | 独立的完整动作跟踪基线 |
| 旧 actor 全身跟踪误差 | 位置 0.3773 m，旋转 0.8595 rad | 按完整帧统计，保留初次失败和跌倒后的帧 |
| v7、新 actor、1,000 轮训练 | 8,192,000 状态转换；414 次跌倒 reset、26,953 次时限 reset | 探索性程序化指令训练；这些比例不是论文场景跌倒率 |
| 新 actor、22 条完整验证动作 | 5/22 成功，2 条跌倒 | 与旧基线相比退步，未提升为最佳模型 |
| 新 actor 全身跟踪误差 | 位置 0.4310 m，旋转 1.2541 rad | 11,982 个真实 post-step 帧；包含失败和跌倒后的帧 |
| 同轮训练第 500 轮 checkpoint、22 条完整动作 | 1/22 成功，0 条跌倒；位置 0.4413 m、旋转 1.0279 rad | 训练样本 4,096,000；是中间诊断 checkpoint，不能称等预算消融 |
| 反向惩罚、1,000 轮训练 | 8,192,000 状态转换；313 次跌倒 reset、26,995 次时限 reset | 与正号训练的配置只相差方向项；采样数据、初始化及预算相同 |
| 反向惩罚、22 条完整验证动作 | **11/22 成功，0 条跌倒；位置 0.3007 m、旋转 0.8615 rad** | 22 对初始状态、参考数组与 future XY 指令逐元素相同；全程保留失败后的帧 |
| 分层专家采样、1,000 轮训练 | 8,192,000 状态转换；203 次跌倒 reset、27,048 次时限 reset | 只改变专家采样先验；较少训练跌倒不能作为效果验收 |
| 分层专家采样、第 1,000 轮完整验证 | 8/22 成功，2 条跌倒；位置 0.3529 m、旋转 0.7913 rad | 相同完整验证输入；未替换反向惩罚基线 |
| 同采样训练第 500 轮完整验证 | 1/22 成功，0 条跌倒；位置 0.4605 m、旋转 0.8700 rad | 中间检查点诊断；零跌倒不能抵消跟踪失败 |
| 真实地面捕获、四人 200 秒 | **0 次跌倒、21 个路线顶点；6,000 控制步、24,000 个连续物理子步** | 独立 CPU 238 项原始数据/来源检查通过；单场景单 seed，不是论文八场景成功率 |

测试日志：[native_style_training_gate.log](style_improvement_20261002/native_style_training_gate.log)。
旧控制器完整验证：[result.json](../output/diffusion_amp/motion_tracking_legacy_full_validation_v1/result.json)。
新控制器完整验证：[result.json](../output/diffusion_amp/motion_tracking_amass2848_child_full_validation_v1/result.json)。
第 500 轮完整验证：[result.json](../output/diffusion_amp/motion_tracking_amass2848_child_step500_full_validation_v1/result.json)。
反向惩罚完整验证：[result.json](../output/diffusion_amp/motion_tracking_amass2848_backward_penalty_full_validation_v1/result.json)。
本阶段训练源代码：[92 文件快照](style_improvement_20261002/native_style_training_source.json)。

跟踪基准每条动作只做一次初始 reset，然后仅向 actor 提供十个未来根位置 XY。
没有按帧传入目标姿势或重置跌倒行人；逐帧全身平均位置误差超过 0.5 m 时记为失败。
短程截断不能计入完整动作成功率。作者动作列表与其他未公开实现细节仍使该基准属于方法重建。
旧控制器采用历史父坐标速度初始化，新控制器采用声明的子坐标速度初始化；
两次运行的重置速度与 AMP 初始历史有明确差异，因此比较是两套控制方案的结果，不是只改变权重的因果消融。

风格评估使用 [冻结验证参考](style_evaluation_20261002/frozen_validation_comparison/heldout_reference.json)，
按实际速度分段比较姿态、速度、步态代理与支持样本数。
现有捕获的脚近地/低垂直速度指标是代理，不是实际脚地接触。
论文 Table III 的 MMD 与本地指标的特征、采样和归一化不同，数值不能直接互换。
按本地同一个冻结风格协议，旧模型的 inclusive MMD² 为 **0.09367**，新第 500 轮为 **0.05023**，
新第 1,000 轮为 **0.20648**。第 500 轮在此任务的风格与跌倒上改善，但完整位置跟踪仅 1/22 成功。
新第 1,000 轮的直立子集 MMD² 为 0.04542，仍有跌倒和朝向退步，不能以删去跌倒后的分数替代主结果。
完整速度段、关节部位与预算区别见 [三权重比较](style_improvement_20261002/tracking_style_comparison.md)。
反向惩罚模型的 inclusive/upright MMD² 均为 **0.04682**，但实际速度分布误差和近地脚原点代理没有全面改善。
旋转误差仍为 0.8615 rad，腕部局部姿态仍有较大残差。见 [四权重及严格配对分析](style_improvement_20261002/backward_penalty_style_v1/report.md)。

## 5. 已完成的长程训练与复跑命令

已完成实验：`output/diffusion_amp/amp_style_amass2848_child_v1`。
实际 256 环境×32 步×1,000 轮，共 **8,192,000 个真实物理状态转换**。
训练主体耗时 1,477.20 秒，不含 Kit 初始化。第 200、500、1,000 轮完整 checkpoint 及续训状态已保存。
第 1,000 轮权重 SHA256：`ee8ea57c4b8566905eb070d2f6dad00a61c9ce3b5ae3c765dd4f014f1b8ac77f`。
训练期间没有调用 Diffusion，详见 [训练结果](../output/diffusion_amp/amp_style_amass2848_child_v1/result.json)。

```bash
source ../IsaacLab-Protomotions/.venv/bin/activate
python scripts/train_paper_amp.py \
  --config configs/diffusion_amp/amp_style_amass_replay.yaml \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --trajectory-source pacer_curriculum --curriculum-mode smooth \
  --curriculum-speed-max-mps 2 --curriculum-accel-max-mps2 1.5 \
  --num-envs 256 --updates 1000 --rollout-steps 32 --episode-steps 300 \
  --terrain-extent-m 192 --spawn-margin-m 48 --checkpoint-every 100 \
  --synchronized-replanning --physics-only \
  --output output/diffusion_amp/amp_style_MY_NEW_RUN
```

## 6. 方向项的配对修正与剩余问题

首轮训练的后退帧比例从第 1 轮的 **0.1125** 增至第 500 轮的 **0.3126**、
第 1,000 轮的 **0.5663**。末轮自身体前方向速度均值为 **−0.1776 m/s**，
但位置奖励仍为 **0.9854**。这说明较高的位置奖励不能证明身体朝向和步态正确。

[论文式 (6)](https://arxiv.org/html/2606.26265v1) 印为
`1 − exp(min(0,u)/1.8)`；在 `u<0` 时它给正奖励，与正文“防止后退”的描述相反。
原配置保留该公式。新的 `backwards_penalty` 对照配置只将这一项取负，
所有其他 `TrainingConfig` 字段、v7 数据和公开初始化保持相同；这属于显式优化实验。
真实等预算训练与配对的完整验证支持这一修正：末轮后退比例从 56.63% 降至 **2.23%**，
身体前向速度均值从 −0.1776 升至 **0.8858 m/s**。仍不能认为它解释或解决了全部姿态问题；
合法倒走动作 `SFU/0005/0005_BackwardsWalk001_poses.motion` 损失一条成功，后续应区分违背指令与需要倒走的情况。

资源预检：[backward_penalty_preflight.json](style_improvement_20261002/backward_penalty_preflight.json)。
必要训练门禁：145 项通过，见 [日志](style_improvement_20261002/backward_penalty_training_gate.log)。
对照训练源码：[93 文件快照](style_improvement_20261002/backward_penalty_training_source.json)。
复跑时将上面的配置改为 `amp_style_amass_backward_penalty.yaml`，并使用新的输出目录。
该对照已完成，输出目录为 `output/diffusion_amp/amp_style_amass2848_backward_penalty_v1`。
训练主体耗时 1,507.27 秒，固定第 1,000 轮 checkpoint SHA256 为
`62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b`。

另一个独立风险是地形观察统计量：平地训练使 1,024 个高度坐标的标准差中位数从公开初始化的 1.528
降至第 500 轮的 0.335、第 1,000 轮的 0.240。按实际 `(center−height)×5` 特征公式，
人工输入“中心地面为 0、单元障碍高 1 米”时，新统计量会将全部单元归一化后截断，公开统计量约为 15.8%。
这是 [checkpoint 统计诊断](style_improvement_20261002/terrain_normalizer_diagnostic.json)，尚未证明真实导航退步；
保持地形统计或增加障碍训练需要后续独立实验，未混入本次方向奖励对照。

继续按冻结验证集比较动作风格、完整跟踪、跌倒和导航表现。
真实脚地支撑探针 v1 在 SDK 物理视图查找时失败，未进入接触验证。
v2 已通过当前 Lab 的公开物理视图读到真实四子步地面力并验证 COM 组合，
但固定中性 PD 姿态在第 59、60 控制帧倒伏，因此整体仍为失败，不等于步态质量通过。
原结果见 [v2 原生探针](../output/diffusion_amp/foot_support_native_qa_v2/result.json)。
场景碰撞监测也已改用当前世界所属的物理视图，保留传感器所有权记录；30 项必要测试通过，
见 [兼容修正门禁](style_improvement_20261002/contact_manager_gate.log)。它未被本次无场景碰撞采集的训练导入。
八场景、多随机种子及与作者相同的风格协议也尚未形成完整对照。
训练奖励提高不能单独证明动作更自然或超过论文。

## 7. 真实地面支撑与下一项采样改进

v3 探针完成地面站立、离地及离地触墙三个阶段：地面/墙过滤、四子步时钟和真实 COM 检查通过，
但固定 PD 站立在第 60 个控制步跌倒一次，因此保留整体失败结果。
[83 项原始数据复核](amp_improvement_20261001/foot_support_native_v3_validation.md) 明确区分传感通道有效与姿态控制失败。

可选 `--measure-ground-support` 评估路径已接入实际 AMP 行人：接触报告在世界初始化前启用；
每个控制步归档四个已完成的 PhysX 子步，保存四足部位法向力、实际质量、link 状态和真正的身体 COM。
地面档案独立于既有 `physical_states.npz`，原 peer/static 四字段及默认关闭行为保留。
219 项集成测试通过，见 [集成审查](style_improvement_20261002/ground_support_integration_review.md)。
四人 200 秒实测经 [独立 238 项核验](style_improvement_20261002/ground_support_200s_actual_verification.md) 通过；
双足支撑占 48.02%、无足部支撑占 0.75%，有支撑时脚身体原点切向速度代理均值 0.07314 m/s。
身体与障碍、其他行人的过滤法向接触均为零。该值不是接触点滑移；接口没有精确接触点时间戳或原生接触点速度，
见 [SDK 接触位置速度合同](amp_improvement_20261001/sdk_contact_point_velocity_contract.md)。

[导航风格与停滞分解](style_improvement_20261002/backward_penalty_navigation_style_v1/comparison.md) 使用同一个冻结参考：
本地 inclusive MMD² 从前阶段模型的 0.09274 降至 0.07238，
但速度分布 Wasserstein 从 0.25129 增至 0.35337 m/s、低于 0.1 m/s 的间隔从 4.26% 增至 24.35%。
旧模型到达 26 个顶点/5 循环，新模型为 21/4；两模型数据、训练预算及数值模式不同，不是权重唯一变化。
actor 0 最长连续慢速段为 38.27 秒，原始 Diffusion 0.5 秒位置仅距当前身体约 3.4 cm，
执行参考误差约 4.2 cm，真正路线顶点仍远。这支持优先检查高层近时域指令不足；尚不能归因于地形统计。
同一个新模型的引导权重 20→40 完整 200 秒对照已结束，保留相同支撑捕获与数值模式：
到达 29 个顶点/6 循环，但跌倒一次、出现同行人接触；未替换零跌倒配置。
更强的全时域引导减少了停滞，也带来了稳定性和局部风格的取舍。

独立 [专家窗口采样器](style_improvement_20261002/expert_window_sampling.md) 已验证 v7 的 494,792 个合法窗口中心，
按真实十帧窗口分配低速、快行、侧移、转向和移动样本；完整来源、概率与 RNG 状态可复查。
集成已完成并通过 104 项采样专项测试、真实 v7 抽样/恢复和独立复核；保持 reset 分布及原 corpus RNG 消费。
见 [完整集成合同](style_improvement_20261002/expert_sampling_integration.md)。
联合训练/采样/关节坐标/测量/冻结评估的 **490 项 CPU 测试通过**，
见 [最终联合门禁](style_improvement_20261002/integrated_training_measurement_gate_review_v2.md)。
最初三项隔离测试失败及修复记录保留；修复了关闭捕获时的测试上下文，并明确 AST 检查仅针对控制步循环。

真实 GPU 试训 4 轮及恢复后新增 4 轮均完成，累计 65,536 个物理状态转换；
前段 5 次、续训段 1 次跌倒不属于效果验收。随后从公开初始化重新启动并完成 1,000 轮等预算训练，
输出 `output/diffusion_amp/amp_style_amass2848_stratified_experts_v1`，没有从试训或上一模型续训。
启动源码见 [108 文件快照](style_improvement_20261002/stratified_expert_training_source_v1.json)。
完成 **8,192,000 个物理状态转换**，训练段 203 次跌倒 reset、27,048 次时限 reset，主体耗时 1,599.65 秒。
这些训练 reset 不是导航场景跌倒率。第 1,000 轮 checkpoint SHA256 为
`93c25e567d88cc4f4c639b1dea5e14494f0ef51dd304661fb23d36690e55328d`。
独立保存了真实第 500、1,000 轮；第 200 轮未及时留存，不以其他轮次冒充。
见 [真实训练结果](../output/diffusion_amp/amp_style_amass2848_stratified_experts_v1/result.json)
和 [检查点记录](../output/diffusion_amp/amp_style_amass2848_stratified_experts_v1/snapshots.json)。
固定 22 条完整验证已完成，第 1,000 轮为 8/22 成功、两条跌倒；第 500 轮为 1/22 成功、零跌倒。
第 1,000 轮局部腕旋转和身体 yaw 误差减小，但位置误差、成功数、速度分布及冻结风格指标退步。
inclusive MMD² 从基线 0.046821 升至 0.089115；仅保留未倒伏帧也升至 0.091907，不能归因于两个跌倒片段。
完整数据见 [冻结风格对照](style_improvement_20261002/stratified_experts_style_v1/report.md)、
[同帧误差分解](style_improvement_20261002/stratified_expert_tracking_decomposition.md) 和
[训练/输入配对审计](style_improvement_20261002/stratified_expert_training_comparison_audit.md)。
本轮没有整体改善，未替换反向惩罚基线。后续真实根轨迹课程的跟踪结果见第 9 节。

## 8. 总引导强度不变的近时域加权

新增显式评估参数 `--near-horizon-waypoint-mass .25`，保持总 goal 权重 20，
将十个定时路线点的归一化系数从各 0.1 改为 `[0.2, 0.2, 0.075×8]`。
前一秒最终占总权重 0.4，`.25` 是新增分配质量，不是最终近域占比。
梯度与候选过滤采用同一目标；默认 0 保留原均值计算分支。
这是工程改进实验，不能写成论文作者参数。

151 项专项 CPU 检查及真实 Diffusion5000 的两个捕获上下文、100 步 DDPM 回放完成；
旧默认与当前参数 0 的全部输出逐位相同。两个上下文的加权效果混合，不能据此宣称导航改善。
见 [实现与回放证据](style_improvement_20261002/near_horizon_waypoint_implementation.md)。
联合训练、恢复、测量及引导门禁 **596 项通过**，见
[联合日志](style_improvement_20261002/integrated_near_horizon_training_measurement_gate_v1.log)。

四人 200 秒单变量物理对照已完成，模型、场景、seed、同步规划时钟、参考适配及接触阈值保持基线设置。
启动来源见 [112 文件快照](style_improvement_20261002/near_horizon_eta025_source_v1.json)，
[实际参数预检](style_improvement_20261002/near_horizon_eta025_actual_preflight.json)。
结果为 **20 个顶点、两次跌倒**，未优于原 21 个顶点、零跌倒配置。
冻结 inclusive MMD² 为 0.077826，原配置为 0.072381；速度分布差距也没有改善。
见 [三配置风格对照](style_improvement_20261002/near_horizon_eta025_style_v1/report.md) 和
[340 项原始地面/来源核验](style_improvement_20261002/navigation_near_horizon_eta025_comparison_v1/ground_support_actual_verification_v4.md)。
核验通过表示捕获数据和来源有效，不表示动作效果通过。
默认配置尚未改变。后续训练实验接入真实 AMASS 根 XY 轨迹课程，严格记录 reset 帧、平移和动作结束时钟；
不把逐帧目标姿势输入策略，效果仍需真实训练及保留集验证。

## 9. AMASS 原始 XY 轨迹训练课程

新增 `configs/diffusion_amp/amp_style_amass_root_curriculum.yaml` 和
`CrowdSim/humanoid/training/amass_trajectories.py`，显式选择 `--trajectory-source amass_root`。
它相对当前负方向惩罚基线改变训练指令、合法 reset 相位与片段结束规则；
核心 TrainingConfig、专家均匀采样、PPO、PD、物理步长及风格奖励保持相同。
这是工程训练课程，不能冒充论文作者未公开的训练配置。

- 保留全部 2,848 条 v7 真实动作；reset 要求十帧真实历史及至少 31 个后续真实控制步。
- 每段在 reset 时绑定实际动作编号和起始帧，只平移一次。目标不随实时物理根位置重锚，跟踪滞后不会被抹掉。
- 原始区间直接消费 30Hz、151 点轨迹，下一控制目标与实际源帧一致；策略仍只收到十个未来 XY 点及物理自身/地形状态。
- 边界之外的未来 XY 使用最后十一帧、十个间隔估计的速度延续；不增加 AMASS 动作帧，不称作真实未来数据。奖励和物理 rollout 在真实末帧截断，末状态按现有时限语义 bootstrap。
- 课程、阶段、固定偏移、源末端速度、计数及共享 corpus RNG 随完整训练检查点保存；异常和不匹配的恢复状态被拒绝。

[组合 CPU 门禁](style_improvement_20261002/integrated_amass_root_gate_v1.json) **643 项通过**，
[独立审查](style_improvement_20261002/amass_root_curriculum_independent_review.md) 另核验真实源阶段和逻辑恢复。
全部 409,352 个合法 reset 相位与 256 个出生位置的目标边界通过解析检查；这不证明全身安全或实际跟踪稳定。
启动 [124 文件快照](style_improvement_20261002/amass_root_linear_training_source_v1.json)
重新计算并核对训练 bank、manifest、公开初始化和 Diffusion 文件 SHA。

真实 GPU 四轮短训完成：32,768 个物理状态转换。已封存
`output/diffusion_amp/amp_style_amass_root_curriculum_pilot_v1/step_4.pt` 和 `result_step4.json`；
恢复后新增四轮已完成，累计 65,536 个转换；两段分别 1/0 次跌倒、227/330 次真实末端截断。
完整权重、优化器、回放、随机状态和 256 actor 源阶段的 [独立实际审计](style_improvement_20261002/amass_root_actual_pilot_independent_review.md) 166 项通过。
1,000 轮训练已从公开初始化重新开始并完成，不从短训继续；
输出 `output/diffusion_amp/amp_style_amass_root_curriculum_v1`。
实际 [启动参数预检](style_improvement_20261002/amass_root_actual_training_preflight_v2.json) 使用每 100 轮保存，
独立 CPU 监控在训练开始前守候 200/500/1,000 轮。
实际完成 **8,192,000 个物理状态转换**，训练主体耗时 1,688.68 秒。
157 次跌倒 reset 与 79,923 次时限 reset 是训练统计，不是导航场景发生率。
固定终点 `step_1000.pt` SHA256 为 `dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e`。

### 完整验证与风格结果

22 条动作的 **11,982 个实际后步帧全部保留**，没有删除失败或跌倒后的帧。
各模型的源参考、未来 XY、相位、初始物理状态及 AMP 历史逐位相同。

| 权重 | 真实训练转换 | 成功 | 跌倒动作 | 全帧位置误差 m | 全帧旋转误差 rad | 冻结 MMD² | 速度分布距离 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 负方向惩罚 1,000 | 8,192,000 | 11/22 | 0 | 0.300708 | 0.861454 | 0.046821 | 0.143767 |
| AMASS 根轨迹 500 | 4,096,000 | 12/22 | 1 | 0.357862 | 0.875744 | 0.038601 | 0.047519 |
| AMASS 根轨迹新续训 750 | 6,144,000 | 16/22 | 2 | 0.401208 | 0.805335 | 0.085542 | 0.053742 |
| AMASS 根轨迹 1,000 | 8,192,000 | **17/22** | **0** | **0.229298** | **0.831882** | 0.056075 | **0.065651** |

第 1,000 轮新增六条成功动作，没有丢失原十一条成功；根 XY 误差减少 37.48%，
腕关节旋转误差减小，但身体朝向对齐后的局部位置误差增加 5.21%。
第 500 轮风格距离较小，却有跌倒且位置跟踪退步，因此不能只按 MMD 选权重。
本地 MMD² 和成功率协议不等同于作者论文协议，不能直接声称超过论文数值。
见 [完整跟踪分解](style_improvement_20261002/amass_root_curriculum_tracking_v1/report.md)、
[独立 1,000 轮协议核查](style_improvement_20261002/amass_root_step1000_tracking_independent_review.md)、
[冻结八模型风格报告](style_improvement_20261002/amass_root_style_v2/report.md)。

为检验中间权重的风格与跟踪折中，已从真实封存的第 500 轮恢复，**新增 250 轮、2,048,000 次物理转换**；
新终点累计 750 轮、6,144,000 次转换，46 次跌倒 reset 是该续训段统计。
这是一次新续训，不是之前 1,000 轮运行遗失的第 750 轮快照；PhysX 求解缓存未序列化。
终点封存于 `output/diffusion_amp/amp_style_amass_root_from500_to750_v1/step_750.pt`，
SHA256 为 `1e87b82b643a1d94bf2117de5a57a27ebb1e0e9811a4cabf66ab1588fcaf9592`。
续训 [178 项独立核验](style_improvement_20261002/amass_root_from500_to750_independent_terminal_review.md) 通过。
完整验证已完成：16/22 成功，BackwardsWalk001 与 Moonwalk001 两条跌倒；所有 442 个跌倒帧仍计入风格和跟踪主结果。
直立子集的 MMD² 为 0.03717，不能取代计入失败帧的 0.08554。
冻结九模型结果见 [风格报告 v3](style_improvement_20261002/amass_root_style_v3/report.md)。
第 1,000 轮因此选作跟踪候选，并在开启独立测试集之前保存 [选型记录](style_improvement_20261002/amass_root_step1000_candidate_selection_v1.json)。
新四人 200 秒导航已完整执行：**13 次跌倒、5 个路线顶点**，对照为零跌倒、21 个顶点，**未替换旧零跌倒部署基线**。
这是训练指令和时长向真实 Diffusion 部署转移的失败证据，不以完整 AMASS 跟踪改善掩盖。

### 部署与独立测试

新导航完整采集 6,000 个控制步、24,000 个真实物理子步，所有跌倒与重置记录保留。
[502 项独立原生测量核验](style_improvement_20261002/amass_root_step1000_navigation_ground_actual_verification_v1.md) 通过，
只说明记录完整，不表示效果通过。与旧基线相同的冻结风格 MMD² 为 0.07820（旧 0.07238），
速度分布距离 0.51975（旧 0.35337），四人慢速停留合计 251.27 秒（旧 194.77 秒）。

在选型记录封存之后首次开启完整 34 条独立测试，随后对固定第 1,000 轮权重只评估一次。
全部 7,739 个后步帧保留，无缩减或挑选动作：

| 权重 | 成功 | 跌倒动作 | 位置误差 m | 旋转误差 rad |
|---|---:|---:|---:|---:|
| 负方向惩罚 1,000 | 4/34 | 1 | 0.445018 | 1.522401 |
| AMASS 根轨迹 1,000 | 11/34 | 0 | 0.344130 | 1.535234 |

测试结果见 [固定基线](../output/diffusion_amp/motion_tracking_amass2848_backward_penalty_full_test_v1/result.json)
和 [固定候选](../output/diffusion_amp/motion_tracking_amass_root_curriculum_full_test_v1/result.json)。
这是泛化检验，后续课程设计依据此前完整验证和导航失败，不按该测试集挑选参数。

[两权重静态迁移核查](style_improvement_20261002/amass_root_transfer_static_review_v1.md) 确认地形归一化 1,024 个坐标逐值相同；
没有把退化归因于地形归一化。未来五秒前向位移均值 2.774→1.332 米，平均训练 episode 为 298.41→102.09 步。
后续固定环境分槽的混合课程已实现，实际训练见第 10 节。

负方向惩罚基线的可视化见 [30 秒实测动作回放核验](style_improvement_20261002/backward_penalty_overview_video_verification.md)；它不是新根轨迹模型的视频。

根轨迹模型的 [完整 200 秒视频](../output/diffusion_amp/recorded_amass_root_curriculum_200s_v1/pedestrian_motion.mp4)
回放上述真实导航状态，保留画面中的来源与时钟标签。
视频为 1,280×960、15 fps、3,000 帧，按固定每两个控制帧取一帧；原始档案中的 13 次跌倒全部保留。
回放没有新增策略调用或物理步，不能作为再次运行的导航验收。
见 [视频核验](style_improvement_20261002/amass_root_step1000_full200_video_verification_v1.json)。
两组记录的 [原生足力时序诊断](style_improvement_20261002/native_gait_timing_v1/report.md)
各保留 96,000 个 actor-substeps，按实际速度分层；没有把阈值抖动次数当作生理步频，也没有补造 AMASS 地面力。

## 10. 固定分槽的 AMASS／导航指令混合课程

新配置为 [`amp_style_mixed_amass_procedural.yaml`](../configs/diffusion_amp/amp_style_mixed_amass_procedural.yaml)，
新模块为 [`training/mixed_trajectories.py`](../CrowdSim/humanoid/training/mixed_trajectories.py)。
通过显式参数 `--trajectory-source mixed_amass_procedural` 启用。
设计依据此前完整验证与四人导航失败，已在训练开始前保存 [计划](style_improvement_20261002/amass_mixed_training_plan_v1.json)。
34 条测试已有评估历史；本阶段没有把它当作新的盲测集或用于挑选课程参数。

- 256 个环境固定划分：128 个使用全部 v7 AMASS 的真实根轨迹，128 个使用原导航程序课程。
  分组按实际环境编号固定，避免短片段频繁 reset 改变两个分支的转换比例。
- 程序分支保持基线的 smooth 指令、2 m/s 速度上限、1.5 m/s² 向量加速度上限和 300 步 episode；这些是指令界限。
- AMASS 分支保持真实十帧历史、固定源相位/平移及源末帧截断。两个分支的奖励目标统一到 30Hz、151 点；策略仍只接收十个未来 XY 点。
- AMASS reset 使用独立 seed 8；原 corpus／专家采样保留 seed 7；程序指令使用独立随机状态。
  三组 RNG、分组编号、实际源阶段、活动计划及两个分支的转换计数随检查点保存。
- PPO、专家采样、PD、动作屏蔽和物理步长保持基线。训练不调用 Diffusion；部署仍使用固定的 Diffusion5000。

[联合 CPU 门禁](style_improvement_20261002/integrated_mixed_curriculum_gate_v1.json) **700 项通过**；
[135 文件启动快照](style_improvement_20261002/mixed_curriculum_training_source_v1.json)
记录实际业务源码和四份资源的重新计算 SHA。
真实 GPU 四轮短训及恢复新增四轮均完成，累计 65,536 次物理状态转换，两个分支各 32,768 次。
[独立实际审计](style_improvement_20261002/mixed_curriculum_pilot_independent_review_v1.md)
分别通过 170／198 项检查，另有 15 项真实 RNG 推进核验；这些证明训练与逻辑恢复可运行，尚未证明动作效果。

**全新 2,000 轮训练已实际完成**，从公开初始化开始，输出
`output/diffusion_amp/amp_style_mixed_curriculum_v1`，没有从短训权重续训。
[实际启动参数](style_improvement_20261002/mixed_curriculum_full2000_preflight_v2.json)
实际产生 16,384,000 次物理转换，各分支 8,192,000 次，训练主体耗时 3,827.67 秒。
训练记录为 182 次跌倒 reset、106,999 次时限 reset；这些是训练事件，不是场景质量指标。
总预算是此前 1,000 轮实验的两倍，不能称等总预算消融。
启动前登记的 [快照监控](style_improvement_20261002/mixed_curriculum_snapshot_watch_launch_v1.json)
封存真实 500／1,000／2,000 轮字节；它只负责运输和基础计数，完整训练状态需另行独立审核。
训练中的 reward、跌倒 reset 和可变 `last.pt` 均不作为质量验收。
固定 22 条完整动作、包含跌倒帧的冻结风格指标和同配置四人 200 秒导航均已完成，结果见下文；两个混合检查点未通过导航选型。

三个固定检查点已封存，[运输账本](../output/diffusion_amp/amp_style_mixed_curriculum_snapshots_v1/ledger.json)
没有缺失轮次。最终权重 SHA256 为 `28aed874464379f0b0628719290d36af0a8b0c53691c0dd51f56102c5f428f72`；
[实际训练结果](../output/diffusion_amp/amp_style_mixed_curriculum_v1/result.json) 的运行状态为完成，收敛声明为 false。

真实第 500／1,000 轮已封存，分别累计 4,096,000／8,192,000 次转换。
两个分支在各检查点都恰好各占一半；两份 [500 轮](style_improvement_20261002/mixed_curriculum_step500_milestone_admission_v1.md)
及 [1,000 轮](style_improvement_20261002/mixed_curriculum_step1000_milestone_admission_v1.json)
中间状态审核均通过 161 项。中间审核没有读取当时正在增长的训练日志，也没有构造完整运行结果；
三个检查点的 [完整终态审核](style_improvement_20261002/mixed_curriculum_terminal500_1000_2000_independent_audit_v1.json)
已通过 220 项；实际 2,000 行日志与各轮 optimizer/RMS/replay、三组 RNG 和分支计数完成对账。
最终 182 次跌倒分为 AMASS 109／程序 73；106,999 次时限分为真实 clip 末端 79,737、
AMASS cap 43、程序 cap 27,219。初始 256 次 reset 另记，没有混入训练终止次数。
[完整验证预登记](style_improvement_20261002/mixed_curriculum_validation_plan_v1.json)
在任何混合模型 GPU 验证前保存。

第 500 轮的 [完整 22 条验证](../output/diffusion_amp/motion_tracking_mixed_curriculum_step500_full_validation_v1/result.json)
已完成，全部 11,982 帧保留：**16/22 成功、零跌倒、位置 0.271462 m、旋转 0.924934 rad**。
与原导航基线相比成功条数增加、位置改善，但旋转误差变差。
[独立配对审核](style_improvement_20261002/mixed_step500_tracking_comparison_v1/report.md)
确认原 11 条成功全部保留；根 XY 改善，heading-local 位置和 parent-relative 旋转较差。
[冻结风格比较](style_improvement_20261002/mixed_curriculum_style_v1/step500_comparison_v1/README.md)
为 inclusive MMD² **0.062570**、速度 W1 **0.090754 m/s**；前者比基线 0.046821 差，后者比 0.143767 改善。
全部失败帧保留，没有按 upright-only 结果挑选模型。尚未据此选择部署权重。
第 1,000 轮 [完整配对验证](style_improvement_20261002/mixed_step1000_tracking_comparison_v1/report.md)
为 **19/22 成功、零跌倒、位置 0.203485 m、旋转 0.875661 rad**，
[冻结风格](style_improvement_20261002/mixed_curriculum_style_v1/step1000_comparison_v1/README.md)
为 MMD² **0.054961**、速度 W1 **0.056390 m/s**。
它的总转换数与原 1,000 轮模型相同，但两个课程分支各只占 4,096,000；不能称每源等预算。
根轨迹与速度改善，局部关节/腕部姿态仍有短板。

第 2,000 轮 [完整配对验证](style_improvement_20261002/mixed_step2000_tracking_comparison_v1/report.md)
为 **19/22 成功、零跌倒、位置 0.200096 m、旋转 0.768363 rad**；
parent-relative 关节旋转为 0.256404 rad，左右腕部为 0.490408／0.458698 rad。
[三检查点全部对照](style_improvement_20261002/mixed_curriculum_style_v1/milestone_comparison_v1/README.md)
保留失败集合变化：1,000→2,000 恢复 motion20，但新增 motion259 失败。
最终 inclusive MMD² **0.060441** 比第 1,000 轮差，速度 W1 **0.045834 m/s** 最好；
不能把腕部改善写成整体风格全部优胜。
[选型登记](style_improvement_20261002/mixed_curriculum_step2000_navigation_candidate_selection_v1.json)
将最终权重选作完整物理导航候选，尚未选为部署模型。
同参数四人 200 秒导航已完成：**11 次跌倒、19 个到达顶点、3 个完整循环**，
障碍接触为 130／24,000 actor-control frames，行人互相接触为零。
[2,259 项独立测量核验](style_improvement_20261002/mixed_step2000_navigation_ground_actual_verification_v1.md)
通过，确认完整 6,000 控制步与全部四子步测量；这是记录完整性通过，导航质量没有通过。
第 1,000 轮相同的完整导航测试已完成：**13 次跌倒、15 个到达顶点、1 个完整循环**，static／peer 接触为零，独立 2,259 项完整性核验通过。
两次结果均没有替换零跌倒、21 个顶点的原基线。
[冻结导航风格对照](style_improvement_20261002/mixed_curriculum_navigation_style_v1/README.md)
保留全部失败帧：基线／混合 1,000／混合 2,000 的 inclusive MMD² 为 0.072381／0.072470／0.116486，
速度分布距离为 0.353368／0.360403／0.306563 m/s。最终模型慢速停留减少，但姿态分布与近地脚原点速度代理变差；该代理不是接触点滑移。

### 真实场景接入

已修复四个 DynBench 场景的缺失地图路径和有效块式 YAML `origin` 读取，40 项 CPU 回归通过，
32 份地图及来源文件字节保持不变。见 [修复与运行时接口计划](style_improvement_20261002/amp_diff_runtime_bridge_v1/README.md)。
这些原始地图仍是离线 USD 投影。早期 Warehouse 原生捕获的坐标门禁失败、v4 的共面地面所有权失败均保留。
[原生探针 v5](style_improvement_20261002/paper_scene_maps_probe_readiness_v5.md)
在不修改源 USD 的 session 中禁用原平面／地面贴花，创建单一有限参考地面。
Warehouse 全部 56,239 个 known5 栅格的原生地面射线与 0.15–1.8 m 高度带检查完成，
520 个格子因真实障碍被阻挡；unknown 未升级为 free。独立 120 项实际核验通过。
其他三场景 v5 暴露了 SDK 一格 padding／Office free-position 缺失。新的 v6 已实际运行四场景：Warehouse 通过，其他三场景因查询时钟或近地非地面障碍分类失败；所有原始结果保留。Office／Hospital 的源 USD 起点为 -1 帧，是负时间线的来源推断，v6 未持久化拒绝时实际 clock，不能冒称已测因果。
[探针 v7](style_improvement_20261002/paper_scene_maps_probe_readiness_v7.md) 的 91 项 CPU 检查及四场景预检通过，固定 reset 后时间线、保存实际 clock，并把已知静态非地面低障碍命中单元保守降为 occupied。
真实四场景 v7 已完成：[独立实际审核](style_improvement_20261002/native_map_admission_v7/actual_four_scenes_partial_v1/report.json)
准入 Warehouse／Full_Warehouse；Office／Hospital 仍失败，查询 counter=2、timeline=−1/60 s，原始源范围为 −1..0 帧。
所有源层重新计算 SHA 后未变，失败结果没有升级为有效地图。新的 session 时间范围修正有真实 OpenUSD CPU 证据，仍须实际 SDK 验证。
[v7 消费端](style_improvement_20261002/native_map_admission_v7/README.md)的 116 项 CPU 检查通过；只由固定探针 SHA 选择新版协议，旧 v5 Warehouse 的描述、四条实际 A* 路径及 FP32／FP64 SDF 值和梯度逐位保持一致。

新 `runtime/dynbench_scene.py`、`native_grid_scene.py` 与 `application/native_deployment.py`
已接入既有单 App／单世界控制循环：真实源 USD、保守栅格 A*、SDF Diffusion 指导、足地与障碍接触使用同一场景。
新组合门禁 **347 项 CPU 检查通过**。Warehouse 首次五秒真实物理运行完成 150 步，
出现三次跌倒，没有到达顶点；独立 492 项原始捕获核验通过，质量未通过。
随后仅以另一个显式配置扩大路线余量至 0.8 m，选择距障碍／unknown／边界至少 3 m 的出生点：
五秒复测零跌倒、零 static／peer 接触，四人位移 4.1–5.4 m；独立 492 项核验通过。
这同时改变了余量与出生／路线位置，不是单变量因果证据。
**新的 30 秒运行已实际完成：900 控制步、零跌倒、8 个到达顶点（每人 2 个）、零完整循环。**
static 接触为 0／3,600 actor-control frames，peer 为 0／5,400 唯一 pair-control frames，最低 pelvis 0.825075 m。
[独立 865 项实际核验](style_improvement_20261002/native_scene_sdk_review_v2/README.md)
确认全部 3,600 个共享物理子步、14,400 actor-substeps、真正 COM 及 132 条原生 all-hit 地面查询。
当前 SDK stage／singleton context 身份一致、一个 active PhysicsScene、一次源 session、单一地板＋745 静态路径；
查询 counter 0→0，53 份源层 rehash 未变。历史 App／Context 构造次数与 native engine 时钟仍未观测。
后续四人200秒实测已完成：6000控制步、零跌倒、零时限reset，57个路径点（14／14／14／15）及12个完整循环。
[983项独立原始核验](style_improvement_20261002/native_scene_200s_actual_review_v1/actual_report.md)
确认24000共享物理子步、96000 actor子步、384000足观测、132条地面查询与source168执行前后字节一致。
static为0／24000 actor-control frames，peer为0／36000无序pair-control frames；测量使用strict法向合力>1N。
这证明该固定四人场景的200秒运行，15人密度和作者八场景效果仍须分别验证。
初次失败与短复测的 [154 文件来源](style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v1.json)
和实际新 30 秒的 [158 文件来源](style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v2.json) 分别保留。
165与168文件历史来源均保留；已完成四人200秒属于source168。
随后V8修正使用独立[source171](style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v5.json)，
合并后226项CPU检查通过。真实四图V8仅Full Warehouse准入；Office／Hospital／Warehouse均因source root RAM时间元数据被改而严格拒绝。
原始文件全层SHA未变，CPU逐字节重构确认变化仅为source root内存的start/end时间范围；具体SDK调用仍不能仅凭此结果归因。
V9匿名运行根层修正已合入：270项CPU检查和Ruff通过，旧V5／V7／V8身份与数学语义保留。
四场景真实探针均已完成资源门禁：Office、Warehouse、Full Warehouse、Hospital的原source RAM与文件不变，SDK接入相同stage，查询范围为零起点；[独立4,441项核查](style_improvement_20261002/native_map_v9_four_scene_independent_review_v1/README.md)已通过，547,321原生格点、2,740源文件层逐一核验，原source RAM保持不变。
执行源码为[source174/V7](style_improvement_20261002/native_runtime_epoch_v7/native_pedestrian_source_v7.json)，不将新源码混记到此前source168的四人200秒。
旧 `CrowdSim.humanoid.simulate` 入口仍使用原 SFM＋MaskedMimic 方法。

### 纯推理与实验腕臂风格

[显式 AMP＋Diffusion 薄入口](paper_pedestrian_deployment/README.md) 复用既有评估循环。
原入口及新 `--inference-only` 各自完成真实 30 秒测试；900 控制步、全部物理／规划／地面字段
与原 200 秒基线前缀逐位相同，零跌倒、两个到达顶点。
[纯推理实际对照](paper_pedestrian_deployment/pure_inference_actual30s_prefix_comparison_v1/report.md)
确认部署不构造优化器、策略回放，不暴露 critic 或判别器结果，明确保留旧初始化的 RNG 消耗。
随后发现旧底层 `policy()` 实际计算过被丢弃的 critic 值；不能把这份历史证据解释为没有 critic 前向。
这不是新的完整 200 秒质量验收。

[actor-only 修正](style_improvement_20261002/evaluation_actor_state_action1000_v1/README.md)
在 CPU 上严格检查完整网络张量后，只保留已加载的 task encoder 和 actor，丢弃 critic、D 和 log-std。
两个真实 1950／2019-D checkpoint 的六批输入各保留三个输出，36 项与旧推理及训练器逐字节相同；
其中物理输入由已保存状态和指令重建，不是原 runner 原样保存的观察数组。
93 项相关 CPU 门禁通过，旧完整 CPU 初始化的 RNG helper 不变。
[新版实际 30 秒对照](paper_pedestrian_deployment/actor_only_actual30s_prefix_comparison_v1/report.md)
对旧 pure30 及原基线200的前900步共154个字段逐字节一致，足力／COM／时钟203项检查通过；0跌倒、2个顶点。
这验证了指定基线的30秒前缀，不是所有配置的通用 CUDA 等价声明。
新版运行源见 [168 文件封存](style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v4.json)。

`runtime/arm_style.py` 支持显式实验腕臂 donor，只混合两侧胸带／肩／肘／腕的 24 个坐标。
其余坐标保持基线，再做一次动作 mask 与 clamp。两份真实网络的 alpha=0
[实际 30 秒对照](paper_pedestrian_deployment/experimental_arm_alpha0_actual30s_prefix_comparison_v1/report.md)
同样全部逐位相同。alpha=0.25 的另一次真实 30 秒试验完成：零跌倒、3 个路线顶点，
相同基线前缀为零跌倒、2 个顶点。
[同窗冻结评分](paper_pedestrian_deployment/experimental_arm_alpha025_actual30s_comparison_v1/README.md)
为 MMD² 0.071807→0.074733，速度 W1 0.207179→0.135578 m/s，停留 5.033→2.767 actor 秒。
主姿态分布略退，速度与停留改善；原生 231 项核验通过，未升级默认。长时稳定性仍待完整 200 秒。

### 状态与动作判别器

[`amp_style_mixed_state_action.yaml`](../configs/diffusion_amp/amp_style_mixed_state_action.yaml)
采用 10×195 历史＋69 个实际动作的 2,019 维判别器输入，独立初始化判别器和 AMP 统计量。
公开 PACER 的 actor／critic 仍作初始化，不加载公开 state-only 判别器。
专家动作是下一 AMASS 关节目标构造的 PD 目标代理，不是 AMASS 实测控制力。
真实四轮及恢复新增四轮已完成，共 65,536 次物理状态转换，分段跌倒 1／0。
[独立 995 项审计](style_improvement_20261002/state_action_isolated_pilot_v2_review/audit.json)
核对模型、两优化器、统计量、回放、物理状态、课程阶段及三组下一次 RNG 采样。
历史 GPU 全模块来源 sidecar 因 SDK 快速退出未生成，该缺口明确保留；新的 v3 启动器在真实导入时同步落盘，
通过了包括 `os._exit` 在内的六项 CPU 回归，完整训练使用此新版。
**新的 2,000 轮全新训练已完成**，输出 `amp_style_mixed_state_action_isolated_v1`；
实际 GPU 86 个项目模块的同步来源已逐项匹配冻结副本。每 500 轮保存，独立 CPU watcher 封存 500／1,000／2,000 轮；
第 500／1,000 轮已封存 4,096,000／8,192,000 转换；
[500 轮](style_improvement_20261002/state_action_isolated_checkpoint500_review_v1/README.md)及
[1,000 轮](style_improvement_20261002/state_action_isolated_checkpoint1000_review_v1/README.md)
各通过 1,077 项独立检查，完整 CPU 恢复包含优化器、回放、统计量、三组 RNG 与实际课程相位。
回放用同一索引读取历史和实际动作；接受数量受已声明的 0.01 保留概率影响，不等于全部训练转换。
实际 GPU 来源账本已记录 86 个隔离项目模块；SDK 快速退出时未观察到终态回调，保留该不可观测边界。
完整 2,000 轮训练已完成：16,384,000 次转换、64,000 控制步，每分支 8,192,000 次转换，主体耗时 4,886.31 秒。
训练累计 429 次跌倒 reset、106,839 次时限 reset；`convergence_claim=false`，这些训练计数不能代替导航评估。
[最终完整恢复审计](style_improvement_20261002/state_action_isolated_checkpoint2000_terminal_review_v1/README.md)
通过 1,079 项检查，实际 PPO Adam 步数 18,218、D Adam 步数 4,000；实际 result 与分支计数一致。
固定最终权重为 [`step_2000.pt`](../output/diffusion_amp/amp_style_mixed_state_action_isolated_snapshots_v1/step_2000.pt)，
SHA256 `eeac037798d2244e0cd4f0a81b40f51020c2aa3320106c7f10620e8c2ddde526`。
[终态训练曲线](style_improvement_20261002/state_action_training_final_v1/report.md)
使用准确数据／判别器标签，不根据 loss 或 reward 自动宣告收敛。
新权重的[完整22条验证](../output/diffusion_amp/motion_tracking_mixed_state_action_isolated_full_validation_v1/result.json)
已完成：17/22成功、零跌倒、全部11982帧保留，位置0.208382m、旋转0.725853rad。
[冻结风格比较](style_improvement_20261002/state_action_style_v1/full22_step2000_comparison_v1/comparison.json)
为 inclusive MMD² 0.062530、速度W1 0.058332m/s；旧12模型结果逐值不变。
相同预算的mixed2000为19/22、位置0.200096m、旋转0.768363rad、MMD²0.060441。
新D的整体旋转较低，但成功数和主风格分布未改善。
[独立逐帧旋转核验](style_improvement_20261002/state_action_style_v1/full22_paired_tracking_review_v1/README.md)
保留全部帧：相对 mixed2000，左右手腕父关节旋转误差由0.490／0.459降至0.372／0.325rad，
但对应局部位置误差由0.211／0.204增至0.235／0.243m；不把旋转改善当作整体动作改善。

完整四人200秒导航已实际完成：6000控制步、2次跌倒、2个路径点（各人0／0／0／2）、零时限reset。
[原始物理审计](style_improvement_20261002/state_action_navigation_ground2000_review_v1/README.md)
通过1148项一致性检查，全部跌倒和失败帧保留；这不代表导航质量通过。
低速停留为同一episode相邻帧根XY速度小于0.1m/s，累计591.733actor-s／23994有效间隔，约73.985%。
相同冻结风格公式的inclusive MMD²为0.073728（基线0.072381）、速度W1为0.738402m/s（基线0.353368）。
障碍接触actor-frames为0；行人接触30actor-frames／15无序pair-frames，峰值法向合力794.13N。
零跌倒／21路径点的部署基线保持不变。
训练代码使用隔离的只读字节副本，防止并行源码改动改变在运行的实验。
这修正了论文 Eq.2 的输入形式，但奖励参数、私有训练数据及作者八场景协议仍有明确差异。


## 11. 姿态多样性与地形统计诊断

[固定15关键点45D诊断](style_improvement_20261002/pose_diversity_candidate_v1/README.md)保留同episode的全部合法区间端点、跌倒和reset边界，不与旧90D指标混算。中心仅由固定AMASS参考拟合；K50／100、初始化、seed及MMD核均记录，未读取34条测试集。新建fixture的基线／mixed2000／state-action2000的45D MMD²为0.020157／0.053822／0.047599；K50 coverage为0.86／0.70／0.68。原生Warehouse四人200秒属于另一场景：coverage为0.34、MMD²为0.038297。零跌倒并未证明动作多样性充分。这套本地指标不是作者Table III的数值复刻。

[原生腕臂alpha=.25同900帧对照](style_improvement_20261002/native_arm_alpha025_30s_style_v1/README.md)双方均零跌倒、8个路径点、零接触。90D MMD²由0.075233降至0.073257，45D由0.037231降至0.034705；速度W1由0.255966增至0.266753m/s，K100 coverage由0.31降至0.30。仅30秒且源码168→171，不能称严格单变量长期改善；默认模型不变。

[地形RMS独立核查](style_improvement_20261002/navigation_terrain_rms_review_v1/README.md)发现1024维地形方差受平地零输入稀释：std中位数为公开PACER1.528、基线0.240、两个mixed模型0.171；两个mixed的地形统计逐字节相同。已保存导航的每秒post-step重建中，全部地形像素clipping比例为1.147%／2.830%／0.727%。这些不是原runner保存的pre-action观察，也不能把共同统计变化当作两个mixed导航差异或停留的唯一原因。下一项训练消融需记录实际pre-action地形输入，并以显式新配置保护统计；不回写旧权重。


## 12. 原生室内视觉回放与四场景配置

已生成可观察行走动作的[Warehouse室内30秒视频](../output/diffusion_amp/native_warehouse_recorded_replay_indoor_30s_v1/pedestrian_motion.mp4)。[媒体/来源审核](style_improvement_20261002/native_recorded_replay_reader_v1/README.md)验证450 PNG、15fps／30秒、每帧来源时间标识及全部编码帧；源物理状态/结果与53份USD层SHA不变，physics与policy调用均0，时间线0→0。首／中／尾可见行人，未证明四人全程同时可见或动作自然度已经达到论文。镜头位于真实屋顶下方，没有隐藏屋顶、改动几何或修改保存姿势。外部镜头被屋顶遮挡的完整回放及两份早期失败均保留。

[四场景全量V9准入与配置](style_improvement_20261002/native_map_clock_v9_actual_reader_preparation/actual_four_scenes_v1/report.md)提供48份保守配置、2,058条通过CPU A*的路径连接。Office／Warehouse／Full Warehouse的3–15人配置通过；Hospital仅3–11人通过，12–15人的固定maximin布点间距不足1.6m而被拒绝。没有放宽出生点3m余量、路线0.8m余量、未知区阻挡或间距要求；拒绝不证明其他布点不存在。所有新配置的物理质量仍单独标记待测。当前15人Warehouse200秒使用旧已准入V5地图和此前固定配置，不借新地图改变该实验任务。

## 13. 原生 Warehouse：15人完整200秒实测

[1,325项独立全量复核](style_improvement_20261002/native15_200s_independent_review_v1/actual_report.md)通过。15名行人执行6,000个30Hz控制帧，零跌倒、零时间截断；211个路线点、49个同episode完整循环，人人至少完成一次。静态接触为0/90,000 actor控制帧，行人间接触为0/630,000无序pair控制帧，判定为四个物理子步中法向力最大值严格大于1N。保留360,000个actor子步和1,440,000份足部观测；COM重构最大差约1.91e−6m，质量加权COM约5.72e−6m。

此次使用原V5准入Warehouse、固定15人布点、导航基线step1000及Diffusion step5000；174份选定源代码和53份原始USD文件前后摘要一致。它验证本地单场景、单seed的200秒运行；没有证明作者八场景效果、动作风格超过论文或真实接触点滑移。

## 14. 策略执行前的真实输入及地形统计保护实验

[实际30秒观察与204项计量复核](style_improvement_20261002/pre_action_actual_observation_reader_v1/README.md)通过：900个控制帧、零跌倒／零static或peer接触。30个策略调用样本、120个行人采样帧中，轨迹前20维共2,400值无裁剪；地形后1,024维共122,880值，2,061值被裁剪（1.677246%），60/120行人采样帧含裁剪。真实未裁剪z最小约−48.49；30份样本的RMS逐位一致。物理姿势、参考规划和地面支撑全部数组与历史source168的同fixture30秒运行逐字节相同。

新观察器只调用原actor一次、返回同一结果对象；额外两次normalizer forward及CPU落盘不改变模型调用。真实物理数组一致属于这次30秒配对证据，不扩展为200秒或跨平台确定性。裁剪率也不能证明停留或跌倒的原因。

保护地形均值／方差已加入 `training/terrain_statistics.py`，配置显式选择 `task_terrain_statistics_mode=preserve_declared_suffix`。原有checkpoint默认legacy；只在新的完整公共PACER预热后绑定task[20:1044]，轨迹前20维和全局count保留原统计更新数学。原来源count61,931,520,001与预热count上限200,000分开保存，恢复／推理必须验证状态及SHA，不重新加载外部统计文件或替换已训练模型的moments。

合并后的165项CPU检查、Ruff及隔离旧源码52项CPU检查通过。新训练使用931份独立只读源码，921份原文件保持旧训练字节、五份旧文件仅应用此次统计策略delta；新的源码和导入账本与生产181文件快照分开。GPU4轮试训及4→8恢复均已完成，累计65,536次转换，两个完整checkpoint已封存。独立4轮检查1,108项、完整4＋4恢复检查1,150项均通过；恢复后的保护suffix逐位一致，Adam／replay／三个RNG及物理课程状态均核对。全新2,000轮／16.384M训练实际完成：每个课程分支8.192M转换，metrics最后更新时间3813.406910秒、完整result耗时3815.482118秒。实际训练累计510次跌倒reset、106,771次时间截断；这些不是部署导航指标。完整终态独立2,046项全部通过。

[预登记预算与验收](style_improvement_20261002/terrain_protected_experiment_registration_v1/registration.json) · [生产181文件快照](style_improvement_20261002/native_runtime_epoch_v9/native_pedestrian_source_v9.json)。

[真实4＋4恢复审计](style_improvement_20261002/protected_terrain_training_independent_review_v1/actual4to8_initial/audit.json) · [实际完整训练启动登记](../output/isolated_training/mixed_state_action_terrain_protected_v1/fullfresh2000_launch_registration.json)。

### 新训练：真实阶段检查点与蒙皮核验

第500轮已封存，4,096,000次物理转换、每个指令分支2,048,000次；[2,042项独立状态检查](style_improvement_20261002/protected_terrain_training_independent_review_v1/step500_milestone_admission_v1/report.md)全部通过。保护调用数500、task count4,296,000，公共地形suffix及原来源count不变。此为阶段状态准入，不是完整训练终态或导航效果验收。step1000也已封存并通过2,042项阶段检查；完整step2000终态另通过2,046项，阶段报告没有替代终态。

[真实4＋4恢复加强审计](style_improvement_20261002/protected_terrain_training_independent_review_v1/actual_pilot_report.md)共2,094项通过；GPU首段和恢复段训练跌倒分别1／3、时间截断124／153，不能相加或混用过程日志累计字段充当部署表现。

[完整蒙皮审计](style_improvement_20261002/recorded_smpl_skin_audit_v1/README.md)对实际900×4行人帧、86,400关节和24,804,000顶点执行CPU FK/LBS核对。SMPL关节与原生记录最大位置差0.23931mm、旋转差0.000703rad；120Hz末子步与30Hz捕获逐字节一致。现有回放已使用6890顶点纹理SMPL网格，中性体型／两张贴图和部分视角仍有限；蒙皮表面不等于碰撞表面，不能据此声称足部接触点无滑动或新模型步态更自然。

[有界真实专家／回放风格瓶颈诊断](style_improvement_20261002/amp_style_bottleneck_review_v1/README.md)记录action干预、姿势历史梯度和身体朝向后退的实测样本。结论只支持后续独立实验候选，没有改变本轮训练或固定2,848段数据。

### 实际完整训练与22条跟踪结果

[完整终态审计](style_improvement_20261002/protected_terrain_training_independent_review_v1/terminal2000/audit.json)通过；最终权重为[step_2000.pt](../output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_snapshots_v1/step_2000.pt)，SHA `e11a9a47d5e1147c4f894a519827d50539faf449a892c714d690957f1b267663`。独立931源码、实际87个owned导入、完整优化器／replay／三个RNG／课程状态和严格推理均核验；SDK最终清理标记仍为false，不冒充完整PhysX solver/cache重放。

[实际22条物理跟踪](../output/diffusion_amp/protected_tracking2000_v1/result.json)执行全部11,982个计量帧、15/22成功、零跌倒、位置0.247776m／旋转0.775494rad，用时299.633972秒。固定90D MMD²0.054528862、速度W1 0.05249288；旧同预算state-action的MMD²0.062529750／速度W1 0.05833213，但跟踪位置0.208382m／旋转0.725853rad更好。rootXY误差0.123692→0.171763m，heading局部姿态误差基本持平。风格分布指标改善与跟踪退化同时保留，不能据此宣称更自然或超过论文。四人200秒导航现已完成，完整结果如下；默认部署保持原step1000。

### 完整200秒导航与风格门槛

[完整CPU审查与比较](style_improvement_20261002/terrain_protected_navigation2000_complete_review_v1/README.md)：6,000控制帧、零跌倒／零时间截断，路线点[7,6,7,5]，共25个点与4个完整循环。静态／peer接触均零（法向力严格>1N）；1,153项原始计量一致性检查全部通过。停滞总93.8667人秒，比部署基线194.7667下降约51.8%，四人分别0.0667／25.7333／0.1667／67.9秒。

固定90D MMD²为0.09013555，劣于基线0.07238060；速度W1为0.32736323，好于0.35336799。新模型支持4个固定速度桶，基线5个，缺失桶保持未知，不填零。使用原b25dffa… AMASS-only固定中心的45D MMD²0.04140942、K50／100覆盖0.58／0.49；基线0.02015677、0.86／0.72。新模型虽然行走进展改善，但风格门失败，不晋升默认、不启动登记要求的原生follow-on，也不声称超越论文。

公共state-only AMP判别器先验的零扩展已合并为显式可选配置：保留2019输入和后续动作权重学习，公共AMP moments与D权重是耦合初始化改变，默认开关false。[真实合并CPU门禁](style_improvement_20261003/merged_action_state_prior_CPU_v1/README.md)通过44项新增与224项回归；[V11来源183文件](style_improvement_20261003/native_runtime_epoch_v11/native_pedestrian_source_v11.json)已冻结。新934文件隔离副本完成实际4轮／32,768转换，退出0并封存；完整状态审计、续训和完整风格评估尚待完成。数据2848与验证22／测试34保持不变，不声称效果提升。
