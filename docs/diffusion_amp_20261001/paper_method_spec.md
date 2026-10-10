# NavIsaacLab Diffusion+AMP：公开规范与可训练重建

原论文第6–7页已渲染并视觉核对，完整规范/源码哈希见 [paper_method_spec.json](paper_method_spec.json)。本文的论文指标属于作者报告，不是当前训练结果。

## 公开方法

IV-A：root平面状态 `[x,y,u,v,theta]`，自身轨迹历史H、邻居轨迹N、占据地图M作为条件；固定噪声schedule的高斯reverse process；temporal U-Net与map卷积编码、down/up与skip connections；预测clean `tau0`，MSE监督，未来5秒轨迹供低层控制。

IV-B：SMPL articulated humanoid、goal-conditioned PPO，self state、障碍特征、2D目标轨迹进入MLP actor/critic；AMP判别器也是MLP。当前资产为24body、69受控旋转DOF（23组三轴；pelvis是free base）。69DOF是项目资产合同，不能混写成论文列出了69个关节。

| 公开公式 | 当前实现 |
|---|---|
| `r = r_amp + 0.5*r_task` | `paper_total_reward`，未继承公开PACER配置的0.5AMP权重 |
| `r_task = r_tau+r_e+r_d` | 三项独立可微Torch函数/指标 |
| `r_tau = exp(-2*delta)` | delta由明确对齐的当前XY与目标XY平方差求和 |
| `r_e = -0.0005*sum(torque_i*omega_i)` | 每个真实 PhysX 子步的投影关节广义力与同子步角速度，四子步平均有符号功率；该通道不是分离的纯电机力矩 |
| `r_d = 1-exp(min(0,u)/1.8)` | 默认 `paper_equation` 原式；显式 `backwards_penalty` 才反转该项 |
| `r_amp = D(h[t-10:t],a_t)` | 独立训练history+action判别器；公开PACER的state-only D没有迁入 |

**公式歧义不会被隐藏。** 印出的方向等式在u<0时给正奖励，和“防止向后走”的文字矛盾。有符号能耗也会奖励负机械功。训练记录各分量、signed power和backwards fraction。反走惩罚模式是不同实验，不能声称改了符号仍严格使用印出等式。

## 实际训练实现

`CrowdSim/humanoid/training/`包含真实PPO clipped-surrogate、GAE、critic、共享task encoder、action-conditioned AMP分类/专家gradient penalty、两个Adam optimizer，以及完整训练状态保存/恢复。它接收真实物理runner的rollout，不创建假的episode或checkpoint。

公开PACER架构作为明确的重建选择：task1044→512→256；self357+encoded256→actor/critic2048→1024；actor69、critic1；固定logstd−2.9；新D输入为10×195+69=2019→1024→512→1。论文只公开MLP/PPO，未给这些宽度与超参数。具体参数源为`third_party/pacer/pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml`。

`warm_start_pacer`先用已校验loader验证公开no_shape checkpoint的SHA256和完整schema，再复制全部actor/critic/task/sigma及obs/value统计。新D和AMP frame统计保持独立初始化。初始化来源在训练checkpoint provenance中记录；这不是NavIsaacLab作者AMP权重。可以不调用warmstart，从零实际训练。

## 数据与严格模式

三个profile分开保存：

- `configs/diffusion_amp/method_reconstruction.yaml`：允许真实用户供应/筛选数据，明确假设，默认迁移初始化，实际训练用。
- `configs/diffusion_amp/strict_author_experiment.yaml`：作者精确2848清单、八场景protocol及完整训练配置缺失时明确拒绝。
- `configs/diffusion_amp/reference_weights.yaml`：公开PACER/TRACE资源身份，不冒充论文定制checkpoint。

Root已从真实AMASS包筛选319train、22validation、34test：文件名动作标签与upright/height/speed门槛见`data/diffusion_amp/amass_manifest.json`。这是375条有明确来源的重建语料，不强凑2848，不是原9条Walking，也不是作者精确清单。当前完整性gate已核对3个包的尺寸、tensor24body/69DOF、每个ID/source/帧数/dt/有限首帧和source跨split泄漏；原大包SHA沿用manifest声明，未再次哈希9GB。证据：[manifest gate](paper_reconstruction_manifest_gate.json)。

专家action采用下一帧mocap关节位置映射为normalized PD targets的代理。AMASS不含训练policy commands或实测专家力矩，代理不等于inverse dynamics。该假设进入profile/manifest；runtime实际能耗reward不能用这个代理代替。旧首轮误用了子步前的 implicit PD 估算与子步后速度；修正代码与重新训练独立保存，详见 [力矩语义与时序审计](torque_semantics_audit.md)。

## TableII与还未确定的参数

论文评估8场景，每场3–15行人、200s，SFM+AMP和Diffusion+AMP保留同一个AMP controller。

| 作者报告 | 成功% | 跌倒% | 地图碰撞% | 行人碰撞% | 路径误差m |
|---|---:|---:|---:|---:|---:|
| SFM+AMP |74.24|4.99|3.74|17.17|0.189|
| Diffusion+AMP |88.67|1.12|2.40|7.81|0.153|

另外作者给出body位置平均误差0.218m、旋转0.343rad、最大body位置0.433m；每步mean body position error超过0.5m即失败，clip成功率64.6%。这不等于本项目的pelvis高度失败检测，应分开评估。

仍未公开/无法据论文唯一确定：D输出到reward的变换与loss、10还是11帧及排列、专家action定义、delta的时间/多点reduction、PPO/network/normalizer细节、diffusion schedule/input transforms/guidance、作者2848截取/权重/预处理、精确八场景/seed/episode数和成功碰撞百分比的定义与聚合。此前公开分支/补充网页/基线研究已保存在`docs/resource_completion_20261001/paper_publication_sources.*`，本任务没有重复下载，也不能从有范围的搜索推断作者资源绝对不存在。

## 验证范围

最终29项CPU测试通过（2.31s），全部本轮Python源码与测试Ruff PASS。随后发现并修复hand/toe冻结坐标的判别器shortcut：新D必须看`clamp(raw)*declared_action_mask`，和expert代理及真实PD解码一致；raw69Gaussian仍保留作PPO likelihood。补充回归验证物理PDtargets完全不变、冻结坐标policy/expert同为0、显式mask不能重开物理冻结关节以及checkpoint恢复mask。另验证先构造其他网络再调用`restore_rng()`，仍能恢复完整checkpoint RNG序列；父runner先`load(restore_rng=False)`，构造planner并恢复物理状态后再调用，避免初始化消耗保存的训练随机流。最终测试数与SHA见JSON/JUnit。

CPU测试中的算法夹具不是仿真采集结果。本代理未启动Kit；父代理负责实际PhysX训练与Diffusion计划集成。真实训练完成、是否收敛及持续物理稳定性必须以父代理训练日志/模型/heldout评价为准，不能用以上测试代替。

当前四份 DynBench CPU 投影地图及两项缺失场景依赖已补齐；最新证据见 [完整场景地图报告](../maps_projection_resource_complete_inventory.md)。真实 PhysX 地图复核和作者八场景协议仍未完成。原 [场景与地图清单](paper_scene_resource_inventory.md) 保留为补齐前的审计记录。
