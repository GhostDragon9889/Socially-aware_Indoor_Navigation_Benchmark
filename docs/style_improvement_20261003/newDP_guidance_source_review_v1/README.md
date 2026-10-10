# 新DP：guidance源码与完整物理结果核查

2026-10-03。只读本地／官方源码和已经闭合的结果；未运行GPU、SDK或模型，未修改185源码、默认权重、现有评估协议或历史记录。

## 1. 当前已经采用TRACE的clean guidance机制

本地公开TRACE checkout：`594b2af0824c5853e1a1884e167cd5ed097dd81c`。官方[`p_sample`](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/models/trace.py#L635)的`guide_clean=True`在干净预测上计算损失，经过denoiser对带噪输入求导，修改干净预测，再用同一带噪输入重算后验。官方公开代码省略额外的方差缩放，并注明这是经验选择。[TRACE论文](https://arxiv.org/html/2304.01893v1)Eq.6则写有Σ缩放。

| 本地模式 | 实际计算 | 与官方clean路径的关系 |
| --- | --- | --- |
| `legacy_posterior` | posterior mean上稀疏归一化XY平方误差梯度，乘posterior variance后平移mean | 没有denoiser输入Jacobian；不是官方clean方法 |
| `trace_clean` | 物理单位clean轨迹损失→对noisy输入求导→修改clean→重算posterior | 当前使用；上述机制与官方公开实现对应 |
| `trace_clean_variance` | 与上一行相同，再给全部目标梯度乘本地posterior variance | 现成的方差缩放消融；不等同于官方公开默认 |

本地实现见[`paper.py`](../../../CrowdSim/humanoid/diffusion/paper.py)，主要在231–285、629–655行。它扩散五维state；官方TRACE通过扩散动作及unicycle dynamics得到轨迹，且只取动作维度梯度。本地目标按每条轨迹的waypoint均值、跨actor/sample求和；官方[`compute_guidance_loss`](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/utils/guidance_loss.py#L889)对loss取均值。本地root圆盘碰撞代理也不同于作者body extent／多个圆盘／地图body采样点。这些差异意味着同名权重的数值不能直接当作者超参数。

## 2. driver、地图与history的真实路径

- 通用训练parser默认`legacy_posterior`，但实际[`deployment.driver_arguments`](../../../CrowdSim/humanoid/application/deployment.py)显式使用`trace_clean`、goal20、peer5／半径0.6m、scene10／半径0.8m，候选1、eta0、15控制步＝0.5秒同步重规划。`scripts/train_paper_amp.py`的`replan()`调用`PaperDiffusionPlanner`，不是另一个`diffusion/planner.py`包装器。
- 学习地图上下文：`PaperTrajectoryDiffusion.encode_context/forward`将occupancy raster编码成学习特征，在轨迹位置查询这些特征。显式地图guidance另走几何cost：fixture采用声明的真实静态AABB与domain；native采用[occupancy SDF callback](../../../CrowdSim/humanoid/runtime/native_grid_scene.py)，不是学习feature的标量cost。原始地图是当前静态场景，不包含未来真值轨迹。
- [`AMPWorld.planner_inputs`](../../../CrowdSim/humanoid/amp_world.py)使用真实执行状态history的30→10Hz下采样和真实邻人history。未来5秒有10个timed-path路点（0.5…5秒）；执行reference有1.5m/s／2m/s²的causal adapter。planner的speed／accel penalties实际为0，不能把其声明阈值当硬限幅。
- 官方MapCollisionLoss使用`drivable_map`及body采样点，也没有把学习feature直接作为显式地图cost。[Nav论文](https://arxiv.org/html/2606.26265v1)只披露TRACE式上下文和未来5秒→local targets，未披露上述guide20、半径、候选数、15步、adapter等具体值。TRACE-PACE物理实验描述跟踪2秒后以实际运动history重规划；本地0.5秒不能称作者精确设置。goal guidance也不是classifier-free sampling强度，本地没有启用作者的CFG采样分支。

## 3. 开环改善与闭环退化的证据边界

新旧100场景[真实配对评价](../orca_maps_diffusion_actual_runs_v1/README.md)调用[`planner.plan`](../orca_maps_diffusion_paired_evaluation_v1/evaluate_paired.py)时没有waypoints或碰撞cost，**guidance未启用**；记录中的默认`legacy_posterior`标签不参与计算。其ego开环预测以记录的专家history／邻人及静态地图为条件。ADE 1.132818→0.852463m、FDE 2.636335→1.879224m改善，但最坏障碍穿透仍退化。它不能推导多行人guided闭环／身体物理成功。

完整相同200秒／6000控制帧的[现有物理结果](../default_NEW_samples10_actual_runs_v1/README.md)：

| 同default AMP62c4、同185、guide20、固定路线 | 跌倒事件 | static actor-controls | peer pair-controls | 路线顶点 | idle有效区间比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLD63e，S1 | 0 | 0 | 0 | 21 | 24.3499% |
| NEWb563，S1 | 1 | 0 | 70 | 9 | 30.0354% |
| NEWb563，S10 | **3**（2名行人） | 0 | 279 | 18 | 19.5390% |

S10改善部分进展和停滞，却增加跌倒与peer接触，不能作为新默认或native准入。NEW S1行人1最长idle28.7667秒，其raw5秒向goal投影均值−0.4116m；S10同一行人最长idle3.8秒、投影+5.6856m。S10行人2最长idle仍有负5秒投影−0.1393m。近0.5秒计划与身体执行量也被完整保留：远端投影变化不能证明跌倒机制。NEW S1的唯一跌倒与已有peer/static事件不重合，[原事件诊断](../default_newDP_failed_event_diagnostics_v1/README.md)保留，不归因于已测碰撞。

其他已完成物理证据：[早期legacy→clean](../../amp_improvement_20261001/clean_5000_physical_comparison.md)同时改变DP训练步数、权重、碰撞cost和运行路径，16→11次跌倒不能隔离mode效果；后续同模型[goal20→40](../../style_improvement_20261002/navigation_guidance40_comparison_v1/comparison.md)路线21→29、跌倒0→1、peer pair-controls0→13；[eta0→0.25](../../style_improvement_20261002/near_horizon_eta025_style_v1/report.md)路线21→20、跌倒0→2。所审阅的真实物理记录未包含`trace_clean_variance`，也没有证明扩大引导／候选能统一提升质量。

## 4. 优先一个物理消融，尚未运行

固定NEWb563＋defaultAMP62c4＋S1，只将`--diffusion-guidance-mode trace_clean`改为`trace_clean_variance`。guide20、peer5、scene10及半径、15步、reference adapter、四人200秒、seed7、alpha0、场景／路线／物理全部保持。模式同时缩放现有三个目标的梯度，保留它们的相对权重；不改训练或重新选择离线排名。可由独立薄部署argv入口显式覆写，生产185无需修改。

待检验假设：新denoiser在无额外方差缩放的引导下，timestep梯度幅度对身体执行更敏感。当前证据只支持提出这个假设；官方公开代码曾认为方差缩放不利，不能预期本地一定改善，也不能称严格作者复现。

直接保存完整6000帧，比较跌倒／静态／peer接触、每人路线／idle、已保存同相位0.5秒与5秒投影及reference响应，保留全部失败。原零跌倒／零静态／零peer准入不变，MMD不用于筛选启动或删帧。若物理失败仍在，或进展／停滞进一步恶化，则本轮不能支持该候选；即使此单seed fixture改善，也不表示跨场景成功或默认可迁移。未新增模型实验、reader／freeze框架或基准重跑。
