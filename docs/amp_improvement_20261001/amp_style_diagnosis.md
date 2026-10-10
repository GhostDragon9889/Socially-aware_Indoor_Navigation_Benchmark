# AMP步态诊断与显式改进建议

日期：2026-10-02。先只读论文、官方代码、现有训练日志和CPU checkpoint，再按父任务授权新增纯镜像工具。没有运行CUDA/Kit、修改既有训练器、权重、历史工件或主指南。新训练器/profile集成由父任务负责。

## 最关键的实测结论

**当前证据不支持“动作代理作弊是判别器分类差距的主要原因”。** 完成的独立课程step1000模型仍存在真实专家/策略状态分布差异，缺少旧策略replay和镜像正则等公开PACER机制。宜验证独立的新style profile，而不是把判别器概率提高直接称为步态修复。

### 1. 动作消融：影响很小，不能据名称认定作弊

使用不可变checkpoint SHA256 `e0d6ed4a9acb39c581bed92fab722a426bf23c34a3949bc48712cc13bf04c333`，在CPU加载真实2019维D及其归一化状态。专家为512个真实AMASS验证窗口，覆盖22个clip；物理数据为checkpoint内64个**实际保存的AMP历史窗口**。

| 输入 | 平均专家概率 |
|---|---:|
| 验证历史 + 匹配下一帧姿态动作代理 | 0.816364 |
| 同一验证历史 + 打乱动作 | 0.816242 |
| 同一验证历史 + 清零动作 | 0.816069 |
| 实际保存物理历史 + 清零动作 | 0.121885 |
| 同一物理历史 + 当前姿态动作代理 | 0.121573 |
| 同一物理历史 + 随机专家动作 | 0.121612 |

`AMPWorld.expert_batch`把下一帧mocap关节角除以实际PD尺度，截断至[-1,1]；它没有恢复人体控制动作、力矩或逆动力学。这仍是Nav式D(h,a)复现的语义假设，但本次动作替换的概率变化仅约0.0003，而状态分类差距约0.69。不能将后者归因于动作代理。

D的57个关节特征已排除手和脚趾；冻结动作坐标的局部梯度RMS约`9.4e-7`，没有观测到它们成为主要分类来源。64个历史窗口不等于200秒全量采样；旧NPZ缺少精确完整AMP输入和实际动作，未伪造全运行分类结果。[CPU原始证据](amp_style_cpu_diagnostic.json)。

### 2. 判别器分类高，不代表充分检查时序

同一验证批把当前帧重复10次，分类概率仍为0.814239，接近正常历史的0.816364；反转时间顺序为0.784467。该消融暗示当前D对本批时序一致性的约束较弱，不能推断所有状态都如此。重复帧仍保留mocap记录的速度，因此这个输入并非真实静止运动。

真实物理快照的关节速度RMS为1.713rad/s，验证mocap为1.046；根线速度分量RMS为0.988m/s和0.575。它们可能来自任务/目标速度分布、运动风格或动力学差异。各特征组归一化夹断比例均小于0.5%；尚无证据证明一个严重的夹断、传感器或关节速度接口错误。D不包含接触标记或根高度，不能声称它直接利用了这些字段作弊。

### 3. Replay和归一化：已有具体证据

审查的旧训练器每次D更新使用本轮策略与专家平衡采样；**类别本身没有不平衡bug**。但物理rollout仅2048条、PPO minibatch16384，6个epoch会6次重用同一完整批次；D更新次数依附于PPO minibatch，没有跨迭代旧策略replay。

原始AMP明确用历史策略replay避免D只适应最新一批；公开PACER也组合当前策略和replay，配置200000容量、0.01保留概率。[AMP原论文§6](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)，[PACER官方训练器](https://github.com/nv-tlabs/pacer/blob/main/pacer/learning/amp_continuous.py)。旧代码证据为归档版`training/trainer.py::_discriminator_update`及`update`，原源码pin见[诊断JSON](amp_style_diagnosis.json)。

旧模型继承的actor观测计数约`6.193e10`，value计数`6.883e9`；新2.048M条actor样本只占约`3.3e-5`，统计量近似冻结。这是保留原权重统计的显式选择，不是自动断言一个bug。新profile可保留初始mean/variance并显式截断先验计数，但这会改变后续适应行为，必须作为新实验保存，不能静默改恢复中的旧checkpoint。

### 4. PPO与训练量

独立课程1000更新共2.048M真实transition。末100次平均：D专家概率0.861683、策略0.123090；AMP奖励0.289495、轨迹奖励0.977526，均为外层权重前的值。第一轮PPO KL为0.04756、裁剪比例0.432；末100轮KL约0.01597、裁剪比例0.1987。

公开PACER配置本来就是LR`2e-5`、6个epoch，故不能仅凭这两个参数称原训练错误。实测KL支持试验可选`target_kl=.02`，记录实际更新、覆盖率和提前结束理由；D更新应与PPO提前结束解耦。2.048M transition本身也不足以推出风格已收敛。

## 论文/公开实现应分开解释

- 原始AMP采用状态转移输入、正负目标±1的最小二乘D，以及有界二次style奖励。它的损失不是当前BCE/softplus。[AMP Eq6–8](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)。
- PACER论文有D(h,a)记号，并说明10帧窗口可能出现跛行，因此新增镜像策略损失。公开no_shape代码实际D为1950维state-only，并用BCE/softplus。[PACER附录B](https://arxiv.org/html/2304.01893v1)，[官方代码](https://github.com/nv-tlabs/pacer)。当前复现应明确采用的是哪一个合同。
- PHC同样采用本体状态历史style prior与任务奖励；其tracking/recovery目标不是当前单纯导航目标，不能无条件复用整套终止条件。[PHC原论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Luo_Perpetual_Humanoid_Control_for_Real-time_Simulated_Avatars_ICCV_2023_paper.pdf)，[官方AMP训练器](https://raw.githubusercontent.com/ZhengyiLuo/PHC/master/phc/learning/amp_agent.py)。原PDF搜索索引可读取，本次直接打开返回403，代码读取成功。
- ASE有可复用的潜变量技能结构；其官方AMP模块可参考replay，以及混合确定/随机rollout避免D只区分噪声的设计理由。将技能潜变量直接加入当前actor是新架构，需要独立训练，而不是checkpoint兼容修复。[ASE官方项目](https://xbpeng.github.io/projects/ASE/index.html)，[官方AMP代码](https://raw.githubusercontent.com/nv-tlabs/ASE/main/ase/learning/amp_agent.py)。

## 建议给父任务的新profile接口

保留旧Nav Eq2 D(h,a)与字面奖励默认行为；命名的新style profile采用state-only公开PACER-D、逐history-coordinate moments，并明确记录该方法变化。

| 参数 | 旧默认 | 建议实验 |
|---|---:|---:|
| policy replay容量 | 0 | 65536 |
| replay负样本比例 / 满后保留概率 | 无 | 0.5 / 0.01 |
| 独立D更新次数 | 跟随PPO | 2 |
| warm-start moments计数上限 | 无 | 200000 |
| AMP / task外层权重 | 1 / 0.5 | 2 / 0.5 |
| effective physical likelihood | 关闭 | 开启 |
| KL限制 | 无 | 0.02 |
| 镜像MSE系数 | 0 | 50，实际57有效动作归一化 |

这些数值是有出处和诊断依据的试验设计，尚无本代理的GPU/物理质量验证。65536×1950×float32历史约488MiB，应记录内存、replay真实采样比例、cursor/RNG与完整恢复。缓存原始历史，采样时使用当轮冻结moments；分类损失仍平衡专家与全部负样本。似然、entropy、bounds、KL应一致忽略12个不生效坐标，同时保持69维实际物理投影不变。

增大style权重可能牺牲路径进展；应同时验证实际速度、追踪、跌倒、接触、连续路线完成和held-out姿态统计，不能只看D奖励。旧选定319/22/34个clip全部30Hz；扩展银行存在50Hz数据，必须先按实际dt转换为30Hz窗口，不能误称旧语料已经混用50Hz。

## 已实现的镜像工具及验证

新增`CrowdSim/humanoid/training/symmetry.py`，提供`mirror_policy_observations(rawself357, rawtask1044)`与`mirror_actions(raw69)`；必须在归一化前应用。位置/线速度Y翻转，角速度与exp-map动作X/Z翻转；左右body/DOF按名字交换；切线X/法线Z的6D旋转遵守`SRS`，32×32 ij地形翻转Y列。

20项CPU测试通过，Ruff通过。测试用独立反射矩阵对带倾斜/朝向的状态、非线性地形重建实际self/task观测；动作与SO(3)矩阵对照；只执行官方源码中纯任务镜像和动作损失方法，核对值及两个分支的梯度。还验证反射两次恢复、输入不被修改、batch/device/finite/名称限制。

官方`_sym_loss`**两侧都有梯度**，MSE乘50，并没有detached target。父任务接受两侧可微，按57个实际生效动作而非官方69维归一化，属显式物理动作适配。中性SMPL左右骨架并不精确对称：镜像正则不是物理等变保证。

256个真实AMASS验证状态的静态actor CPU对照：公开PACER有效57维镜像MSE为0.00020223，旧独立课程step1000为0.00132097，约6.53倍。使用各自moments、显式平地和当前速度线性目标，非未来oracle、非物理rollout。它支持试验镜像正则，不能直接证明跛行改善。[残差证据](amp_symmetry_cpu_residual.json)，[测试结果](amp_motion_symmetry_tests.xml)。
