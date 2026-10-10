# 独立 pose-only 45D 多样性候选诊断

**新增 CPU 诊断，不替换原 90D 主评分，不用于推广新模型。** 依据 [NavIsaacLab IV-C / TableIII](https://arxiv.org/html/2606.26265v1#S4.SS3) 的姿态多样性评估思路：采用统一关键点并移除根位置、朝向和尺度，再统计 AMASS 姿态模式覆盖及分布距离。作者完整关键点、尺度、聚类设置、取样及 MMD 带宽未公开；这里所有参数明确，**不能直接比较 TableIII 数值**。

## 固定实现

直接读取冻结 `sequence_style_samples.local_pose`：15 个关键点 × XYZ = 45D；不含旧主特征的速度 45D。Feet 明确映射 L_Ankle/R_Ankle，另含 Neck/Head；原固定 per-actor head-chain + mean-leg-chain 尺度保持不变。引用 c274 同一 validation selection，实际选中数组 SHA 精确复核，所有训练 manifest 的父源与完整 heldout 父源保持不相交。

仅用固定 AMASS 15360 个合法间隔端点中的等距 2048 个样本拟合 K50/K100。明文 NumPy kmeans++ / Lloyd，seed7、n_init10、最高300迭代、绝对中心移动阈值1e-4；选择 reference SSE 最小的启动。生成样本从不参与 fit，不重新调权或标准化。全生成合法间隔端点赋类；coverage=occupied/K，H=-Σp log(p)/log(K)，Dist 为最近中心平均欧氏距离。Pose-only MMD² 为 Gaussian biased estimator，σ=1，每侧最多2048个等距 inclusive 样本。

| 同场景模型 | 合法 pose 端点 | K50 Coverage / H / Dist | K100 Coverage / H / Dist | Pose-only MMD² |
|---|---:|---|---|---:|
| selected guidance20 | 23996 | .86 / .666210 / .271772 | .72 / .636018 / .269437 | .02015677 |
| mixed state-only2000 | 23985 | .70 / .683962 / .327361 | .67 / .690073 / .323010 | .05382186 |
| mixed state-action2000 | 23994 | .68 / .442133 / .229038 | .65 / .389542 / .218004 | .04759910 |

每次原始数据均为6000帧×4=24000 actor帧；复用 helper 会排除4个首帧，以及分别0/11/2个跨 reset 端点。**不排除摔倒、低身、停留或其它异常有限姿态，但并非全部 post-step 帧都进入 pose 样本。** Coverage/entropy 全部合法端点使用；MMD cap 不保证每个失败姿态都进入最终子样本。Reference-self MMD 为0是同样本校准，不是独立 reference holdout 得分。

State-action 条件下 H 明显较低，与其73.985%停留同时出现；其更小 center distance 并非更强动作多样性。该指标忽略时序、速度、接触和导航进展，不能将较小 pose-only MMD 或距离等同完整步态改善。既有速度/失败 inclusive90D 主指标及默认 selected 0F/21顶点结论保持。

## 单列不同场景

Native Warehouse4（actual source168）只作为 **不同场景上下文案例**：同 selected checkpoint、0F/57顶点，但原生场景/路线/部署源码不同，不能与旧 cuboid 行作为方法因果对照。23996个合法端点：K50 coverage .34/H .575606/Dist .155709；K100 .31/.567596/.155255；pose-only MMD² .03829664。较小距离同时可能对应较窄的姿态模式，不能据此断言缺乏物理稳定性。

[原生 source/admission](actual_evaluation_v1/native_case_admission.json) 只绑定已有983项实际审核及 source168 归档；没有推断当前171源码等于历史运行。

## 验证与产物

18项 CPU 测试通过，覆盖已知中心均值、重复参考的 kmeans++、空簇真实样本迁移、平距 tie、最小SSE启动选择、确定性、生成数据不污染参考中心、MMD对称和非法有限性/schema；Ruff通过。初始 sklearn 缺依赖失败保留；最终仅依赖现有 NumPy/Torch，不安装或升级 SDK 环境，不启动 Kit/GPU，不读取34个 test capture。

[完整报告](actual_evaluation_v1/evaluation.json)、[固定参考中心](actual_evaluation_v1/reference_basis.npz)、[复跑命令](commands.json)、[测试](test_pose_diversity.py)。历史主报告字典、特征、核、阈值、模型、资源与训练均未修改。
