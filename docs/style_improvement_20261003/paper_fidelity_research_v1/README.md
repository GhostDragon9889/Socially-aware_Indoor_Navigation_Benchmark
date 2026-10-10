# 下一轮论文依据与本地场景扩展

2026-10-03，只读研究及 CPU 文件核对。没有修改生产代码、历史冻结配置、默认模型，没有启动 GPU，没有读取封存的34条测试输出。新增论文和作者代码副本仅用于研究；授权条款仍以原作者仓库为准。

## 1. 下一轮只改变 AMP 外部权重

建议从已验证的公共姿态先验＋动作判别器、地形统计保护配方重新训练：`training.amp_reward_weight: 2.0 → 0.5`。其对应声明 `reward.amp_weight`同步为0.5。其他训练字段、2848动作、22条验证、34条测试来源、课程、预算、seed保持相同。这是一个奖励比例变量，而不是两处独立机制。

令 `b(z)=min(softplus(z), -log(1e-4))`。当前路径为：

| 路径 | 判别器内部尺度 | AMP 外部权重 | task 外部权重 | 对 b(z) 的有效系数 |
|---|---:|---:|---:|---:|
| 当前已训练候选 | 2 | 2 | 0.5 | 4 |
| 下一轮建议 | 2 | 0.5 | 0.5 | 1 |
| 公开 PACER 作者配方 | 2 | 0.5 | 0.5 | 1 |

作者[固定 commit 配置](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml#L93)给出尺度2及外部0.5/0.5。[作者 `_calc_disc_rewards` 与 `_combine_rewards`](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/learning/amp_continuous.py#L599)显示先做负对数奖励和尺度，再乘外部权重。

本地实际路径：`CrowdSim/humanoid/training/trainer.py:311`、`training/rewards.py:66`、`scripts/train_paper_amp.py:782`。没有增加新接口，也不需要转换旧 checkpoint。应 fresh 初始化，不能将改变奖励合同的模型冒充原 run 的继续恢复。

**精度边界：**[NavIsaacLab IV-B](https://arxiv.org/html/2606.26265v1#S4.SS2)给出 `rAMP + 0.5 rtask`，但没有定义判别器输出到 rAMP 的概率变换或内部尺度。所以不能宣称其论文奖励存在“重复乘2 bug”；这里恢复的是**公开 PACER 的数值奖励比例**，不证明等同 NavIsaacLab 私有训练实现。

**预期是待验证的假设：**减少动作分布奖励相对任务奖励的主导程度，可能改善路线进展；也可能损害动作风格。完整2000行训练日志最后100行：内部 AMP 均值0.36882016，原外部贡献0.73764032；原任务外部贡献约0.42636432。见[只读汇总](training_reward_component_readonly_summary.json)。高奖励、低loss不是质量验收。

## 2. 当前主要方法差异

| 主题 | 论文／作者依据 | 当前代码及结论 |
|---|---|---|
| D 的输入 | Nav式(2)及 TRACE/PACER正文写历史＋action；公开 PACER运行实现的示例 D 是历史状态 | 当前2019输入 =10×195＋69 action；公共1950状态权重扩展为0初始 action 列。AMASS没有实测控制动作，expert为下一帧PD目标代理，不能称作者逆动力学标签。 |
| 10帧与镜像 | TRACE/PACER补充B.2讨论10帧局部窗口出现跛行，式(22)直接约束镜像动作 | 当前镜像系数50，匹配作者 `_sym_loss`；57有效动作平均而作者69均值，是明确工程差异。此轮不要同时改镜像。 |
| 能量 | Nav式(5)是 signed torque×velocity；PACER补充B.2及作者 terrain 实现使用 absolute power | 当前遵循 signed 同步物理子步功率；不是作者PACER完全一致。此轮保留，避免混合变量。 |
| 后退方向 | Nav式(6)打印值对负前向速度为正，与“防止后退”描述有冲突 | 当前显式 `backwards_penalty`取相反号，原literal模式保留；是已声明修正。 |
| 动作采样 | Nav只说明按导航标签选择2848条；没有公开ID清单及采样分布 | 当前uniform clip→合法uniform frame。`expert_sampling=null`。现成stratified sampler会改变expert prior，且不应用重要性校正；作为之后独立实验。 |
| 运动条件 | TRACE/PACER补充B.3有procedural速度0–3 m/s、加速度0–2 m/s²、约200手选动作 | 当前固定半AMASS根轨迹、半procedural 0–2/1.5，是额外课程；2848是真实本地AMASS片段，但不是作者2848 IDs。 |

TRACE/PACER作者[正文及补充PDF](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf)本地下载21页，正文§3.2及补充B.2/B.3（印刷页16–17）。[下载来源与 SHA](fetched_sources_actual.json)、[commit来源](pacer_commit.json)、[逐文件固定commit比对](pacer_commit_binding.json)完整保留；3个main下载文件均与 `baeb6241f03fc9d78370a64e3c2dcef9d830d97f`逐字节一致。

## 3. 风格覆盖的研究结论

[AMP §5.2式(6)/(7)与§7局限](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)使用least-squares判别器／有界奖励，并承认多样数据的mode collapse。当前BCE/softplus是公开PACER路线；修改为LSGAN需要新奖励合同，此轮不混入。

[ASE §6.4式(15)、§10.4](https://arxiv.org/html/2205.01906)用技能latent、发现目标和多样性约束改善覆盖。当前无latent输入，不能只加一个“多样性loss”就称ASE。[ADD §5及§6.1](https://arxiv.org/html/2505.04961v1#S5)需要参考姿态差分及目标姿态条件，以精确跟踪代替整体分布近似。部署只有2D扩散轨迹，所以直接接ADD需要额外pose生成；不能偷偷向当前政策传参考未来姿态并称原论文复现。

下一轮验收应同时保留完整22条失败帧、腕／肘误差、同场景共同速度桶90D、固定45D覆盖、真实路线进展、跌倒与接触。原八场景/指标参数未公开时，不能把本地MMD数值与论文Table III直接排名。

## 4. 八个明确的本地评估上下文

[CPU核对清单](eight_local_contexts_proposal.json)绑定现成配置、路线证明、USD和V9原生地图SHA：

| 本地 USD 场景 | 稀疏人数 | 较密人数 | 每个计划时长 |
|---|---:|---:|---:|
| Warehouse | 3 | 15 | 200s |
| Office | 3 | 15 | 200s |
| Full_Warehouse | 3 | 15 | 200s |
| Hospital | 3 | 11 | 200s |

这只是**四个不同USD、八个场景密度上下文**，不是八个不同场景，更不是作者的八场景。现有52候选中48组通过CPU出生点＋A*，Hospital12–15因最小间距1.6m失败，不应回退填充。Warehouse旧V5四／15人实测结果也不能自动转移到此V9上下文。

第二个实际场景优先 **Office3人**：地图测得223,776个free cells全部有射线支撑；现成0.8m净空路线有CPU证明。源地图只做2步snapshot、gravity=0，且 `human_navigation_validated=false`、`native_cooking_verified=false`。所以须用新epoch的CPU预检，然后在独占GPU下短时真实重力／SMPL足部支撑、地形、静态及行人接触检查，成功后再跑6000控制步。不要用射线地图证明替代真实人形物理。当前admitted catalog没有Factory。

`/project/EXP/NavIsaaclab2.0/CrowdSim/world/map.py`按净空连通域和seed采样start/goal并A*；`scene_setup.py`的PNG出生点不足时有fallback grid，不能用于此严格本地协议。其Office YAML仍引用 `/data/...`，必须解析到已审计本地资产和地图。该仓库的humanoid诊断是SFM＋MaskedMimic，不能直接作为作者私有Diffusion＋AMP训练代码。

完整扩展应在同一模型、身体、route、seed上比较 Diffusion＋AMP 与 SFM＋同AMP，公开所有失败/未知支持桶。现有资源限制仍不足以声称达到了论文完整场景效果。
