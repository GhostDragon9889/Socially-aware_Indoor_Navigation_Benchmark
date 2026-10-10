# 地形 RMS 保护：限定集成提案

**状态：仅 docs 候选，尚未修改生产174、部署、既有模型或训练。** 用户已授权修正与真实训练；本提案等待 root/deps 读审及运行中任务结束后实施，不新增用户审批环节。现有16项原型测试及真实公开统计逻辑检查是 CPU 组件证据；最终新增计数门禁的测试结果见 [source_freeze.json](source_freeze.json)。

## 实际依据与实验边界

既有 [地形统计诊断](../navigation_terrain_rms_review_v1/README.md) 中，公开地形标准差中位数1.527620，negative1000为0.239732，mixed/state-action2000均为0.170591；后两者地形 mean/var **逐位相同**。平面1024维持续零观测、暖启动计数上限200000、最终共享计数16584000可精确解释这一稀释数学过程。保存姿态重构的地形观测并非实际 pre-action输入；仅凭这些统计不能把空闲或两个模型的不同导航效果归因于 RMS。

state-action2000实际末次更新 discriminator expert/policy概率0.80684/0.14572，live/replay为0.14333/0.14812，AMP均值0.32897、轨迹奖励0.98109。训练—专家判别差仍存在，末次 replay/live接近，不能据此宣称 replay 引起空闲。专家动作仍是 next-frame mocap姿态的归一化PD目标代理，不是测量控制或逆动力学；本实验不改变此假设，也不同时改专家采样、replay、奖励、D输入或命令课程。

论文使用运动先验与任务指令，但没有公开这一统计保护规则；这是工程消融，不是作者方法的精确实现。[NavIsaacLab IV-B/C](https://arxiv.org/html/2606.26265v1)、[AMP 原论文](https://arxiv.org/abs/2104.02180)。主验收沿用冻结参考与既有失败帧全包含的风格/物理协议，不与论文TableIII数值直接等同。

## 唯一训练策略变化

新增纯 stdlib `TrainingConfig.task_terrain_statistics_mode: str = "legacy"`，仅允许 `legacy` / `preserve_declared_suffix`，拒绝 bool、空值和未知模式。opt-in要求 canonical357/1044、normalize_observations=True、clip5，以及实际完整公开 PACER actor/critic/task暖启动。

保护 task[20:1044] 的1024维地形 mean/variance；task[0:20]轨迹统计与**共享 task.count**仍逐位执行原 `RunningNormalizer.update` 数学。更新顺序不变：PPO/D使用 rollout冻结统计 → 原 self/task/AMP统计更新 → 仅在 opt-in复制保存的地形 mean/var → 原 update_count/physical_transition_count递增。保护不移动统计更新到PPO重算之前，不修改count、self/AMP/value、clip/epsilon、模型、优化器、动作mask、PD或RNG。

公开来源的独立 `reference_count=61931520001.0` 不是当前共享count。暖启动cap200000单独记录；共享count随实际训练样本增长。保存的受保护suffix是参考统计，不应描述成对应当前共享count的经验估计。

## 最小实现范围与接口

| 文件 | 变更范围 |
| --- | --- |
| 新 `CrowdSim/humanoid/training/terrain_statistics.py` | 受保护1024维统计生命周期、shared checkpoint validator；不创建优化器/replay/专家/SDK，不消耗RNG |
| `application/settings.py` | 一个 optional模式字段及纯stdlib校验；旧配置缺字段解释为legacy |
| `training/trainer.py` | constructor建立未绑定policy；warm_start绑定；update一个suffix hook；save/load严格policy状态 |
| 新profile | 以现有 `amp_style_mixed_state_action.yaml` 为基准，仅训练模式字段改变；其他参数/数据/初始化/课程保持 |
| `runtime/checkpoint_contract.py` / `runtime/amp_inference.py` | 校验新policy字段与saved task moments相符；仍安全CPU/mmap，无优化器/replay，四normalizer schema不变 |
| focused tests | 真实warmstart/update/save-resume/legacy/inference/篡改与caller-RNG检查 |

不改 AMPWorld、规划器、骨架、动作解码、Diffusion、原训练循环、原profile、旧权重或v7bank。`application/resources.py` 若不能通过仅settings及现有门禁拒绝 `--from-scratch`，需 root批准的一个纯frontend错误门禁：protected + from-scratch 在SDK之前失败；不默默降级legacy。

拟定生产接口：

```python
TerrainStatisticsPolicy.legacy()                      # 无统计写入
TerrainStatisticsPolicy.unbound(mode)                # opt-in尚不可act/update/save
policy.bind_public_warm_start(task_normalizer, *,
    initialization, original_reference_count, warm_start_task_count)
policy.after_task_update(task_normalizer)             # suffix唯一写入
policy.validate_checkpoint(task_normalizer, *,
    update_count, physical_transition_count)
policy.state_dict()                                  # legacy无额外key
TerrainStatisticsPolicy.from_checkpoint(saved_policy, *,
    config, initialization, saved_task_normalizer,
    update_count, physical_transition_count)
```

`bind_public_warm_start` 只在 `warm_start_pacer` 成功验证并转移完整 actor/critic/task/obs统计后调用。先保留actual `obs.count` 原值61931520001及来源SHA，再记录task暖count的cap结果200000。当前suffix必须等于实际公开 `obs.running_mean/var[377:1401]`，不覆盖已经变化的 learned统计。已训练、已绑定重复暖启动、from-scratch、非canonical结构或无normalize-observations明确失败。重复绑定的phaseguard须置于任何模型或统计copy之前；不能等末尾attach才拒绝，否则失败调用可能已覆盖live对象。

现 [statistics_policy.py](statistics_policy.py) 使用17KB NPZ是 docs独立fixture，不是正式部署资源。正式bind用已加载和严格校验的public对象；resume和推理直接从新checkpoint的内部policy状态验证/恢复，不需再次下载或读取public weights/该NPZ。

## checkpoint 与恢复合同

新 opt-in 顶层key `task_terrain_statistics_policy` 保存：版本schema、模式、受保护半开slice[20,1044]、source初始化模式/path/fullSHA、原reference_count、warm_start_task_count、post_update_calls、独立float64[1024] suffix_mean/variance及规范化数组SHA。source provenance和实际初始化记录必须一致，均不能宣称作者NavIsaacLab权重。

检查严格所有key/type/shape/finite/非负variance；counter必须严格int非bool；`post_update_calls == trainer.update_count`。新模式要求完整policy state，否则拒绝；legacy带非空policy state或unknown额外policy字段也拒绝。保存的suffix与 saved task.mean/variance[20:]逐位一致；task共享count等于声明暖启动count + actual physical_transition_count（float64安全精度范围内）；数组SHA与保存的suffix互证，不能把自校验SHA声称外部作者真实性证明。

恢复不再次运行warmstart、不重置counter、不复制参考覆盖损坏的task统计、不更改模型/RNG。`Trainer.load(map_location=device)` 会把policy tensors迁到目标device：shared validator先校验其dtype/shape/layout/device合同，再仅把该小policy字段显式归CPU；不能直接把当前原型的CPU-only要求接上GPU载入而拒绝合法resume，也不改legacy通用加载行为。推理从CPU/mmap载入，复用同一shared validator，然后仅保留actor/self/task与小JSON provenance；不保留优化器/replay或训练RNG。

旧checkpoint缺模式字段与policy key = legacy，旧四normalizer keys/张量格式保持。默认保存不添加policy key，默认统计与输出/RNG逐位不变；新增dataclass默认字段会改变新保存的config metadata字节，不能宣称整个checkpoint文件SHA仍相同。

## 合入门禁与实际训练次序

1. **默认合同**：既有legacy真实CPU updater与旧checkpoint下次Gaussian/optimizer/RNG完全一致；unknown/bool模式、legacy附带policy状态拒绝。cold application import不带Torch/Kit。
2. **绑定合同**：actual public warmstart moments与61931520001来源计数正确，cap200000另存；未绑定不能act/update/save；已训练suffix不能通过重新attach悄悄转换。
3. **更新合同**：真实CPU rollout/PPO/D update后仅地形suffix受保护；prefix20/count与未保护数学更新逐位一致，模型/优化器为真实更新结果；调用数恰好update_count。
4. **完整恢复**：真实 `torch.save` + `weights_only`恢复policy/Adam/normalizers/RNG，下一次更新继续；少/多hook、额外key、sourceSHA、mean/var/count/模式篡改均失败。训练load与无训练inference均验这一合同，caller成功/失败RNG行为沿用原明确约定。
5. **推理合同**：新opt-incheckpoint strict actorload与已载trainer的deterministic三输出逐位一致，统计不变化，不构造优化器/replay；旧checkpoint输出不变。
6. **观测证据**：root仅opt-in记录实际 action前 task原terrain1024、未裁剪z、clipmask/每actor比例及statsSHA；不能用post-step重构冒充实际动作输入。跨epoch初始/命令/RNG与来源限制仍披露。
7. **实际次序**：合入后先fresh4更新、恢复再4的真实GPU pilot。只有完整payload/counters/来源/恢复验收通过，root才决定fresh2000预算实验。此docs候选不启动训练，不生成checkpoint，不先填导航/风格改善结果。

当前 [staged YAML](amp_style_mixed_state_action_terrain_protected.yaml) **故意被当前loader拒绝**：新mode字段未生产集成。它不能直接运行；门禁结果见 [staged_profile_gate.json](staged_profile_gate.json)。新profile成功被加载后才形成可执行训练命令。旧论文/风格核、参考c274、失败包含分母、TableIII解释和34测试封存政策不变。
