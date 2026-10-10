# future81 单变量课程：真实AMASS CPU接口准备

[训练时长诊断](../shared_social_episode_horizon_CPU_v1/README.md)显示已完成S1/S2的共享组episode很短，
且约一半canonical规划节点处于真实片段之外。本候选将AMASS参考重置所需真实未来从31改为81帧，
保留全部2848动作；目标是验证更长连续来源是否改善路线执行和动作风格。
这是假设实验，尚未训练，不能宣称改善或作者原配方。

## 本批实现与实测

- 新课程：[minimum_horizon_curriculum.py](../../../CrowdSim/humanoid/training/minimum_horizon_curriculum.py)。
  独立类型/状态schema；只修改该实例的AMASS重置帧规则，不修改原模块或SDK类。
- 合法区间仍为先uniform clip、后uniform frame；所有源片段必须支持history10+future81，
  否则整份实验拒绝，不能丢弃短动作、重采样或替换来源。
- procedural重置仍future1；原AMASS/procedural绑定、目标查询、组重置及专家采样算法沿用。
  本候选不构造NN、判别器或物理World。
- 真实256行人CPU接口会话78833实际退出0（`2e1e4d`）。
  [实测报告](actual_corpus_probe.json)绑定原始bank/manifest/时长诊断和6项源码。
  [退出及检查记录](tool_execution.json)保留初版检查失败与修正后的真实结果。

实际固定seed7/private seed8抽样中，128条AMASS重置有127条帧改变，最少剩余81帧。
全部2848 ID保持；AMASS动作ID抽取、procedural重置帧、每次调用的来源RNG消耗保持。
新旧帧均按同一实际torch.rand结果与各自合法区间逐值重算验证。
给定同一合法来源帧，两实例的51点canonical/151点native轨迹、10个未来目标完全相同。
随后4096个专家来源窗口的状态、历史、ID、帧及下一帧标记完全相同。
候选CPU holder状态可往返，旧课程不能误接收新schema。

**每次调用规则相同不代表整段训练RNG或专家曝光相同**：真实物理结果改变重置频率后，
共享组及procedural重置会继续使用原corpus generator，后续抽样次序可能随episode变化。
上述数据只证明接口和单次抽样契约，不证明完整训练轨迹、风格或物理性能。

## 验证

7项新检查和原课程30项回归，共37项通过，Ruff通过。
把剩余帧边界条件反转后，实际3项失败、34项通过；随后逐字恢复并再次37项通过。
初版测试误调用不存在的`sample_command_points`，1项失败、6项通过（`f96ff4`）；
已按原源码实际`interpolate_reference`和`future_targets` ABI修正，失败记录保留。

本次probe只读取真实来源bank并验证CPU接口，不会模拟接触、行走或跌倒。
CPU holder state往返也不是完整SDK/corpus/history训练恢复；原训练entry仍不保存此恢复状态。

## 后续运行接入

当前封存World明确要求原`MixedTrajectoryCurriculum`类型与来源。
新类型不能直接注入现有entry；需要独立且显式的World来源准入、训练入口与出处记录迁移。
后续应在相同数据/seed/预算、专家规则和独立评价协议下进行单变量对照。
先通过实际SDK smoke与256人检查，再进行正式训练；原默认权重不随CPU准备推广。
S3当前owner cell581仍持有GPU，必须等待训练、终态审计与串行独立评价完成后才启动新GPU工作。

[NavIsaacLab原论文IV-B/IV-C](https://arxiv.org/html/2606.26265v1#S4)
要求真实AMASS风格和目标条件AMP，再独立评价跟踪、风格及路线/物理表现；
没有公开与本地相同的重置剩余帧设置，本候选需以实际效果验证。
