# 公共 state prior 零扩展 action 判别器：隔离候选 v1

候选尚未合并或用于 GPU/Kit 训练。`proposal/` 是四个现有文件的小修改、一个 stdlib helper、一个新增 profile；`before/` 保存四个精确源字节，`merge_manifest.json` 记录六个目标的 before/after SHA256 和字节数，`proposed.patch` 可用 fuzz=0 验证。总 payload 85,874 字节。

显式布尔 `warm_start_action_discriminator_from_state_prior=False` 默认保持旧行为。开启必须使用 state_action、10×195 历史、D 层 1024/512、per_history_coordinate、旧 warm_start_discriminator=False、观察/值 normalization 与 clip5。from-scratch 在 SDK-free preflight 独立拒绝，即使 terrain 模式为 legacy；未初始化的 actor/value/AMP reward/update 在模型执行、RNG 抽样或统计更新之前拒绝。

唯一的新实验变量是一项 **D 与 AMP history RMS 的耦合初始化**：fresh warm_start_pacer 验证公共文件 SHA `bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615` 后复制第一层的 1950 个 history 列和 bias，后 69 个 action 列置零，复制后层与 head，并复制公共 1950 项 AMP mean/variance/count；随后沿用原 count cap。没有新增随机网络构造。Actor/critic/task、保护 terrain 统计、PPO、PD mask、reward、replay、expert/data/command 采样的算法与顺序保持原路径。

Provenance 明确写 public_pacer_state_prior_zero_action_extension、耦合 RMS 转移、原 counts、零扩展的初始语义以及 public prior 对当前 AMASS 的重叠未知。它不声称 independent_random，也不声称作者 NavIsaacLab 权重。公共原始 counts 为 self/task 61,931,520,001、AMP 15,482,880,001、value 6,881,280,001。cap=None 也记录四项真实原 counts，false+None 保持旧空 counts 字典。

共享 helper 在 save、resume、inference preflight 严格校验 flag/provenance/模式/SHA/count-cap 的一致性。关闭或缺失 flag 时仍发现新策略 marker 会拒绝。恢复和推理只读取所选训练 checkpoint 的已保存状态，不重新读取公共模型、不重新零化训练后的 action 列。初始零列的声明描述初始化历史；它不要求训练后的列继续为零。

## CPU 验证

SDK Python，CUDA_VISIBLE_DEVICES=''，OMP/MKL 四线程；所有候选导入来自隔离 /tmp overlay。

- **44 个新增集成测试 PASS**，真实公共文件与独立篡改的训练 checkpoint：严格 flag/shape/mode/hash/provenance、cold readiness、legacy from-scratch、完整 CPU trainer.update、保存/恢复/推理和 cap=None 的全过程。
- legacy false 的 model、outputs、metrics、Python/NumPy/Torch CPU RNG 和旧 provenance 字典逐位相同，包括公共 warm start 的 cap=200000 与 None。旧 checkpoint 缺 flag 默认 false。配置序列化唯一新增默认字段单独记录为 `warm_start_action_discriminator_from_state_prior: false`。
- 公共 1950 网络与 2019 零扩展网络的初始 logits 最大 FP32 差 `4.76837158203125e-7`，小于 `2e-6`；不同合法 actions 的输出变化为 0。宽度变化可改变 GEMM 舍入，所以没有声明与原公共 logits 的逐位相同。
- 完整一次 CPU trainer.update：57 个 active action 列均有非零梯度，12 个 frozen 列梯度与权重恒零；58,083 个 action 权重在更新后非零。合成 CPU rollout 只用于 trainer 契约测试，16 个 transition 计数不代表真实物理采集。
- 保存/恢复所有 model tensors 逐位一致，恢复 D optimizer state，非零 action 列保留；推理动作与恢复 actor 一致、CPU RNG 不变，公共 controller 重读调用为 0。
- **222 个相关旧回归测试 PASS**，两项现存 runner AST fixture 缺 `pure_evaluation` 的失败单独复现于原生产源码并记录在 `inherited_failure_baseline.*`，候选不修改 runner 或旧测试。
- Ruff PASS。真实 mutation 在 /tmp 副本删掉新 bool type guard，测试得到预期 2 个 DID NOT RAISE 失败；随后恢复副本原始 SHA，冻结候选未改变。

`cpu_gate.json`、`cpu_evidence.json`、`cpu_tests.xml/log`、`regression_tests.xml/log`、`guard_mutation.json/log`、`source_bound_cpu_proof.json` 绑定具体源字节。早期 pytest 默认缓存写入了既有生产 `.pytest_cache`，没有修改代码或训练数据；复现工具现禁用 cacheprovider，后续写入只在此目录与 /tmp。该运行器缓存例外已向 root 报告，未擅自覆盖无 before 快照的既有缓存。

## 复现与有界应用

```bash
python docs/style_improvement_20261003/staged_action_state_prior_v1/run_cpu_gate.py --keep-overlay
python docs/style_improvement_20261003/staged_action_state_prior_v1/run_guard_mutation.py
python docs/style_improvement_20261003/staged_action_state_prior_v1/apply_bounded_candidate.py
```

有界应用工具默认只验证，拒绝未知目标、before SHA/字节不匹配、已存在的新目标、symlink 转向以及超出 120,000 单文件/150,000 总字节的 payload。它先验证 fuzz=0 patch，再由显式 `--apply` 原子发布六个指定 payload；六文件发布不是跨文件事务。这里只执行 dry-run，后续是否合并及训练由 root 完成。

新增 profile 是原 `amp_style_mixed_state_action_terrain_protected.yaml` 的同配方副本，仅添加新初始化开关和如实的初始化 metadata/assumptions。初始 D 函数等价不表示 public PACER rollout reward 时序等价；现有 state_action pre-step 语义不变。CPU 验证没有证明导航质量、风格提升或 AMASS 泛化。
