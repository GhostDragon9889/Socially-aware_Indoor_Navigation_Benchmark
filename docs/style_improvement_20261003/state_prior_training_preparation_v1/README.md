# 公共 state prior 的独立真实训练准备

这里只准备新实验，`future_effect=false`。最终候选 SHA、独立源码 CPU 审查、输入冻结、独立 bundle、冷启动检查、GPU pilot 与终态结果均必须按实际状态单独登记。旧 protected-terrain 终态及历史报告不修改。

实际准备已完成到 CPU 准入，见 `actual_build_CPU_readiness.json`：peer source gate、exact old931 delta、934文件独立只读副本、实际 cold/CLI preflight/old full-network adapter 和其 finalized owned import ledgers 全部通过。旧931保留927原bytes、4modified+3new。Source manifest SHA `3db16db2ad3a4a8585b89557e7d88a9eb89d2e90461482f721c1ab1d52d44e2a`。`root_GPU_launch_commands.json` 提供可直接消费的完整 argv。GPU pilot/full2000 和效果检查仍未执行；初始 registration 的 prepared 字段是登记时状态，实际已完成状态由 readiness 独立记录。

`registration.json` 固定训练和评估合同；`commands_prepared.json` 从旧修正版 commands_v2 提取全部真实 CLI。新绝对 profile 只改变 coupled D/history-RMS 初始化及其说明。新训练仍使用 256 actors、rollout32、seed7、fixed half mixed curriculum、相同 PPO/reward/PD/replay、地形后缀保护。Fresh2000 为 16.384M transitions，每组 8.192M；不从 pilot 恢复。

## root 执行顺序

1. 等候 candidate 最终 manifest 和 peer source-bound CPU PASS。`--record-candidate` 只记录候选源身份，不能单独放行 build。
2. 独立 reviewer 对 exact candidate contract 写 `status=passed`、`candidate_contract={path,bytes,sha256}`、`source_bound_CPU_proof_passed=true`，并保存其实际证明。概念性的手工内存迁移不充当源码准入。
3. `--record-candidate` 在 `/tmp` 仅对修改目标复制旧文件并应用 fuzz0 delta，绑定每个旧931 before 和 rebased after 的完整 SHA/bytes。resources.py 的生产 before 与 old931 不同，因此其结果要求 old931 delta 后的 SHA，不直接复制生产 after。`--freeze-inputs` 绑定 builder、旧 manifest/launcher/profile/commands/watcher、完整候选 contract 和 peer review；之后 `--build` 才复制全部旧931 regular files并应用相同 delta，再核验 exact rebased SHA。拒绝未知目标、缺目标、已有新路径、SHA 变化或 patch 失败。常规文件独立复制，代码 0444、目录 0555。
4. 使用新的 strict launcher 做 `--verify-only`、`--probe-imports`、`--preflight -- <pilot4 argv>`，然后运行 `check_cold_bundle_cpu.py`。CPU 命令必须设置 `CUDA_VISIBLE_DEVICES=''`、四线程；SDK 解释器保持词法 `.venv/bin/python`。
5. root 单 GPU pilot4：真实退出后，独立复制封存 `step_4.pt` 及原始 result/metrics/owned import ledger，经新实验 reader 检查后才运行 resume4to8。真实退出后同样封存 step8。不得拿 mutable last.pt 作为恢复证据。
6. pilot 的 finite gradients、RMS、replay、complete resume 技术准入通过后，root fresh2000；旧 byte-identical watcher 独立封存 500/1000/2000。Watcher 仅检查传输、payload presence/counters，不代替完整恢复或终态准入。
7. 新 terminal reader 核验 exact source/actual GPU ledger、2000 metrics 原始前缀/result、16.384M、两组预算、完整 model/normalizers/Adam/replay/RNG/physical runner、公共 terrain reference/hook。新实验必须 `transferred=true`、bool=true、策略 provenance 正确。旧 reader 的硬 false 合同不能直接复用；训练后的 action 列不得要求仍为零，resume 不得重新归零。
8. 封存终态后，才按既有 full22／11982 expected frames 与四人6000controls／200秒协议评估。保留 failed/reset/idle；原90D reference 和 b25dffa 固定45D centers 不重拟合。test34 输出不读取。执行通过、loss、零跌倒或 route vertices 不单独证明样式改善，也不自动放行 native follow-on/default promotion。

## 已准备命令

以下为 root 未来显式操作；准备阶段未执行 freeze/build 或训练。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261003/state_prior_training_preparation_v1/build_isolated_training.py --record-candidate docs/style_improvement_20261003/state_prior_training_preparation_v1/candidate_delta_contract.json
```

通过的 peer review 文件形状：

```json
{"status":"passed","candidate_contract":{"path":"ABSOLUTE_ACTUAL_CONTRACT_PATH","bytes":0,"sha256":"ACTUAL_SHA256"},"source_bound_CPU_proof_passed":true}
```

上面的示例记录是明确无效占位。只有实际文件完整 SHA 与实际证明可用于输入冻结。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261003/state_prior_training_preparation_v1/build_isolated_training.py --freeze-inputs docs/style_improvement_20261003/state_prior_training_preparation_v1/inputs_manifest.json --candidate-contract docs/style_improvement_20261003/state_prior_training_preparation_v1/candidate_delta_contract.json --peer-review docs/style_improvement_20261003/state_prior_training_preparation_v1/peer_source_review.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261003/state_prior_training_preparation_v1/build_isolated_training.py --build docs/style_improvement_20261003/state_prior_training_preparation_v1/inputs_manifest.json
```

完整训练命令为 SDK Python + 新 launcher + `--import-report <对应GPU_import_reports>` + `--` + `commands.<pilot4/resume4to8/fullfresh2000>`。GPU 选择由 root 的唯一活动 Kit 会话控制；CPU preparation env 不用于 GPU 训练。数据、third_party、公共权重和 Diffusion 均以共享只读操作访问，未修改外部 OS 权限；新 `code/output` 只链接本 bundle 的 runtime_scratch。
