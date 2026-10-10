# State-action 2000 后续验收准备

**身份：仅 CPU 工具、来源和协议准备。未生成未来 checkpoint、22 段捕获、200 秒捕获或效果结果。** 不读取 sealed34，不改训练、数据、模型或既有指标。

隔离 v3 的 926 个复制文件已逐项核对 SHA256、文件集合及只读权限；共享大数据只复核软链/size/mtime，本次不重复整库 SHA。实际评估须另有自己的同步 import ledger，不能拿训练 import ledger 代替。缺少 SDK 关闭后的 final 事件时如实保留 `runtime_finalization_observed=false`；由完整物理 result 证明运行结束。

## 工具和 admission

- [终态 CPU 审计](../audit_state_action_terminal2000_v1.py)：调用冻结的 1077 项 payload 审计，然后检查真实 fresh2000 result、16,384,000 transitions、组曝光及 fall/TL 计数。保持 `state_action.actual_training_checkpoint_audit.v1` schema，增加 `actual_terminal_fresh2000_confirmed=true`。
- [既有隔离评估入口](../run_isolated_state_action_evaluation_v1.py)：字节不变，其 `validate_admission` 与新增终态 JSON 兼容。只接受真实 step2000 SHA、D2019、完整恢复和隔离来源；不会从文件名猜次数。
- [full22 比较](../compare_state_action_full22_style_v1.py)：使用冻结 d19c9 调用链、c274 reference 和原 90D kernel；增加新模型并保持此前 **12 模型的 evaluation/comparison 字典逐值相同**。原 d19c9 中的 source135 仅对应旧9个模型；新模型另绑定 isolatedv3/old146 来源，不宣称全部模型源码相同。
- [200 秒原生记录审计](../audit_state_action_navigation200s_v1.py)：复用 f851 的 force/COM/clock/episode 计算，来源检查改为真实隔离评估的 entry/argv/checkpoint/import ledger；ee0 的停留和 plan-phase 计算保持原式。
- [纯 stdlib 来源 reader](../state_action_evaluation_evidence_v1.py)：检查实际执行的模块、所有复制文件和唯一 old146 control-loop AST；导入不加载 Torch、NumPy 或 SDK。

## 固定分母和门槛

| 项目 | 固定值 / 规则 |
|---|---|
| 完整跟踪 | validation22；11,982 metric frames；每段初始 reset 帧按旧规则排除 |
| 跟踪导数 / 加速度 | 11,960 同段区间 / 11,938 同段三帧 |
| 完整导航 | 4人 × 6,000控制帧 = 24,000 actor frames，200秒 |
| 原生物理 | 24,000提交物理步；96,000 actor-substeps；384,000足部 actor-substep 样本 |
| 导航导数 | 23,996 基础区间减去**实际**跨 reset 区间；不猜固定 reset 数 |
| 固定参考 | validation256个61帧窗，c274 字节不变 |
| 跟踪失败 | 原 mean-body-position error >0.5m 规则，不提前停止/丢弃失败序列 |
| 导航跌倒 | **现有 old146：Pelvis-ground <0.45m 或 root-up-z <0.1**；非有限状态抛错，非改成成功/跌倒计数 |
| 停留 / 接触 | 同 episode 位移速度 <0.1m/s；原生 normal-force 严格 >1N |
| 风格 | 原 90D speed-conditioned Gaussian MMD²，sigma1，dt1/30，固定 bins/min30/max256 |

全部失败、跌倒、reset、末帧保留；只按原规则去除跨 reset 的导数。upright 子集仍为 up-z≥0.8、height≥0.7 的**辅助**结果，不替代 inclusive。足部 link 原点运动不是接触点 slip；AMASS reference 真 COM/contact/friction 缺失继续明确。

`status=passed` 只表示运行/审计完成。既有工程导航推广门槛为完整6000、0F、路线顶点不少于 negative-guide20 的21、inclusive 风格不劣化，同时报告完整跟踪。它不是作者论文阈值、统计显著性或八场景复现证明。

## 执行顺序

完整机器 argv 见 [commands.json](commands.json)。以下未来目录都必须为空或不存在。GPU 命令仅由 root 在单 App 时段执行；本准备没有启动它们。

**1. 实际训练完成、sole watcher 保存 step2000 后，CPU admission：**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/audit_state_action_terminal2000_v1.py \
--output docs/style_improvement_20261002/state_action_isolated_checkpoint2000_terminal_review_v1
```

**2. root 读取上一步真实 SHA，运行隔离 full22 / 200 秒（串行）：**

```bash
STATE_ACTION_SHA2000=$(../IsaacLab-Protomotions/.venv/bin/python -c "import json;print(json.load(open('docs/style_improvement_20261002/state_action_isolated_checkpoint2000_terminal_review_v1/audit.json'))['checkpoints'][0]['checkpoint']['sha256'])")
../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/run_isolated_state_action_evaluation_v1.py \
--protocol tracking \
--checkpoint output/diffusion_amp/amp_style_mixed_state_action_isolated_snapshots_v1/step_2000.pt \
--checkpoint-sha256 "$STATE_ACTION_SHA2000" \
--checkpoint-admission docs/style_improvement_20261002/state_action_isolated_checkpoint2000_terminal_review_v1/audit.json \
--output output/diffusion_amp/motion_tracking_mixed_state_action_isolated_full_validation_v1 \
--import-report output/isolated_training/mixed_state_action_v1/state_action_tracking2000_v1_GPU_imports.json
```

导航相同 admission/checkpoint/SHA，`--protocol navigation`，输出 `output/diffusion_amp/shared_mixed_state_action_isolated_ground_support_200s_v1`，ledger `output/isolated_training/mixed_state_action_v1/state_action_navigation2000_v1_GPU_imports.json`。完整注册 argv/场景/guide20/eta0 等由既有 adapter 生成。可先附 `--preflight-only`，但应为预检另选 ledger 文件，实际 ledger 必须新建。

**3. 实际 full22 终态后，CPU 固定比较：**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/compare_state_action_full22_style_v1.py \
--input output/diffusion_amp/motion_tracking_mixed_state_action_isolated_full_validation_v1 \
--checkpoint-admission docs/style_improvement_20261002/state_action_isolated_checkpoint2000_terminal_review_v1/audit.json \
--import-report output/isolated_training/mixed_state_action_v1/state_action_tracking2000_v1_GPU_imports.json \
--output docs/style_improvement_20261002/state_action_style_v1/full22_step2000_comparison_v1
```

**4. 实际 200 秒终态后，CPU 原生记录审计及原停留计算：**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/audit_state_action_navigation200s_v1.py \
--input output/diffusion_amp/shared_mixed_state_action_isolated_ground_support_200s_v1 \
--checkpoint-admission docs/style_improvement_20261002/state_action_isolated_checkpoint2000_terminal_review_v1/audit.json \
--import-report output/isolated_training/mixed_state_action_v1/state_action_navigation2000_v1_GPU_imports.json \
--output docs/style_improvement_20261002/state_action_navigation_ground2000_review_v1
```

导航新模型风格仅运行 `commands.json` 的 `navigation_new_only_style_CPU`，使用同 c274/min30/max256。已有 `mixed_curriculum_navigation_style_v1` 的 guidance20/root1000/mixed1000/mixed2000 字典保持冻结；未来以追加方式比较，不重拟 kernel/bins/reference。

## CPU gate 的范围

focused 测试只检查 schema/终态/计数/固定 argv/来源事件/冷导入；synthetic 测试 metadata 没有作为物理证据保存。真实926文件 SHA 及 sole-loop AST 已只读复核。未来 checkpoint、物理捕获及模型质量待实际执行，不能从这些 CPU 门禁推断。

最终联合门禁 **56 passed**（新增38项 + 冻结registration18项），Ruff PASS。见 [tests_final.log](tests_final.log)、[tests_final.xml](tests_final.xml)、[ruff_final.log](ruff_final.log)、[来源快照](source_freeze.json)。
