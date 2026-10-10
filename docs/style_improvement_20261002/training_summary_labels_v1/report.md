# 训练汇总标签修复

已修复 `scripts/summarize_paper_training.py`：Diffusion 图标题和数据边界从输入 JSON 的明确数据声明生成；AMP 判别器标题从记录的 `discriminator_input` 生成。缺失、矛盾或 AMP result 的 update 与最新日志不一致时使用 `unknown` / 通用标题。

旧脚本 SHA-256 为 `0d354070b6b3ad1e223b8091a1b99dd100003ee2fb9addcf3fe0389126482df2`；副本及已有手工验证记录保存在 [before/manifest.json](before/manifest.json)。检索三个测试目录未发现旧专属 pytest 文件。

## 实际 CPU 验证

| 已完成输入 | 输出数据类型 | 判别器标签 | AMP update |
|---|---|---|---:|
| social_diffusion_training_extended_v2 + amp_style_mixed_curriculum_v1 | ORCA 社会轨迹 | State-only | 2000 |
| diffusion_training + amp_independent_curriculum_v1 | AMASS 根轨迹 | Action-conditioned | 1000 |

对应输出见 [ORCA/state-only 汇总](orca_state_actual/report.md) 和 [AMASS/state-action 汇总](amass_state_action_actual/report.md)。两组均实际生成三张 PNG。ORCA 的 AMASS 初始化模型不会成为训练数据标签依据；专家动作字段也不会成为判别器输入模式依据。

9 项 CPU 回归通过，Ruff 通过。在 `/tmp` 的独立副本反转 ORCA 判断条件后，两个标签回归失败，证明它们能检出分类错误；工作区源码未为该检查改动。

## 保留的合同

原有 CLI 参数、输入 SHA 快照、`read_metrics`、不完整 JSONL 尾行处理、恢复训练进程分段、数值曲线和 reset 差值算法保持不变；上述 helper AST 和 CLI 参数 AST 与归档副本相同。`summary.json` 新增标签原始字段、训练配置、profile 路径、数据 provenance；输入未提供的数据集文件 SHA 保持未知。

本次修改仅涉及汇总脚本及其专属测试、证据；未修改训练代码、配置、环境、权重、训练 source bundle 或根代理管理的四份指南。图表不自动判断收敛或论文复现成功。

精确源码、输入、PNG SHA 与门禁记录见 [verification.json](verification.json)。

## 已执行命令

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python scripts/summarize_paper_training.py \
  --diffusion-result output/diffusion_amp/social_diffusion_training_extended_v2/training_result.json \
  --amp-metrics output/diffusion_amp/amp_style_mixed_curriculum_v1/metrics.jsonl \
  --amp-result output/diffusion_amp/amp_style_mixed_curriculum_v1/result.json \
  --output-dir docs/style_improvement_20261002/training_summary_labels_v1/orca_state_actual
```
