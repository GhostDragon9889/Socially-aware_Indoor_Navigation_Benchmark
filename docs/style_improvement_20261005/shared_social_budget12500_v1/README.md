# S3：固定真实 AMASS2848，仅延长训练预算

已从全新初始化启动实际 GPU 训练；训练工具会话为28927。
终态退出尚未观察，当前没有新的效果结果。[实际启动](training_launch.json)、
[当前状态](live_status.json)及[训练前登记](registration.json)分别保留。

## 单变量实验

相对已完成的S2，仅将 `runtime.updates` 从2000改为12500。
固定真实AMASS v7的2848片段／原clip-uniform先验、seed7、256人／64组、T32，
Pop-Art、完整六轮PPO、奖励、判别器、物理、初始化和采样均保持。
不续训S2，不挑选中途最佳checkpoint，预选 `step_012500.pt`。

计划为102,400,000物理转换、51,200,000专家窗口、150,000 PPO minibatch；
这些是计划数量，完成必须以真实终态、原始rollout、模型／RMS／优化器审计为准。
当前第一轮已实际施加12个PPO minibatch并生成8192物理转换，不能据此证明成熟动作。

[原始AMP论文 §8.1](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf#page=8)
的任务训练使用100–300M样本。本轮以此检验训练时长，但原论文的任务与本地不同，
这个数量不是NavIsaacLab作者预算，不保证合格或超越效果。
以实际S2的28,373.745秒／2000更新估算，本轮约49.3小时；实际速度可能变化。

## 配置与来源

[training.json](training.json)已经封存；[CPU检查](CPU_configuration_preflight.json)
通过原入口完整字段验证和10项训练来源检查；还原updates后与S2配置完全相同。
v1 profile的旧purpose标识包含pilot2000，该兼容标识保持；
实际预算由`runtime.updates`与完成计数决定。
训练入口、安装SDK、此前完成的baseline和权重未改动。

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/train_shared_social.py --value-normalization popart \
  --config docs/style_improvement_20261005/shared_social_budget12500_v1/training.json \
  --device cuda:0 --output output/diffusion_amp/shared_social_budget12500_new_run
```

只同时运行一个GPU训练／模拟进程；新输出必须为空或不存在。
当前训练输出为 `output/diffusion_amp/shared_social_budget12500_v1`。

## 终态独立评价接线

原CPU训练审计、封存和权重准入按实际runtime动态验证预算。
原 `run_shared_social_post_training.py` 固定2000／16.384M，不能直接用于本轮终态。
新增 [预算评价入口](../../../scripts/run_shared_social_budget_evaluation.py)及
[独立评价登记](evaluation_registration.json)，从登记配置计算终态更新／曝光／checkpoint。
[4项CPU测试、来源检查及配置验收](budget_evaluation_CPU_preparation.json)通过；
与原入口的全部五阶段命令、选窗、指标、物理、清理及子进程退出检查字节一致。
未来12500终态权重和捕获尚未产生，因此未执行终态NN／SDK评估。

[外部工具owner启动](terminal_owner_launch_execution.json)已经接管会话28927，
exec cell581串行观察真实退出，再审计／封存／独立评价／稳定性／S2-S3对照。
[阶段状态](terminal_owner_state.json)记录实际工具结果；不得由PID或日志推断退出0。
冻结来源漂移或某阶段失败会停止后续并保留数据。

后续顺序：实际训练退出0 → 完整CPU审计／封存 → 全validation22及冻结风格 →
同正式Diffusion40k的N4 smoke／200s → 全量原生CPU → 与S2逐轴独立比较。
既有比较reader约束固定评价协议；训练预算不同将明确披露。
完成后的严格补充准入和模型质量判定还需实际执行，不能将CPU准备当作未来效果。
固定验证参考和中心、失败数据全部保留，不用Test34筛选、不自动晋升默认。
八场景／N3–15、多seed及作者私有协议仍需验证。
