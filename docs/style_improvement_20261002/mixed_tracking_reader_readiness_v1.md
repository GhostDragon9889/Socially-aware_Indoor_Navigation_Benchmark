# 混合课程完整22段 CPU reader 已准备

仅新增 docs 工具/测试，业务源码、权重与数据保持冻结。27项 focused tests 和 Ruff 通过；真实 sealed500 检查点载入/嵌套状态、三套 RNG、real151/proc51 活动计划与 episode history 门禁通过，未导入 SDK，未读取新 validation 或 test 捕获。该结果只证明 admission，不代表 GPU 评估或行人质量。

严格绑定训练前 source135、watcher/actual full2000 argv，以及任何 mixed GPU eval 前登记的 full22 protocol。实际物理 benchmark 记录6份 runtime 源；CPU admission 使用8份实现源，另列并分别校验。seal row 本身没有 source SHA，不能把字节封存误称源码/模型语义审核；独立 payload audit 是必需输入。

500/1000/2000 每组实际转换为2.048M/4.096M/8.192M，总转换4.096M/8.192M/16.384M。2000总预算是旧 negative1000 的两倍。完整22参考、初始化、history、phase、clock、dtype和bitwise配对后保留所有失败/跌倒帧，重算原0.5m序列失败标准、ALL22全局/局部误差与速度分层；success-only/common-success仅为辅助分母。官方 style 指标由另一审计工具负责，不重复计算。34test此前已开封，本轮不是新盲测。

运行时替换 `CANDIDATE_RESULT` 和独立审核路径，输出必须是新目录：

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
  ../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/audit_mixed_curriculum_tracking.py \
  --candidate-result CANDIDATE_RESULT --candidate-step 500 \
  --checkpoint-seal output/diffusion_amp/amp_style_mixed_curriculum_snapshots_v1/ledger.json \
  --independent-payload-audit docs/style_improvement_20261002/mixed_curriculum_step500_milestone_admission_v1.json \
  --output docs/style_improvement_20261002/mixed_step500_tracking_comparison_v1
```

1000/2000使用对应 sealed snapshot、完成后的独立审核、candidate-step与全新输出目录。该 reader 不启动 Kit/GPU、不修改部署默认。
