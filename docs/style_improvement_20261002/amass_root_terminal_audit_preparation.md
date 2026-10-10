# AMASS root课程终点与22条tracking验收准备

本页是验收准备，不表示新1000更新训练或新完整22条tracking已经验收。仅新增docs下CPU审计脚本，未读取mutable checkpoint，未打开封存34条test捕获，未修改业务代码。

## 训练终点

[audit_amass_root_training_final.py](audit_amass_root_training_final.py)只接收不可变`step_1000.pt`、passed结果和完整metrics。调用此前冻结的checkpoint/source审计器，进一步核对：

- 1000条update和8,192,000 transitions逐项连续，最后累计fall/timeout与结果一致；四个250-update段分别统计。
- 从实际warm-start计数、cap和每轮真实batch大小精确推导四个normalizer最终count；记录mean/std及1024维height部分，不凭标准差判断收敛。
- 每个Adam参数state的step等于metrics中接受的optimizer minibatches累计值；两个optimizer分别核对。
- Replay容量/size/accepted/cursor、history/actions形状及冻结applied action坐标；完整finite检查由核心审计器执行。
- Python/NumPy/CPU Torch RNG在局部validator接受；CUDA RNG只核对序列化字节，没有在CPU恢复GPU状态。
- 固定checkpoint/结果/metrics SHA在检查前后不变，与124-source archive及保存物理/逻辑状态恢复合同绑定。

新的统计函数已用真实不可变pilot step4验证：[准备JSON](amass_root_terminal_auditor_preparation.json)。这是4-update实际数据测试，不是未来1000-update验收。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/audit_amass_root_training_final.py \
  --checkpoint ACTUAL_TRAIN_DIR/step_1000.pt \
  --result ACTUAL_TRAIN_DIR/result.json \
  --metrics ACTUAL_TRAIN_DIR/metrics.jsonl \
  --output NEW_REPORT.json
```

## 完整22条tracking

[audit_amass_root_tracking_protocol.py](audit_amass_root_tracking_protocol.py)验收全部22份NPZ，严格按结果列举文件，不用glob补缺。核对实际checkpoint SHA、capture SHA、manifest/实际train-parent与validation不交叉、连续30Hz源frame/time、一次初始reset、完整11982 poststep metric帧、所有fall帧保留。

用原始world-space24-body Euclidean与SO(3) geodesic函数重算存储的primary errors、序列成功/跌倒规则和frame-weighted整体聚合，确保报告与raw一致，不重新对齐或换指标。新旧全部reference姿势/速度、XY命令、时间/phase和初始实际完整物理状态必须逐位一致；各自后续物理轨迹不被假定相等。

准备模式已经用旧negative完整22条实际捕获验证：[准备v2 JSON](amass_root_tracking_protocol_preparation_v2.json)。新候选实际验证字段明确为false；没有计算MMD或误差分解。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/audit_amass_root_tracking_protocol.py \
  --candidate ACTUAL_COMPLETE22_DIR \
  --output NEW_PROTOCOL_REPORT.json
```

误差分解由CrowdSim代理负责，官方冻结风格/MMD由资源代理负责。科学效果评价必须使用完成的实际报告，不能以训练执行/恢复通过替代风格或tracking质量。
