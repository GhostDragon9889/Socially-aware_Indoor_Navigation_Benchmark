# state-action2000：完整22 / 200秒预登记

2026-10-02。**协议及隔离旧146 parser检查通过；未来step2000尚未做payload/资源预检，也没有模型效果结论。** 实际fresh2000_v3已由主任务启动，16.384M transitions，两固定half各8.192M。本工具没有启动GPU/Kit。

## 固定评估

- 完整validation22：不指定motion IDs、不限clips/steps；同120Hz物理/30Hz控制、rootXY-only参考、原.5米tracking门槛、seed7确定性、全部失败帧保留。相同fixture预计11982正式poststep帧，实际验收以未来capture计数为准。
- 多人闭环：同原guide20/eta0、4人6000控制步/200秒、Diffusion5000、timed_path、原1.5m/s causal参考、15步同步replan、原cuboids、native normal-force contact/ground、1N阈值、seed7，sole App/world。新D不作为部署必要资源或伪造训练transition。
- 主风格：原c274 reference、82ee/b09的90D pose+velocity/sigma1/原速度bins/min30/max256，全时段inclusive；upright仅次要。跨reset导数间隔排除，跌倒/低姿态/低速帧保留。unsupported保持unknown。
- `result.status=passed`只是执行完成。promotion最低要求完整200秒0fall、路线累计顶点至少既有negative guide20的21，inclusive风格不退于同200秒基线；仍需报告静态/peer接触与全部tracking tradeoff。无作者同式/八场景、人口显著性或contact-point slip声明。

## 精确源码执行适配器

[run_isolated_state_action_evaluation_v1.py](../run_isolated_state_action_evaluation_v1.py)仅复用已冻结v3 `verify/activate/ImportLedger`，执行只读CODE中的原benchmark/driver；不修改926复制文件或旧146源码。同步ledger记录实际owned模块及adapter/benchmark/manifest完整SHA。prelaunch bytes与实际import观察的范围分别记录，不将CPU parser假称GPU attestation。

独立未来payload审计必须提供 `state_action.actual_training_checkpoint_audit.v1`：实际sealed2000完整SHA、2000更新/16,384,000transitions/2019D、完整恢复、未迁入公开D/RMS、实际observed imports隔离及所有检查通过。未有该证据时adapter拒绝执行；不存在的检查点没有被填充虚假SHA或生成checkpoint。

```bash
../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/run_isolated_state_action_evaluation_v1.py \
  --protocol tracking \
  --checkpoint output/diffusion_amp/amp_style_mixed_state_action_isolated_snapshots_v1/step_2000.pt \
  --checkpoint-sha256 ACTUAL_SEALED_SHA256 \
  --checkpoint-admission ACTUAL_INDEPENDENT_PAYLOAD_AUDIT.json \
  --output output/diffusion_amp/motion_tracking_mixed_state_action_isolated_full_validation_v1 \
  --import-report output/isolated_training/mixed_state_action_v1/full22_validation_actual_imports_v1.json
```

以上两个大写位置是明确待填的未来证据，不是可直接执行的伪hash。将`--protocol navigation`与output/import-report改为[登记JSON](registration.json)中的200秒新路径即可使用完全固定闭环argv；`--preflight-only`可先做实际存在资源预检。GPU由root串行运行，本准备没有执行adapter物理入口。

[18项CPU拒绝/完整协议测试](tests.log)与[Ruff](ruff.log)通过；[纯stdlib profile/parser导入证据](configuration_only_imports.json)确认未导入Torch/NumPy/SDK。只核对metadata和既有source/manifest，不重算1.15GB bank、不读sealed34输出。[源与证据冻结](source_freeze.json)。

## 重建边界

history+action符合论文Eq2输入形式，但10帧区间约定、BCE/奖励尺度、负向backward选择及专家nextframe归一化PD位置代理属于明确工程实现，非作者测量expert control。2848为本地AMASS-derived非重叠段，未确认作者原ID；D输入、随机初始化、独立RMS、pre-step history一起改变，不称只增加69维的严格单变量消融。
