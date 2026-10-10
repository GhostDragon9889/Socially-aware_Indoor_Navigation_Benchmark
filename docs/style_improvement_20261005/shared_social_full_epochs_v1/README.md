# S2：保持 Pop-Art，执行完整六轮 PPO

## 当前实际状态

2026-10-05 已通过真实8人、256人短训练、终态 CPU 审计、冻结推理与
四人正式 Diffusion 的30控制步接线检查；各进程实际退出0。
全新256人／2,000更新训练已由外部owner观察真实退出0（`4d7dfa`，工具会话35913）；
CPU终态审计`6e8be8`／0及封存`14ca61`／0。实际16,384,000物理转换、
8,192,000专家窗口，终态权重SHA `950a07f9831188b592cf6bc7c949519c38137d673f0bf794dfa90e4a1dba4f05`。
完整跟踪／风格子流程已退出0，四人30步接线通过，200s导航仍在运行；
完整后续退出、CPU复算与严格补充准入尚待完成。所有部署默认模型保持。

- [预登记](pilot2000_registration.json)：固定v7真实 AMASS2,848、seed7、256人／64组、T32、2,000更新，终态`step_002000.pt`。
- [实际启动](pilot2000_launch.json)：`84c50b`／会话35913；启动记录中的终态退出为null。
- [完整训练实际退出与封存](pilot2000_execution.json)：外部训练／CPU退出绑定、终态result／checkpoint／audit SHA；启动时的null未改写。
- [运行阶段](terminal_owner_state.json)：工具owner为exec cell278；只有真实训练退出0才进入终态审计及后续评价。
- [封存owner源码](terminal_tool_owner.js)、[准备说明](owner_preparation.json)：源码／配置／登记／基线退出记录与清单共22项绑定。异常停止后续步骤，保留已有数据。

源于 [PACER 官方循环](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/learning/amp_continuous.py)
与[公开 SMPL CNN 配置](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/data/cfg/train/rlg/amp_humanoid_smpl_cnn_task.yaml)：
公开循环执行全部mini-epochs，配置使用6轮及constant LR。
本地 S1 末100更新全部因KL停止，平均只施加3.34／12个minibatch。
S2只将`training.target_kl`从.02改为null，其余S1训练与runtime字段逐字段相同。
保持 PPO clipping、value RMS 输出补偿和原D／replay配方。
这个单字段改动不使本地控制器等价于作者或PACER；效果仍待测量。

## 已实际完成的检查

| 检查 | 实际结果 | 退出证据 |
| --- | --- | --- |
| CPU 六轮 fixture | 6 epoch／12 minibatch，无KL停止 | `703d99`／0；[记录](CPU_preparation_execution.json) |
| 真实8人、T4、1更新 | 32物理转换；6 epoch／6 minibatch | `9400de`／0；[训练与审计](smoke8_execution.json) |
| 8人冻结 CPU 推理 | 实际pre-action原始观测；69D输出／57D mask／RNG保留 | `6baa33`／0；[退出](smoke8_inference_execution.json) |
| 真实256人、T32、2更新 | 16,384物理转换；每次6 epoch／12 minibatch | `dc7db4`／0；[训练与审计](smoke256_execution.json) |
| 256人冻结 CPU 推理 | 严格Pop-Art配置及全部RMS加载 | `f9ba63`／0；[退出](smoke256_inference_execution.json) |
| 4人Diffusion接线 | 正式EMA40000／30控制步；正常原生释放 | `14d0ef`／0；[退出](navigation_smoke_execution.json) |
| 独立比较reader | 5个CPU support/kernel/receipt检查及真实S1 admission通过 | [源码修订记录](comparison_source_revision_execution.json) |

8人短检查的平均KL为508.3；256人两次更新为2.0928／.2461，所有模型和优化器张量有限。
这些检查确认实际完整更新、严格权重加载和退出链路，不能证明长训练稳定性或步态质量。
一次CPU审计CLI参数错误的退出2及原日志保留在[8人外部退出记录](smoke8_observed_exits.json)；
随后按位置参数重试实际退出0。未将失败调用计作通过。

## 自动后续与结果边界

工具owner串行执行：实际训练终态退出→CPU终态审计→封存权重与rollout→
独立冻结加载→全部validation22／11,982控制帧与固定90D／45D风格→
四人30步及6000步／200秒导航→原始数组CPU审计→终态GAE稳定性重算→S1/S2独立对照。
所有进程真实退出分别记录；不由PID消失、日志尾或结果文件推测正常退出。

预选2000步终态，无最佳clip或最佳checkpoint筛选；失败帧全部保留，
共同速度区间另报，不填补无支持区间，不重新拟合中心，不读取Test34数组。
预期全预算为16,384,000条物理转换、8,192,000个专家窗口和24,000个PPO minibatch；
预期数量不是完成证据，以实际终态审计为准。

完成后应生成`pilot2000_execution.json`、`post_training_execution.json`、
`training_stability_CPU.json`、`S1_S2_independent_comparison.json`及
`completed_CPU_evidence_execution.json`。这些文件尚未生成时表示对应阶段尚未完成。
任何一步失败，后续未启动阶段保持未测量。

2026-10-05新增的[严格补充准入](../shared_social_admission_v2/S0_S1_supplemental_admission_v2.json)
进一步核对owner/pipeline/child测量绑定、完整tracking协议、底层风格math／本方capture来源。
90D桶需相同平衡样本数与参考索引才能给出方向；其余桶保持unknown，物理／pose轴独立报告。
原冻结owner及canonical reader未修改。S2自动流程完成后，还必须实际运行下面的补充检查，
才能使用它的S1/S2比较；自动JSON的比较字段不是质量通过证据。

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/verify_shared_social_evidence_admission.py \
  --baseline-post-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_execution.json \
  --candidate-post-receipt docs/style_improvement_20261005/shared_social_full_epochs_v1/post_training_execution.json \
  --output /tmp/S1_S2_supplemental_new.json
```

S1完整原始CPU支撑诊断显示80.46%有效位置间隔低速，最长连续双支撑149–176s；
[终态D离线输入诊断](../shared_social_D_input_diagnostic_v1/README.md)的动作通道扰动影响很小。
后续以真实运动／速度／覆盖和路线检查改善，不能只用跌倒、D分类或MMD挑选结果。
同一AMP的[SFM联合预测接口](../sfm_shared_amp_preparation_v1/README.md)已通过22项纯CPU测试，
真实native接入和SFM/DP多场景配对仍待串行实施。

[S1 完整独立结果](../../SHARED_SOCIAL_POPART_RESULTS_ZH.md)显示0/22跟踪、
5次导航跌倒、37个peer pair-controls、3个路线点及覆盖率下降；
当前优化重点仍是完整动作、运动多样性、速度、路线和物理各轴。
[NavIsaacLab 原论文](https://arxiv.org/html/2606.26265v1)的作者私有动作ID、
八场景任务、碰撞分母和风格参数尚未匹配。单seed本地fixture不能证明达到或超越论文。

## 复查运行状态

```bash
python scripts/shared_social_training_status.py \
  output/diffusion_amp/shared_social_full_epochs_pilot2000_v1
```

状态reader只读取日志；进程退出和独立效果以外部owner及完成记录为准。
当前GPU由该owner的独立导航评价占用；训练已正常退出，其后的模拟保持串行。
