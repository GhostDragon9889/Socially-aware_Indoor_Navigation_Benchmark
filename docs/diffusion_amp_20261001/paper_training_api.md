# 真实rollout训练组件接口

## 数据gate

```bash
PYTHONPATH=. ../IsaacLab-Protomotions/.venv/bin/python -m CrowdSim.humanoid.training.validate \
  data/diffusion_amp/amass_manifest.json \
  --output docs/diffusion_amp_20261001/paper_reconstruction_manifest_gate.json
```

`--mode strict_author_experiment`会拒绝当前derived manifest。`--verify-package-hashes`显式进行完整多GB哈希；默认只核对当前尺寸/实际tensor和ID/source metadata，输出不会把声明SHA说成重新验算。

## 初始化与实际采集

```python
from CrowdSim.humanoid.training import PaperAMPTrainer, TrainingConfig
from CrowdSim.humanoid.training.rewards import paper_task_reward, paper_total_reward

trainer = PaperAMPTrainer.prepare(
    TrainingConfig(minibatch_size=2048, discriminator_batch_size=2048),
    device="cuda", provenance={"mode": "paper_method_reconstruction", "dataset_manifest": "..."},
)
# Optional declared transfer initialization. Omit this call for from-scratch.
trainer.warm_start_pacer(checkpoint, actual_dof_lower, actual_dof_upper,
                        body_names=actual_body_names, dof_names=actual_dof_names)
result = trainer.act(self_observation, task_observation)  # [B,357]/[B,1044]
```

把`result["applied_actions"]`通过明确的PACER PD decoder转换为物理targets，再推进真实仿真。`actions`是未经clamp的Gaussian样本，供PPO likelihood；`applied_actions`是`clamp(raw)*declared_action_mask`后的normalized有效命令，供action-conditioned D。默认canonical69中hand/toe12个冻结坐标为0，和物理PDdecoder及expert代理一致，防止D用无效关节的非零噪声区分真实policy。`freeze_toes=false`必须同时传给物理decoder，不能仅改训练侧。PD targets、normalized actions、实际joint torque三者不能混用。

采集后用实际位置/明确时间对齐的target计算delta；实际施加torque与对应omega计算reward：

```python
components = paper_task_reward(delta_squared, applied_joint_torque,
    measured_joint_angular_velocity, root_ego_forward_velocity,
    direction_mode=trainer.config.direction_mode)
style = trainer.amp_reward(amp_history, result["applied_actions"])
reward = paper_total_reward(style, components)
next_value = trainer.value(next_self_observation, next_task_observation)
```

`amp_history`为`[B,10,195]` newest-first，重置时须来自该reset对应reference历史或明确初始化，不能连接上一个episode；config11允许试验inclusive paper窗口但新D须重新训练。每帧AMP195特征来自同事实现的真实SMPL观测模块，不是raw69DOF复制。

物理终止的next value不bootstrap；时间限制需要终止观测的next value，同时切断跨episode GAE。**必须在env reset之前计算next value**，不要用新出生的观测代替终止状态。

## Rollout与更新

`trainer.update(rollout,expert_history,expert_actions)`输入：

| key | shape | 含义 |
|---|---|---|
|self_obs|T,B,357|rawself features|
|task_obs|T,B,1044|rawtrajectory+terrain features|
|amp_history|T,B,10,195|当前episode的历史|
|actions / applied_actions|T,B,69|未clamp样本 / clamp乘声明mask的normalized有效命令|
|log_prob / values|T,B|行为policy统计，normalizers在rollout内冻结|
|rewards / next_values|T,B|真实物理reward与reset前bootstrap value|
|terminated / truncated|T,B bool|物理终止 / 时间限制|
|episode_ids / history_episode_ids|T,B / T,B,10|可选审计，若任一提供须同时提供且窗口不跨reset|

expert history/action为`[N,10,195]`、`[N,69]`，action归一化范围[-1,1]，来源语义必须显式。这里不从AMASS悄悄制造实测专家力矩。

返回loss、梯度norm、D正负probability、KL/clip fraction、实际update/transition计数。可选保存`amp_rewards/trajectory_rewards/energy_rewards/direction_rewards/signed_power_w/forward_velocity`的T,B数组，update会返回mean/min/max，及backwards fraction，供诊断印出奖励的退化行为。

## 保存与恢复

```python
trainer.save(checkpoint_path, runner_state=physical_history_plan_rng_state)
trainer, runner_state = PaperAMPTrainer.load(checkpoint_path, device="cuda")
```

若restore后还需构造diffusion planner/其它model，应`load(...,restore_rng=False)`，完成planner与物理状态恢复之后再调用`trainer.restore_rng()`，然后开始rollout。否则随机模型初始化会消耗刚恢复的RNG，影响后续动作与DDPM采样。

训练bundle保存policy/newD、全部Adam状态、self/task/AMP/value population moments、config/provenance、更新/transition计数、Python/NumPy/TorchCPU/CUDA RNG。loader采用`weights_only=True`。调用方应保存真实simulator/历史/目标/episode状态作为tensor与primitive构成的`runner_state`；只保存trainer不能承诺跨中途物理episode完全复现。CPU-only加载CUDA训练若不恢复CUDA RNG需显式`restore_rng=False`。

本文件提供实际可调用组件API，SDK初始化与完整物理训练启动命令由父代理拥有的runner确定。本目录不含空训练器，也没有把CPU测试更新保存成可用行人预训练模型。
