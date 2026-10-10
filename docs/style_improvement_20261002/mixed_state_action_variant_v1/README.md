# 固定half混合课程：state-action AMP 未训练变体

> 历史身份：以下主体是新增profile时的未训练CPU快照。后续实际隔离4→8 pilot见文末；保留原结论和原源码冻结，不能把旧CPU证据改称物理训练。

2026-10-02。新增 [amp_style_mixed_state_action.yaml](../../../configs/diffusion_amp/amp_style_mixed_state_action.yaml)，已完成 CPU 配置 / 资源 / 张量合同验证，**没有启动实际训练，没有新 checkpoint 或效果声明**。现有 state-only mixed profile、训练 / benchmark / runtime 业务与模型不改。

## 相对原 mixed profile 的变化

| 项目 | 原 state-only | 新 state-action |
| --- | --- | --- |
| `training.discriminator_input` | `state` | `state_action` |
| `training.warm_start_discriminator` | `true` | `false` |
| `initialization.discriminator` 元数据 | 完整公开state-only D与RMS迁移 | `independent_random_history_and_action` |
| D 第一层输入 | 10×195 = 1950 | 10×195 + 69 = 2019 |
| AMP normalization representation | 1950 history coordinates | 同1950，避免另加 per-frame 表示变化 |
| AMP 初始统计 | 公开RMS迁移并cap | mean0 / variance1 / count0，不迁入公开RMS |
| 现有runner选择的AMP时刻 | post-step history | pre-step history与实际applied action |

其余所有 `TrainingConfig` 字段与原 mixed baseline **逐值相同**：PPO / actor / critic、PD及57有效坐标、replay、两个D更新、child-joint专家速度、uniform专家、negative方向、AMP权重2 / task0.5、mirror / likelihood / KL、seed7等。`trajectory_training`、reward与数据声明也完全相同；固定前half AMASS / 后half procedural2m/s、1.5m/s²，151 native plans / 原51 procedural插值、独立 / 共享RNG合同仍沿用原实现。

上表后四项是已有代码由配置选择产生的差异，不是新改业务。它们意味着这不是仅增加69维输入的严格单变量消融。额外输入层参数也改变构造时随机数消耗；seed与保存协议相同不保证不同D变体的随机动作 / reset序列逐位相同。

## 论文公式与重建边界

[NavIsaacLab IV-B / Eq2](https://arxiv.org/html/2606.26265v1#S4.SS2) 明确把 history 与 action 一起送入D；新配置对应这个输入形式。论文并没有给出唯一可还原的专家action定义或完整D训练细节，因此下面选择仍需单独声明。

| 项目 | 此变体实际含义 |
| --- | --- |
| history | 10帧 `t,t-1,...,t-9` / 195维，保留公开PACER式约定；论文区间写法有10/11帧歧义 |
| action | policy实际 `clamp(raw)×57-active-mask` 的69维归一化PD command；12冻结坐标为0 |
| expert action | next-frame AMASS pose除以同PD scale、clamp、冻结12坐标的代理；不是实测专家command / torque，不是inverse dynamics |
| D输出 / loss | 现有 BCE + gradient penalty，reward用softplus / epsilon上限 / scale2；不是作者公开的唯一loss定义 |
| 总奖励 | 沿用优化baseline的 `2*r_amp+0.5*r_task`；[论文总式](https://arxiv.org/html/2606.26265v1#S4.SS2)的AMP系数为1，不能把此配置写成完全同式 |
| 方向项 | 沿用显式negative `backwards_penalty`；与印出的Eq6正向反走奖励符号不同 |
| expert数据 | 本地v7的2848真实AMASS-derived非重叠段，不是已知作者2848原ID；validation/test父源保持独立 |
| 命令 / 场景 | 工程固定half root/procedural课程，单一isolated平地；Diffusion单独训练、用于后续部署，不是已完成论文八场景训练 |

CPU通过只能证明可构造 / 合同正确，不能证明随机新D学出了有用风格、后续PPO收敛或导航稳定。

## 实际 CPU 验证

- [14项专项回归](tests.log) / [JUnit](tests.xml) / [Ruff](ruff.log) 通过。新鲜进程profile/resource gate不导入Torch / NumPy / SDK；错误training字段、mode / history、warm-start bool、state-action试图迁移state-only D、initialization mapping、reward metadata，以及真实公开checkpoint错hash / 不完整schema均拒绝。
- [真实验证工具](verify_cpu_contracts.py) 使用现有 public no_shape checkpoint `bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615`；完整actor / critic / task / sigma / self-task-value moments正常迁移，新2019维D逐元素保持初值、AMP1950统计count0。公开actor与新trainer的固定数学输入均值最大误差0。
- 实际v7读取32个合法 history10+next 窗口。执行**现有 `AMPWorld.expert_batch` 函数体的CPU隔离版**，history逐元素来自对应源帧；没有实例化world / Kit或改真实函数。PD解码与合法clipped-next-pose最大误差 `5.960464477539063e-08 rad`，冻结policy / expert动作同为0。新D在有效动作坐标上有实际非零梯度。
- 六项量的shape / finite已核验：Gaussian mean/std、raw sample、applied action均为 `[6,69]`，behavior log-probability / value均为 `[6]`；likelihood保留原baseline effective57协议，raw69不改。它们是数学输入合同，不是新物理rollout。
- [实际CPU结果](actual_cpu_contract.json) / [日志](actual_cpu_contract.log)、[冷资源预检](resource_preflight.json) 保留来源 / 源码SHA。AMASS manifest重新hash；bank只核对声明与实际大小，不重算1.15GB SHA。无新训练数据写入。

测试中的实际 rig 配置读取触发旧 TorchScript / GLFW无display警告，日志原样保留；没有创建SimulationApp / PhysicsWorld或初始化CUDA。

## Fresh2000 执行配置（尚未执行）

[fresh2000_argv.json](fresh2000_argv.json) 给出完整argv与解析结果。相对已完成的 state-only mixed实际训练arguments，只变 `config` / `output`；它未resume旧state-only权重。预算256×32×2000 = 16,384,000 transitions，两组各8,192,000。当前资源预检已通过，运行GPU由主任务排期。

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/train_paper_amp.py \
  --config configs/diffusion_amp/amp_style_mixed_state_action.yaml \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --output output/diffusion_amp/amp_style_mixed_state_action_v1 \
  --trajectory-source mixed_amass_procedural --num-envs 256 --updates 2000 \
  --rollout-steps 32 --replan-steps 15 --episode-steps 300 \
  --curriculum-mode smooth --curriculum-speed-max-mps 2 --curriculum-accel-max-mps2 1.5 \
  --terrain-extent-m 192 --spawn-margin-m 48 --checkpoint-every 100 \
  --synchronized-replanning --power-source projected_joint_effort_substep_mean \
  --physics-only --seed 7
```

同分布physical pilot、完整22与闭环结果尚未产生；不改变已完成mixed500/1000/2000 evidence或候选选择。`--from-scratch`在此profile可用，但会改actor初始化，属于另一明确实验，不是上面命令。[源冻结](source_freeze.json)。

## 后续实际隔离 pilot（2026-10-02）

实际4更新后恢复到8更新、65536 transitions已完成；两段 fall 为1/0，累计观察到1。2019D、独立AMP RMS、两个Adam、完整RNG/replay/curriculum/physical payload的[995项CPU审计](../state_action_isolated_pilot_v2_review/README.md)通过。checkpoint SHA与第8更新result/metrics已封存。历史实际GPU完整import map缺失仍明确保留，未用CPU探测冒充。新的fresh2000_v3已由主任务启动，其效果和最终评估尚待完成；原上方命令仅为先前方案，实际v3 checkpoint-every500/隔离启动协议另行登记。
