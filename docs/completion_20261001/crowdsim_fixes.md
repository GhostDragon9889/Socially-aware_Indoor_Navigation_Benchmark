# CrowdSim 训练、评估、采集与偏好奖励修正（2026-10-01）

参考：只读 `/project/EXP/NavIsaaclab2.0` 对应代码。当前工作树保留前一轮 IsaacLab 四元数、行人子集 reset、地图、SMPL 与入口修复。本轮涉及训练与学习合同；未启动 Kit、修改环境、提交或暂存。

## 已修复的实证问题

| 编号 / 严重性 | 触发与旧结果 | 当前修复与证据 |
|---|---|---|
| F12 / P2 | PointGoal 在边界裁剪目标，实际距离与朝向偏差脱离课程配置。seed42，10k 初始状态，距离配置 2–4m、heading 0–15°：原仓库 1875 距离小于2m，1956 heading超过15°，最大93.18°。 | `src/crowdsim_ppo/point_goal_env.py:73` 先采样可行位移，再在可容纳两个端点的矩形采样起点；heading使用最终实际目标方向。当前上述两种违规均0，最短2.000109m，最大14.992641°。覆盖接近及恰好地图对角线、不可行距离、子集reset。 |
| F13 / P2 | 变长偏好轨迹 padding 参与 attention 及 reward 求和，同一短轨迹的分数随同批最长轨迹变化；validation 仍用 dropout train mode；仅两对样本时旧拆分无训练样本；首轮accuracy=0不保存。 | `CrowdSim/pref/pref_train_reward.py:68,180,190,219` 传有效帧mask，保留非空训练/验证集合，eval/no_grad验证、恢复train，并保存首个有效epoch。`reward_model_utils.py:715,768,780` 屏蔽 padding key 与输出。真实网络短轨迹单独分数 .8937175，原混合批次1.9207742；当前混合批次.8937178，差2.38e−7。真实两epoch训练验证mode序列准确、权重有限且可保存。 |
| F16 / P2 | reached、collision、timeout、stuck可同时发生，直接相加得到重复episode分母；同时到达并碰撞被算success，并进入课程及成功轨迹集合。 | 新增 `CrowdSim/ppo/episode_metrics.py:6`：episodes仅按done计数，事件按done gate，success=reached且无collision。两个训练loop、logger、finetune summary、eval outcome、collector与PointGoal消费者统一。reached+timeout同一done原分母2/current1；真实两个训练loop的4done、2安全success给出.5成功率，raw timeout率1。事件率允许重叠，不要求和为1。 |
| CT01 / P1 | 普通PPO没有nonfinite保护，旧KL分支只查NaN梯度，漏掉Inf。有限forward、backward钩子注入Inf时，两份PPO的普通/KL四分支均无报错，权重非finite且Adam新增11组污染状态。 | `CrowdSim/ppo/ppo_policy.py:884,1045` 与独立副本在step前检查finite loss，`clip_grad_norm_(error_if_nonfinite=True)` 检查梯度。当前四分支RuntimeError、权重有限且逐项不变、Adam state保持空。NaN/Inf loss同样拒绝；现有单样本/3样本尾批测试证实仍消费每个样本，未引入静默跳过。 |
| CT02 / P1（保存中断时） | 直接torch.save覆盖唯一checkpoint，磁盘错误/进程中断可破坏旧文件。 | 两份PPO和两种偏好训练保存改为同目录临时文件、flush/fsync、atomic replace。CrowdSim复用 `protomotions/utils/checkpoint.py`；可独立安装的src使用新 `src/crowdsim_ppo/checkpoint.py:10`（仅stdlib与torch）。实际序列化写入部分字节再OSError：旧archive逐字节不变、temp清理、旧权重/step/extra仍能恢复。通用helper的SIGKILL/rename/rank测试由dependencies agent负责。 |
| CT03 / P1（headless视觉输入） | 第一帧及机器人/humanoid reset后先读depth，可能得到reset前相机图像；`.get('mean_action', outputs['action'])` 在仅提供mean_action时先求值默认表达式并KeyError。 | `CrowdSim/crowd_sim.py:197`、`CrowdSim/ppo/train_ppo.py:386`、`train_ppo_finetune.py:265` reset后先render再读camera。两个训练loop的动作选择改为条件索引。真实Torch更新使用CPU生命周期fixture，首帧相机10、robot reset后100，两个训练loop正确计数并更新有限权重；仅mean_action可运行。真实渲染/物理由root另测，此fixture不证明仿真效果。 |
| CT04 / P2（num_neighbors=0） | PointGoal.step对空tensor做min异常；PPO向量层仍加一个neighbor统计块，encode要求不存在的channels，K0实际输入5却期望10，完整训练链不可运行。 | PointGoal无邻居最短距离设infinity；两PPO K0省略neighbor统计块并允许None/empty channels，正K维度保持原合同。实际独立CPU训练4步→原子保存→完整resume新目录→3episode评估通过。 |
| CT05 / P2（入口边界） | 空rollout或num_envs=0可使训练循环无法前进；save_interval=0在独立入口除零；total_steps=0被 `or` 忽略；已达到预算的resume返回不存在的输出；离线reward epochs=0返回不存在的best checkpoint；非finite输入可更新并保存坏偏好模型。 | 两buffer构造拒绝空rollout；训练/评估参数明确验证；0保存间隔关闭周期保存；完成resume仍发布加载的完整checkpoint；偏好loss/grad及learningrate finite校验，离线teacher normalization使用population std并拒绝nonfinite teacher数据。真实在线/离线Inf输入停止，权重不变且无坏checkpoint；真实离线一epoch保存有效模型。 |
| CT06 / P2 | Q键仅改变模拟运行flag，app仍open，主入口的循环继续；Kit finally close可能先于Python输出异常stack导致失败原因消失。 | `CrowdSim/crowd_sim.py:89,110,197` 循环同时尊重app和simulation flags，main异常先print_exc再finally close。CPU fixture验证Q终止与close前保留原始异常。 |

## 证据与测试

- 原仓库/当前真实 CPU 数值对照：[crowdsim_reproductions.json](crowdsim_reproductions.json)。复现代码：[reproduce_crowdsim_fixes.py](reproduce_crowdsim_fixes.py)。原仓库始终只读；网络和优化器为实际Torch实现。
- 整个训练辖区30个Python文件、9962行完成语法解析和原仓库hash比对：[crowdsim_training_source_coverage.json](crowdsim_training_source_coverage.json)。此清单表示静态覆盖，不能等同每个路径实际执行。
- 最终必要回归 **175 passed / 6.89s**：[crowdsim_tests.xml](crowdsim_tests.xml)。包括整个 `CrowdSim/tests` 及PointGoal、有限值、单样本/尾批测试。

```bash
PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m pytest \
CrowdSim/tests tests/test_point_goal_env.py tests/test_point_goal_sampling.py \
tests/test_ppo_finite_and_resume.py tests/test_ppo_singleton.py -q \
--junitxml=docs/completion_20261001/crowdsim_tests.xml

PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
docs/completion_20261001/reproduce_crowdsim_fixes.py
```

## 兼容性与验证范围

- 原有正邻居数量的PPO结构不变；旧K0结构本来无法完成encode/update，修复后的K0向量层宽度由错误的10改为5，旧K0权重不能直接加载。
- 偏好网络权重key和tensor形状不变，`valid_mask`可选，固定长度offline模型继续直接调用。私有 `_build_batches` 返回值新增mask，已检索并修正本项目全部调用方。
- PointGoal起点分布现在随所采样位移的可行区域变化；距离/朝向合同恢复，但旧“uniform起点加clamp目标”分布无法同时保留。
- 安全success排除collision；raw reached与timeout等诊断事件仍可重叠。外部dashboard应使用新的 `outcome/success_rate`，不要相加事件count当作episode数。
- 恢复模型、optimizer、step与curriculum已经覆盖；这不是恢复相同随机数或完整环境轨迹的逐bit续训。不同配置训练的checkpoint仍应传匹配配置，load维度检查会拒绝不兼容结构。
- 这里未验证GPU大规模训练、长时间收敛、偏好标签质量或论文指标复现。CPU相机时序fixture无物理、无真实RTX渲染。Root负责真实Kit/物理/SMPL的独立验证与最终全套测试。
- 175测试中的警告含Torch/Chumpy弃用，以及2个部分构造nav fixture的destructor缺少 `_trajectory_timestamp_file`；已通知root，当前报告不把测试PASS表述为warning-free。

## 本轮实际修改清单

业务：

```text
CrowdSim/crowd_sim.py
CrowdSim/flow/collect_ppo_buffer.py
CrowdSim/tools/eval_policy.py
CrowdSim/pref/pref_train_reward.py
CrowdSim/pref/reward_model_utils.py
CrowdSim/pref/train_offline_reward.py
CrowdSim/ppo/episode_metrics.py (新增)
CrowdSim/ppo/train_ppo.py
CrowdSim/ppo/train_ppo_finetune.py
CrowdSim/ppo/ppo_policy.py
src/crowdsim_ppo/point_goal_env.py
src/crowdsim_ppo/ppo_policy.py
src/crowdsim_ppo/checkpoint.py (新增)
src/crowdsim_ppo/train.py
src/crowdsim_ppo/evaluate.py
```

测试：

```text
tests/test_point_goal_env.py
tests/test_point_goal_sampling.py (新增)
tests/test_ppo_finite_and_resume.py (新增)
CrowdSim/tests/test_preference_padding.py (新增)
CrowdSim/tests/test_offline_reward_training.py (新增)
CrowdSim/tests/test_episode_metrics.py (新增)
CrowdSim/tests/test_policy_atomic_save.py (新增)
CrowdSim/tests/test_inference_render_timing.py (新增)
CrowdSim/tests/test_training_lifecycle.py (新增)
```

`crowd_sim.py` 及src train/evaluate含root同时协调的先前入口/默认配置修改；本清单包含我们实际改动的文件，不能归属整文件全部diff。通用checkpoint helper、独立包pyproject/配置/README、采集恢复协议及其他agent的修改在各自清单中。
