# AMASS根轨迹课程：独立CPU审计

## 结论

最终冻结源码通过25项独立CPU检查，六份业务源码哈希与实现代理的冻结清单完全一致。当前选定配置未发现剩余阻塞问题；允许继续真实物理pilot。审计未启动Kit/GPU、未修改业务源码、未读取34条封存测试捕获。

该课程是工程实验：actor接收实际AMASS根XY任务，Diffusion仍用于独立部署；clip以外的线性任务尾部不是AMASS观测数据，也不是已确认的作者训练协议。

证据：

- [25项独立检查](amass_root_curriculum_independent_integration.json)及[复现脚本](audit_amass_root_integration.py)。
- [冻结哈希绑定](amass_root_curriculum_independent_freeze_binding.json)；[实现冻结清单](amass_root_curriculum_source_freeze.json)记录178项CPU回归与Ruff通过，二者为不同检查范围。
- [独立全量地面边界证明](amass_root_curriculum_independent_floor_bounds.json)及[解析脚本](audit_amass_root_floor_bounds.py)。

## 时序与真实源帧

令clip有T帧，reset源帧为f0，物理episode年龄为a。合法reset为`9≤f0≤T−32`：保留10帧历史和31个真实未来控制转换。新采样API的`future_frames`默认1；512份真实train样本的全部状态tensor、下一DOF位置、motion/frame IDs和共享RNG均逐位等于修改前API。

固定世界平移为`spawn_xy−source_xy[f0]`，只在reset绑定。重规划仅刷新源帧窗口，不用实际root重新锚定。`build_plan`提供51点10Hz记录和151点30Hz消费序列；真实范围内直接读取原30Hz根坐标。见[amass_trajectories.py](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/training/amass_trajectories.py:165)。

任务观测每次从当前`f0+a`直接查询`+15,+30,…,+150`帧，不从旧重规划窗口末点夹取。因此随物理年龄增加，最后目标始终对应当前帧后5秒。见[future_targets](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/training/amass_trajectories.py:189)和[runner接线](/project/EXP/SangforAndSaut/ProtoMotions/scripts/train_paper_amp.py:513)。

奖励目标在真实step完成后消费`local_plan_age+1`对应原始下一源帧。四份实际train clip的最后31个源转换全部逐位相等；31×4×10次任务查询与独立源索引/尾部公式逐位相等。这里的年龄推进是CPU函数测试，没有模拟物理行人。

末真实帧被标记为truncated；超出clip再推进会拒绝。`next_values`在reset之前计算，GAE在truncated处bootstrap并切断跨episode递推，物理fall则取消bootstrap。见[runner奖励和next-value](/project/EXP/SangforAndSaut/ProtoMotions/scripts/train_paper_amp.py:671)及[GAE](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/training/contracts.py:57)。完整既有控制/update外层For AST与修改前相同；新行为位于显式分支及辅助函数。

## 明确的线性尾部与地面边界

只对真实末帧以外的XY任务使用：

```text
v_end = (root_xy[-1] − root_xy[-11]) / (10/30)
XY(end + τ) = root_xy[-1] + v_end × τ + fixed_translation
```

`[-11]…[-1]`包含11个位置、10个30Hz间隔。真实源帧完全不变；不向actor提供未来3D姿势，不在clip末端之后继续物理episode。[尾部实现](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/training/amass_trajectories.py:100)明确记录hold/linear_velocity选择；旧hold证据文件保持原样。

独立NumPy float64证明读取全部2,848个train segment，重新核对bank全SHA，覆盖409,352个合法f0与256个出生点，等价覆盖104,794,112组组合。对每个f0检查整个真实源后缀的XY极值及末端线性延伸5秒端点，使用仿射平移极值证明完整出生网格，不抽样、不过滤、不重新采样fallback。

| 项目 | X | Y |
|---|---:|---:|
| 全组合命令最小坐标(m) | 27.688319 | 27.062249 |
| 全组合命令最大坐标(m) | 127.585424 | 129.898893 |
| 距允许下界1m的剩余量(m) | 26.688319 | 26.062249 |
| 距允许上界191m的剩余量(m) | 63.414576 | 61.101107 |

网格出生点为48…108m、步长4m，地面192×192m、根命令margin1m。证明还包含episode300cap以后的真实后缀，因此比实际episode范围保守。末端估计速度最大3.96216m/s；线性命令没有额外速度截断。

边界证明仅保证参考根XY命令位于地面余量内，不保证实际身体碰撞体、root跟踪误差、脚支撑或不跌倒。

## 保存、恢复与修复记录

保存完整绑定motion/f0/固定offset/末端速度/剩余帧/episode年龄/上次计划年龄、统计计数及共享corpus RNG。恢复严格检查schema/config/sourceSHA、形状/dtype、endpoint速度、源phase与151点计划。`weights_only=True`往返后全部逻辑状态/计数一致，恢复后的完整5秒任务及下一批实际reference/RNG逐位相等。bind/replan/task查询不额外消耗随机数。

预审确认并由实现代理修复：

1. 新AMASS source缺显式`trajectory_training`声明现在在preKit阶段拒绝，避免Kit启动后的KeyError。旧Diffusion/procedural未声明profile仍接受。[入口门禁](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/application/configuration.py:347)。
2. reference IDs恢复必须为int64，拒绝浮点等值IDs。
3. AMASS来源元数据已改为固定episode平移和当前source-phase查询；不再误记为PACER随机速度/实root重新锚定。[元数据](/project/EXP/SangforAndSaut/ProtoMotions/scripts/train_paper_amp.py:273)。
4. newly-reset actor的`age=0`恢复必须标记`plan_age=replan_steps`；否则拒绝，确保下一控制步重新规划，避免使用上个clip旧计划。[恢复门禁](/project/EXP/SangforAndSaut/ProtoMotions/CrowdSim/humanoid/training/amass_trajectories.py:293)。

逻辑恢复不包含PhysX接触/solver缓存，不能据CPU逐位相等宣称物理重放逐位等价。真实4+4更新恢复pilot、动力学、奖励幅值和训练效果由独立GPU执行报告确认。

后续真实4+4更新已完成，且[不可变checkpoint独立CPU审计](amass_root_actual_pilot_independent_review.md)166项通过。该结果单独记录执行段与累计计数，不改变上述纯CPU时序/边界证明的范围。

## 可复现命令

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/audit_amass_root_integration.py \
  --output /tmp/amass_root_independent_integration_new.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/audit_amass_root_floor_bounds.py \
  --output /tmp/amass_root_independent_floor_bounds_new.json
```

本轮frozen train manifest SHA为`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`，bank SHA为`46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`。未来源码或数据变更须重新绑定审计；旧证据不应覆盖。
