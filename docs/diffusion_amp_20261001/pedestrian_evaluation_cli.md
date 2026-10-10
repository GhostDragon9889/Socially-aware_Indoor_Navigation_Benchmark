# 实际行人 rollout 的 CPU 评估命令

[脚本](../../scripts/evaluate_paper_pedestrians.py) 读取 `train_paper_amp.py` 导出的 `physical_states.npz` 及同目录 `result.json`，无需启动 Kit、GPU、加载策略权重或安装其他包。

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/evaluate_paper_pedestrians.py \
  --physical-states output/diffusion_amp/实际评估目录/physical_states.npz \
  --manifest data/diffusion_amp/amass_manifest.json \
  --output-dir output/diffusion_amp/实际评估目录/pedestrian_metrics
```

`--physical-states` 可简写为 `--states`；非相邻的 runner 结果用 `--runner-result` 指定。未指定 output 时，写到 capture 目录下的 `pedestrian_metrics`，不会覆盖 runner 的 `result.json`。默认 manifest 根据脚本所在仓库定位，可以从任意工作目录运行。

## 导出合同核对

只读核对当前 `scripts/train_paper_amp.py` 的导出逻辑：

- `.npz` 保存 `root_pos[T,B,3]`、`body_pos[T,B,24,3]`、unit XYZW `root_rot[T,B,4]`、`dof_pos[T,B,69]`、同刻 `target_xy[T,B,2]`、integer `episode_ids[T,B]`、boolean `reset_mask[T,B]`；另有 `terminated` / `truncated`。
- 状态记录在物理 step 之后、reset 之前；`reset_mask = world.age == 1` 表示新 episode 的第一个 post-step frame，使用 **before_frame** 语义。
- 新字段 `global_goal_xy[T,B,2]`、非负 `plan_endpoint_goal_error[T,B]` 可选；旧 capture 缺少时保留未知，存在时严格校验形状与 finite。
- result 的 `state_capture` 明确声明 common SMPL 的 24 个 body 名称、30Hz、world meters 和 XYZW。root 位置必须与该名称对应的 Pelvis body 一致。
- evaluator 使用 `np.load(..., allow_pickle=False)`，验证所有必需字段、形状、finite、boolean/integer 类型和 metadata。缺 body 名称、30Hz 或 reset 定义会报错，不能靠猜 native body 排序补齐。
- 输出会比较 reset mask 与 episode ID 标识的边界；非零 disagreement 应先检查记录时序，不能隐藏这一问题。

## 默认、固定取样与统计

| 项目 | 设置 |
| --- | --- |
| train 参考 | AMASS train split，2,048 个实际姿态，独立 generator seed=7 |
| test 参考 | AMASS test split，2,048 个实际姿态，独立 generator seed=7 |
| 参考取样 | 当前 `AMASSCorpus.sample_reference`：均匀选 manifest 动作，再从安全历史/下一帧范围选控制时刻；保存实际 motion/frame ID |
| generated | 最多 2,048 个姿态，对 `T×B` 的 time-major 展平序列作端点包含的等间隔整数采样；保存 flat/frame/agent index 与每人数量 |
| pose filter | 不按跌倒或重置筛选姿态；包括 reset-first 与 terminal frame，条件明确写入报告 |
| cluster | K=50/100，仅 train 拟合，kmeans++ seed=7、最多 200 iterations，tolerance=1e-7；保存固定中心 artifact 与 hash |
| score | train/test/generated 都分配到同一份训练中心，test/generated 不重新拟合 |
| Gaussian MMD | generated↔test；另给 train↔test 参考值；45D 单帧表示、固定 σ=1.0、biased V-statistic，默认 chunks=512 |
| rollout metrics | 使用全部物理帧；不对路径、跌倒与跟踪指标作稀疏取样 |
| collisions | 只读取实际记录的 contact mask；当前 runner 没有这些字段时输出 `None`，不自动代换成几何代理 |

参考样本数、generated 上限、seed、CPU threads、chunk size 和 kmeans iteration 可显式覆盖，均记录在输出。`--ground-height` 默认 0m，适用于当前明确的平面；新场景不能把不同 ground 定义直接当作此协议。节点与尺度定义见 [评估核心协议](pedestrian_evaluation.md)。

## 目标、路线与碰撞代理的独立统计

新增 `goal_condition_metrics` 分别记录：

- Physical root 到 NPZ `global_goal_xy` 的逐帧 XY 距离：均值、RMS、分位数、首末帧各人距离。
- `plan_endpoint_goal_error`：持有的 diffusion plan 最后 XY 点到 conditioning goal 的残差；每个物理记录帧加权，未假设每帧都重新生成计划。
- 仅同 episode、同一个目标时的相邻帧距离减小量；目标变更和 reset 跳跃不算推进。
- runner 声明的 `route_vertices_reached`（各人/总计）、`global_path_failures` 和 `global_path_method`。

当前 obstacle fixture 的 `ReconstructionScene.route_goal` 经 A* 提供 5m arclength lookahead。因而字段名 `global_goal_xy` 的内容是 sparse planner conditioning point，**并非完整 route vertex/global destination**。它可随 replan 改变；A* 查询失败还可把 goal 设为当前 root。跟踪误差小、goal distance 小或计划终点 residual 小均不能自动判定完整路线到达；`global_arrival_rate` 保持未知。route vertex advancement 是独立 runner 计数，不转换成作者导航成功率。

若 `result.evaluation_scene` 包含显式 fixture 描述，新增 `geometry_proxies`：

- 使用同一 CPU `ReconstructionScene.occupied` 的 axis-aligned cuboid XY footprint，加上 bounds 外 root-point 标记；记录两项分别的 frame-agent 数、参数和总事件统计。它忽略全身形状/垂直接触。
- 仅当 `physics.isolated_environments == false` 明确声明 shared world 时计算 XY circle overlap，默认半径 0.3m，`--proxy-agent-radius` 可覆盖并记录。没有 shared 声明时不猜 agent 之间是否共享碰撞空间。
- 两个代理与 `rollout_metrics.map_collisions/agent_collisions` 分开；actual PhysX pair contact mask 缺失依然为 `None`。这些 fixture/route/A* 均为本地验证条件，不构成论文八场景协议。

`AMPWorld.reset` 将 age 归零并增 episode ID，`step` 把 age 加一；runner 保存 post-step 状态和 pre-step clone 的 episode ID，随后才 reset。因而 before_frame mask 与 ID 在当前 lifecycle 中一致；输出 disagreement 帮助复核任何后续时序变化。

源 manifest/package hash 及 split 源动作无交集检查保留。默认只校验源包 size/metadata 和声明的 hash；要重新计算全部多 GB 数据包 hash，显式加 `--verify-package-hashes`，报告会区分声明值与本次内容验证。

## 产物

- `evaluation.json`：输入 NPZ、runner report、实现 hash，manifest gate、参考与生成的精确采样 ID、所有参数/数量/未知项、聚类与 MMD、完整序列物理指标。
- `report.md`：中文指标表及解释边界。
- `train_pose_clusters_k50.json`、`train_pose_clusters_k100.json`：带训练来源和 embedding spec 的固定 reference centers；可以受限 JSON 载入后重复 scoring。

状态 `evaluated` 仅表示 CPU 分析完成。报告不判断稳定/收敛，不声称作者参数、权重、2,848 动作清单、八场景或 Table II–III 一致；navigation success contract 缺失时 success rate 保持未知。

## 本次接口验证

编写时尚无 runner 完成后的 `physical_states.npz`。因此未冒称实际四人 200 秒评估；只用明确 `--input-kind synthetic_interface` 的临时输入验证接口：16 帧 × 4 人，重复真实 AMASS 姿态，解析根移动与一次 80m reset jump，零物理仿真。train/test 两份参考确实来自本地 AMASS，各 2,048，seed=7。

端到端命令已从 `/tmp` exit 0。根路径核对为 `[0.14, 0.15, 0.15, 0.15]`m，没有把 80m reset 跳跃加入；reset/episode ID disagreement=0，未记录碰撞保持未知。额外验证缺名字、错 Hz、错 world 单位、错 reset 语义、body shape 错误和 root/Pelvis 错序均拒绝。

完整接口验证见 [evaluation_cli_verification.json](evaluation_cli_verification.json)。[合成接口报告](evaluation_cli_interface/report.md) 标明 synthetic；其统计不能替代最终实际生成姿态结果。根 agent 在真实 rollout 完成后运行上述命令即可产生实际评估。

新目标/场景接口另通过 **8 项 CPU 回归**（[JUnit](evaluation_cli_tests.xml)）：0.1m 的物理 target tracking、较远 sparse goal distance 与 2m endpoint residual 独立；goal/reset 跳跃不计推进；可选字段缺失保持未知；坏形状/负残差拒绝；shared circle/cuboid/outside-domain 代理有值而 actual contacts 仍未知。

已再次从 `/tmp` 完成默认 2,048 train/test 真实 AMASS 参考的端到端调用，capture 仍明确为 synthetic interface（非真实物理）。新验证记录见 [evaluation_goal_scene_verification.json](evaluation_goal_scene_verification.json)；[新合成报告](evaluation_cli_goal_scene_interface/report.md) 不能替代 root 待执行的四人 200 秒结果。
