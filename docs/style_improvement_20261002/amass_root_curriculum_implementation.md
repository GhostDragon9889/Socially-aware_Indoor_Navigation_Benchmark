# AMASS root XY 训练课程：冻结实现与 CPU 验证

## 结论与范围

已实现显式 `--trajectory-source amass_root`，仅用于训练。**178 项 CPU 测试通过，9 个 Python 文件 Ruff 通过；独立复核另有25项通过。真实 GPU/Kit 的4+4步训练/恢复探针交由主任务执行，本报告没有宣称仿真成功或行为改善。**

源码与测试的精确路径、大小和 SHA256 见 [source_freeze](amass_root_curriculum_source_freeze.json)，证据哈希与运行参数见 [implementation.json](amass_root_curriculum_implementation.json)。业务源码已冻结；保留了每个修改文件的 before 副本。完整物理控制 `for step` AST 与修改前完全相同，SHA256 为 `1adbfe51b09a2311b3e450925cface7b4f81a1fbb122dd945838d16a9272a74f`。原负向方向奖励配置未修改。

NavIsaacLab IV-B 用未来 XY 路径条件化下层 AMP 控制器；论文描述不能证明作者采用本次 AMASS root 训练命令课程。[NavIsaacLab 原文](https://arxiv.org/html/2606.26265v1#S4.SS2) TRACE/PACER 的训练描述使用生成的轨迹条件；本次按同一动作片段真实 root XY 构造命令，是明确标注的工程变体。[TRACE 原文附录](https://arxiv.org/html/2304.01893v1#A2.SS3)

## 实现合同

- 默认 `diffusion`/`pacer_curriculum` 保留原调用、采样张量、随机数序列和控制循环。新 profile 的所有规范化 `TrainingConfig` 值与负向方向奖励基线相同，专家窗口 sampler 明确禁用；仅命令来源、合法初始相位和真实片段结束规则改变。
- 初始相位 `f0 ∈ [9,N-32]`，保留10帧真实 history 和31个后续真实30Hz步，最短合法 episode 为31/30秒。全部2,848段动作仍可采样，没有隐藏筛选或概率重分配。
- 每次 reset 沿用原物理 reset 一次，绑定实际 motion ID/frame。世界平移只计算一次：`reset_worldXY-sourceXY[f0]`。后续目标使用固定平移，实际行人落后不会在 replan 时消失；不把未来3D姿势、source ID、关节目标或参考速度给 actor。
- 消费151个原生30Hz XY点，5秒时域；原始区间等于真实 source 帧加固定平移。51个10Hz点仅保留为规范表示与验证。目标奖励的下一1/30秒点逐帧精确；actor 的10个未来 XY 点始终从当前实际 episode 相位查询 `.5…5s`，不被较旧 replan 窗口尾部夹住。
- 仅真实 source 之外，使用显式 `linear_velocity` 命令延拓：`v=(root[-1]-root[-11])/(10/30)`。这是11个真实位置、10个时间间隔的末端速度估计；延拓不是 AMASS groundtruth。类/World 的默认 `hold` 保留，新独立 profile 选择线性尾部。
- 在最后一个真实 source 帧截断 episode，不执行或奖励填补帧。保留原 PPO 时间截断语义：reset 前实际末帧观测计算 next value；fall 是终止，两者单独计数，GAE 不跨下一 episode。
- checkpoint 保存并严格验证 source/config/RNG、IDs/frames、固定平移、末端速度、真实剩余步数、episode/plan clocks 和实际 native151计划。刚 reset 的 age0 actor 必须 `plan_age==replan_steps`，拒绝恢复后误用旧轨迹。逻辑恢复不承诺重建 PhysX solver/contact cache。

## 实际数据 CPU 证据

[最终线性尾部 proof](amass_root_linear_cpu_proof_frozen_resume_guard.json) 绑定最终6个业务文件哈希，没有 SDK/GPU/物理 rollout：

| 项目 | 结果 |
| --- | --- |
| 实际训练动作 / 总帧 | 2,848 / 523,272 |
| 全部合法 reset 相位 | 409,352 |
| 原生真实前缀 / 下一30Hz奖励目标 | 与 source+固定平移 bitwise 相同 |
| canonical51 / native151 | canonical 等于 native 每第三点 |
| bind/replan/未来任务查询 | 不消耗 corpus RNG |
| 地面 / origins 解析门禁 | `[0,192]²`、root margin1m；256 origins 各轴48…108m，全部合法相位及末端5秒延拓通过 |
| 所有可能命令 root XY范围 | x27.688…127.585m，y27.062…129.899m |
| 末端估计速度 | 均值.39895m/s，p99 1.92405m/s，最大3.96216m/s |
| 一次固定 seed 每clip计划 | 142,400个未来点；55,041在 source 之外并延拓（38.6524%），held0 |

[独立 floor 证明](amass_root_curriculum_independent_floor_bounds.json) 与 [独立集成复核](amass_root_curriculum_independent_integration.json) 另外确认边界、真实相位、默认 RNG、恢复与完整 For AST。地面证明只覆盖命令 root 点，不能当作全身碰撞、稳定站立或实际漂移安全保证。

### 为什么尾部假设必须明示

不可覆盖的 [hold v1 实际 source 审计](amass_root_reference_cpu_proof.json) 发现：1,646段在 parent 动作结束前裁切；1,283段即使最早合法 reset 也没有完整5秒未来。uniform-clip/legal-frame 分布下，期望真实剩余时长3.4122秒。

| hold 命令分析 | +.5秒 | +1秒 | +5秒 |
| --- | --- | --- | --- |
| 初始相位未来点在 source 外 | 0% | 0% | 80.13% |
| 所有真实 pre-action 相位，按 transition 加权 | 13.68% | 28.33% | 91.19% |

初始10点平均 source 外占41.34%。最后真实一秒平均速度.4475m/s，最后真实间隔平均.3929m/s，50.77%大于.1m/s；不能把裁切末端当作真实停步。线性延拓是对此数据风险的显式课程假设，仍可能把转弯或停步错误延续；没有暗加最小速度、裁切速度或伪造银行帧。上述选择依据训练源分析，未读取或使用34条 heldout 测试捕获结果。

10Hz点线性插值与实际30Hz root曲线曾有最大.08264m、p99 .00502m误差，因此 AMASS 模式消费151原生点；其他模式保持10Hz语义。

## 门禁与下一步

[最终测试 XML](amass_root_curriculum_final_cpu_tests.xml)：178 passed，44项现有警告，2.60秒。测试包含 before 采样/RNG逐张量对照、真实 reset/step 方法的 CPU 合同、异步 reset 固定 source 时钟、下一帧精确 XY、clip-end bootstrap、非法恢复 payload/刚 reset 旧计划、preSDK missing-profile 拒绝及旧 CLI 字段一致性。mock 合同不当作仿真证据。

[预检](amass_root_curriculum_preflight.json)：8项资源通过；没有导入 Torch/SDK或反序列化模型。冻结后的主任务 fresh GPU pilot 命令：

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/train_paper_amp.py \
  --config configs/diffusion_amp/amp_style_amass_root_curriculum.yaml \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --trajectory-source amass_root --num-envs 4 --updates 4 --rollout-steps 32 \
  --episode-steps 300 --checkpoint-every 1 --terrain-extent-m 192 --spawn-margin-m 48 \
  --physics-only --synchronized-replanning \
  --output output/diffusion_amp/amass_root_linear_native_pilot_v1
```

恢复探针同参数增加 `--resume output/diffusion_amp/amass_root_linear_native_pilot_v1/last.pt`；主任务再决定完整训练。训练的 actor 部署依然使用 learned Diffusion，没有 AMASS未来查询。真实身体跟踪、跌倒/接触/滑移、命令完成率与风格质量必须经过实际物理评估；较低任务奖励可能反映固定世界目标终于保留了落后误差，不能只按奖励均值判定退化或成功。
