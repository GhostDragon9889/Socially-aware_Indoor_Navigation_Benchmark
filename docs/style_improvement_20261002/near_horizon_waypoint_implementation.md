# 近时域路点优先级：实现与 CPU 冻结记录

## 结论

实现已完成，151 项 focused CPU 测试与本次 9 个文件的 Ruff 门禁通过。**η=.25 是显式工程实验，不是论文作者配置，也没有证据可以替换默认部署。** 本任务未运行 GPU、Kit 或物理仿真。

## 接口与默认合同

- `PaperDiffusionPlanner.plan(..., near_horizon_mass=0.0)`；`clean_waypoint_guidance_gradient(..., waypoint_weights=None)`。
- CLI `--near-horizon-waypoint-mass .25` 只允许 diffusion evaluation + `timed_path` + clean guidance；纯 stdlib parser 在 SDK 前验证。
- `w=(1-η)/K + η/m * I(time<=1s)`；标准十点 `.5,1,...,5s`、η=.25 得 `[.2,.2,.075×8]`，总和 1。**η 是重新分配的混合质量 .25；最终近域两点总质量 .4**，不是 .25。
- 总目标系数仍为 20，每点系数 `[4,4,1.5×8]`；引导梯度、独立候选筛选和整 scene 候选筛选共用同一组权重。scene、peer、dynamics 项及 DDPM noise/t0 语义不变。
- η=0 直接走原 `mean` 分支，不把均匀权重张量乘法当作数值等价替代。训练 profile、sampler、trainer、world、物理和权重文件未改。
- 输出 metadata 明确工程变体、实际 weights、最终近域质量和目标系数；这些是归一化目标系数，不是随机采样概率。

## 实际固定模型 CPU 对照

完整加载 social Diffusion step5000，SHA `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`。两个真实 guide20 capture 上下文、每个 4 actors，100 DDPM、FP32、固定 seed91、CPU 4 threads，共 10.633s。静态 A* 目标由捕获的实际 root/route vertex 重新生成；没有未来真值、姿态作弊或用 A* 替换扩散输出。

**修改前 mean 与当前 η0 的 xy/yaw/vel/velocity/time/sample_states/selected_sample 在两个上下文逐位一致。** JSON 的 `context_default_regression` 专门记录这个 baseline 对照，不能读成 η=.25 与修改前相等。

| 捕获帧 | η | 终点误差 m | timed 路点误差 m | 前 .5s 平均目标进展 m | 前 .5s actor0 进展 m | 最小静态间隙 m |
|---|---:|---:|---:|---:|---:|---:|
| 900 | 0.00 | 0.7456 | 0.3801 | 0.3633 | 0.4838 | 0.9187 |
| 900 | 0.25 | 0.7517 | 0.3504 | 0.3581 | 0.4855 | 0.3514 |
| 4500 | 0.00 | 1.9898 | 1.2232 | 0.2126 | 0.0371 | 1.1032 |
| 4500 | 0.25 | 2.3180 | 1.3957 | 0.2161 | 0.0447 | 0.9734 |

所有四组 obstacle/neighbor 点级代理碰撞率均为 0，但使用 .4m root discs、50 个未来点，不是 native contacts。frame900 的 timed 误差改善伴随平均早期进展略下降及更小障碍间隙；frame4500 停滞 actor0 进展仅 .0371→.0447m，同时终点/平均路点误差变差。**这个小样本未解决停滞，不能称导航成功。** 原始路径段最大加速度仍约 15–23m/s²，不能把 soft 引导当成执行动力学保证。

## 门禁与复跑

- 最终 CPU 门禁：151 passed，44 个既有 Torch JIT/语法 warnings；23.55s。XML：`near_horizon_waypoint_final_cpu_tests.xml`。
- 验证重复索引解析梯度、batch/sample 不变性、现有 parameter.grad/RNG 保持、独立/scene 候选使用相同加权目标、真实 runner 子集 replan 传参及 clocks、所有原 CLI 字段严格相等。
- before 源码逐位 planner 回归覆盖 legacy/clean/clean_variance；clean 两种模式另含 scene joint 筛选与 dynamics 项。
- 纯 resource preflight 实际 9 个资源通过，未导入 Torch/SDK、未反序列化模型；它只验证资源存在、头部/大小和声明，模型真实加载由上述 CPU replay 提供。

最终 focused CPU 命令：

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../IsaacLab-Protomotions/.venv/bin/python -m pytest tests/test_near_horizon_waypoint_weights.py tests/test_diffusion_near_horizon_guidance.py tests/test_amp_near_horizon_cli.py tests/test_diffusion_clean_guidance.py tests/test_diffusion_paper.py tests/test_diffusion_clean_guidance_benchmark.py tests/test_amp_ground_support_cli.py tests/test_amp_curriculum_cli.py tests/test_amp_deterministic_evaluation.py -q --junitxml=docs/style_improvement_20261002/near_horizon_waypoint_final_cpu_tests.xml
```

从仓库根目录复跑纯 preflight（只读，不启动 SDK）：

```bash
CUDA_VISIBLE_DEVICES='' ../IsaacLab-Protomotions/.venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
from CrowdSim.humanoid.application.configuration import parse_amp_args, load_amp_profile
from CrowdSim.humanoid.application.resources import preflight_amp_resources
argv = json.loads(Path('docs/style_improvement_20261002/near_horizon_waypoint_preflight.json').read_text())['argv']
args = parse_amp_args(argv)
result = preflight_amp_resources(args, load_amp_profile(args.config)).as_dict()
assert not any(n == 'torch' or n.startswith(('isaacsim', 'isaaclab', 'omni', 'pxr')) for n in sys.modules)
print(json.dumps(result, indent=2))
PY
```

父任务明确选择进行单变量物理评估时使用下面的待执行命令；保持 guide20、同 actor/Diffusion checkpoint、场景、接触/ground capture、causal adapter 和确定性标志，只有 η0→.25。输出必须使用新目录，禁止覆盖历史工件：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../IsaacLab-Protomotions/.venv/bin/python scripts/train_paper_amp.py --config configs/diffusion_amp/amp_style_amass_backward_penalty.yaml --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt --resume output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt --scene-config configs/diffusion_amp/shared_obstacle_timed_validation.yaml --output output/diffusion_amp/shared_amass2848_near_horizon_eta025_ground_support_200s_v1 --evaluate-steps 6000 --episode-steps 6001 --num-envs 4 --diffusion-guidance-mode trace_clean --route-guidance timed_path --guidance-weight 20 --near-horizon-waypoint-mass .25 --agent-collision-weight 5 --agent-collision-radius .6 --scene-collision-weight 10 --scene-collision-radius .8 --reference-adapter causal_accel --reference-max-speed-mps 1.5 --reference-max-accel-mps2 2 --reference-initial-speed-policy brake --pairwise-contacts --measure-ground-support --physics-only --deterministic-evaluation
```

## 来源与限制

依据 [TRACE 论文](https://arxiv.org/html/2304.01893v1) 的可微目标引导及 pinned 官方源码 `third_party/trace/tbsim/utils/guidance_loss.py` / `models/trace.py`；近时域混合权重由本项目提出，没有标为官方 TRACE/作者超参。printed-paper covariance 与官方 code clean 模式区别沿用现有命名。

下一有效验证是同 baseline 的真实闭环 200s：记录原始/执行路径、idle、连续 route cycles、falls、native wall/peer/floor contacts 及风格指标。较高近域权重可能离开训练分布、减少远域安全余量或造成重规划抖动；等待/让行也可能是合理行为，不强制最小速度。

完整路径、源代码 SHA256、before 快照、测试 XML 和 measured replay SHA 在 `near_horizon_waypoint_implementation.json`；所有历史 artifacts 保留。中途 AST mock 缺少 `guidance_weight` 的测试失败已补齐 fixture，未因此修改业务代码。
