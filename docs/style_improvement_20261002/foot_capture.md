# 可选真实足部支撑与 COM 采集接口

## 当前状态

`CrowdSim/humanoid/foot_capture.py` 是独立、可选的只读采集模块。它没有接入正在运行的训练、没有修改 world / runner / trainer，也不会启动 Kit、创建新的物理 SimulationView、推进物理或写入人体姿态。新增 29 项 CPU 契约测试和原 foot_support 的 28 项测试全部通过（共 57）；Ruff 通过。新进程 import 没有加载 isaacsim、isaaclab、omni 或 pxr。

CPU 假 SDK 测试验证的是合同、索引和时钟，不是实际足部传感器或稳定步态。模块 manifest 明确保留 `native_capture_probe_passed=False`。现有独立 foot probe 中 neutral stance 真实倒地不能在这里转换成通过；实际 standing / airborne / wall-only 三种原生 QA 和新的完整捕获需独立完成。

## 调用合同

在已初始化并提前启用 contact reporting 的真实 world 中，由调用方明确解析当前资产的四足 rigid-body 路径，传入真实 USD stage 及 floor 根：

```python
from CrowdSim.humanoid.foot_capture import FootCaptureAdapter

old_peer_monitor = getattr(world.simulator, "pedestrian_contacts", None)
capture = FootCaptureAdapter.from_world(
    world,
    actor_foot_paths=exact_actor_foot_paths,  # 每人 L/R_Ankle、L/R_Toe 的完整 rigid 路径
    stage=actual_stage,
    floor_roots=["/World/ground"],          # 明确指定 floor，不包含墙或全部 static
    contact_reporting_enabled=True,       # 必须在人体 PhysX 初始化前已启用
    peer_monitor=old_peer_monitor,         # 原 static/peer monitor 或 None
)
world.simulator.pedestrian_contacts = capture  # 调用方显式替换唯一已有 hook

capture.prepare_control_step(actual_runner_control_step)
world.step(applied_actions)                 # 仍然只调用原有的这一个 step
sample = capture.measurements()            # 必须在 world.reset 前发布/保存

# 当正常控制的采集已经发布，且调用方明确处理新 episode 边界：
capture.reset_episode_boundary()
```

这里的示例没有代替配置代码设置 contact reporting，也不会修正 floor 或姿势；传入错误路径/缺少 enabled CollisionAPI/原生传感器 metadata 不一致会拒绝。`from_world` 调用现有 `resolve_native_contact_factory`，验证 Lab 注册 manager 的公开 view 与 active view 相同；不使用无关的旧 Isaac Sim singleton。floor 解析必须返回真实 enabled CollisionAPI 后代，不能把无物理 shape 的 Xform 当成接触目标。

`PaperPowerSimulator` 现有无参 `begin_control_step()` 和 `sample(dt)` hook 被适配为有明确控制计数、物理计数及 episode 的采集。真实控制计数由 runner 显式 `prepare_control_step` 传入；不从某个 actor 的 age 或私有模拟时钟猜测。每个 30Hz 控制必须完整包含四个 120Hz 子步，在既有 `physics step + scene.update` 后采样。原 peer monitor 每个子步仍只 sample 一次，不重复采样或另建推进循环。

如果一个控制被打断，`measurements()` 不发布部分/零填充数据，普通 reset 也拒绝吞掉该部分。显式 `discard_interrupted_control(reason=...)` 返回 `valid=False`、已采样 state/force 和实际计数的失败证据；调用方负责保存该失败记录后再恢复。正常 episode 边界必须先发布旧 episode 的采集，再显式重置采集状态。

## 产出及定义

| 字段 | 形状 / 含义 |
|---|---|
| `foot_floor_normal_force_w` | `[4,B,4,G,3]`，四子步 × 人 × L_Ankle/L_Toe/R_Ankle/R_Toe × 明确 floor colliders × 世界法向力向量，N |
| `side_contact_substeps`、side/load/support 系列 | 原 FootSupportMonitor 定义；严格 `> threshold_n`，不包括墙、peer、自接触或摩擦 |
| `body_link_pos_w`、`body_link_quat_w`、`body_link_lin_vel_w` | `[4,B,24,3/4]`，verified native → common body 名称双射；quaternion 为 XYZW，线速度为 link/body-origin |
| `body_com_pos_w` | `[4,B,24,3]`，真实 live body COM 世界位置，不是关节原点代理 |
| `local_com_pose_b` | `[4,B,24,7]`，`root_view.get_coms()` 的 link-local COM 位置和惯性主轴姿态 |
| `body_masses_kg` | `[4,B,24]`，当前 `root_view.get_masses()` 的真实质量，按同一 body 名称重排 |
| `mass_weighted_world_com` | `[4,B,3]`，使用上述 live COM 与质量计算的 whole-body COM |
| `com_composition_max_error_m` | 四子步的最大绝对坐标分量误差；验证 `p_link + R_link * local_COM` 与 live COM，默认 ≤ 1e-4 m |
| `body_origin_tangential_speed_proxy_mps` | body-origin 速度在明确单位地面法向的切平面投影；不是接触点滑动或摩擦力 |
| `physics_step_before`、`physics_step`、`episode_ids`、`control_step` | 真实公开 Lab submitted-step counter、当前 world episode、调用方真实控制计数；不冒充 PhysX engine 时间 |

静态 COM/mass API 可能使用不同 tensor transport；先验证其有限值和 shape，再把值复制到 live state 的 dtype/device。几何不重新估计。每次 state/contact 读取前后实际 physics counter 和 episode 必须完全相同；四子步、发布前的额外步、缺步、重复步、episode reset、COM 坐标不一致或非有限值都会拒绝。

力向量是每个 sensor/filter 的 normal-force vector sums。对 ankle/toe 向量取模后的 side load，并不是非平行接触面上所有单个 contact patch 法向力的标量总和。support union、四子步比例和滑速代理仍沿用已有 FootSupportMonitor 的显式定义；不能称为论文未知的作者聚合指标。

AMASS 没有被赋予虚构的地面接触、摩擦或真实 COM。`reference_true_com=None`；以后若要与参考比较，必须另提供匹配参考人体质量/COM 的可信数据和独立合同。

## SDK 证据及验证

Pinned SDK 的 `SimulationContext.get_physics_step_count()` 文档为每次 `step()` 递增的 monotonic counter，见 `../IsaacLab-Protomotions/source/isaaclab/isaaclab/sim/simulation_context.py:324`。`body_link_quat_w` 在 pinned PhysX articulation data 为 XYZW；`body_link_lin_vel_w` 为 actor-origin velocity，`body_com_pos_w` 为 COM 世界位置，见 `../IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py:1341`、1355、1383。

CPU 回归包含四人/四子步、打乱 native sensor/filter/body 顺序、独立 +90° Z 旋转 COM offset、真实 counter 与同 episode 合同、buffers 后续覆写不会污染捕获、仅 peer 单次采样，以及拒绝缺步/重步/reset/读函数误推进/非有限通道/错 COM/错 dtype/零质量/控制 dt/不合法 counter/未发布部分。产物源 SHA 和测试 XML 见 [CPU 验证 JSON](foot_capture_cpu_verification.json)。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python -m pytest -q \
tests/test_foot_capture.py tests/test_foot_support.py
```

现已接入显式 eval-only `--measure-ground-support`，默认关闭，独立 `ground_support.npz` 不改变原 `physical_states` 字段；入口与 219 项 focused CPU 回归说明见 [ground_support_integration_review.md](ground_support_integration_review.md)。真实 pilot / 完整采集由 root 执行。这里没有运行 native physics/GPU 验证，也没有修改任何历史报告。
