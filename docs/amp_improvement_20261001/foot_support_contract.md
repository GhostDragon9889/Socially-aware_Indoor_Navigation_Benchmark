# 脚部地面支撑监测：独立接口与验证边界

日期：2026-10-02。新增 `CrowdSim/humanoid/foot_support.py` 和 `tests/test_foot_support.py`；没有修改 AMPWorld、训练 runner、跟踪基准或历史 capture。默认不集成、不创建 Kit，不生成历史报告缺失的支撑率。

## 固定 SDK 与真正测量对象

接口依据 [固定 SDK 源码审计](sdk_style_channels_review.md)。安装版 `RigidContactView` 提供 `sensor_paths`、`filter_paths`、`sensor_count`、`filter_count` 和 `get_contact_force_matrix(dt)`。每个力是**指定脚刚体与指定地面碰撞体之间法向力向量的和**，单位 N；不是净力，不含摩擦，不含墙或其他行人。

必须由调用者从实际 USD 解析 `L_Ankle,L_Toe,R_Ankle,R_Toe` 的 rigid-body prim；模块还验证每个 leaf name 与语义名字一致。`resolve_floor_collision_paths(stage, floor_roots)` 只接受显式选定 floor root 下 enabled CollisionAPI 后代，拒绝无 collider 的 Xform、缺失路径和重复。它不自动把所有 static geometry 当地面。初始化按真实 sensor/filter paths 校验完整双射并重排，不能只依调用列表顺序。

`contact_reporting_enabled=True` 是调用者确认已在 spawn 前开启 contact reporting 的必需声明，不是模块已经验证传感器活化的证明。原生物理 probe 前 `manifest.native_physical_probe_passed=False`，`author_metric_equivalence=False`。

## 可调用 API

```python
from CrowdSim.humanoid.foot_support import FootSupportMonitor

monitor = FootSupportMonitor.from_factory(
    physics_view,                 # 已存在的 native SimulationView，或其绑定 factory
    actor_foot_paths,              # 每actor一个 {SMPL foot name: exact rigid prim path}
    enabled_floor_collider_paths, # exact floor paths，先解析 enabled CollisionAPI
    contact_reporting_enabled=True,
    physics_dt_s=1 / 120, substeps_per_control=4, threshold_n=1.,
)
monitor.begin_control_step(
    episode_ids=episode_ids, control_step=control_step,
    physics_step_before=physics_step_counter,
)
# 每个 _sim.step(render=False) + _scene.update(1/120) 后调用一次：
monitor.sample_completed_substep(
    1 / 120, episode_ids=episode_ids, physics_step=physics_step_counter,
    foot_body_origin_vel_w=foot_link_vel_w,  # 可省略，明确 [B,4,3] SMPL foot顺序
    foot_body_origin_pos_w=foot_link_pos_w, # 可省略，同一顺序/device/dtype
)
# 只有完整4个子步才可输出；提供vel时必须显式给地面plane normal：
measurements = monitor.measurements(ground_normal_w=world_up_unit_tensor)
# reset、中断或失步后显式清空，不发布全零代替未知测量：
monitor.reset()
```

上述为接口示例，**没有在本次任务中接入或执行仿真**。`episode_ids`、counter 必须先转为 Python integers。完整控制步之间必须连续；子步 dt 不能用控制周期 1/30 替代，否则冲量到力的换算错误。actor reset 出现在子步内、重复/缺失 physics tick、partial control、非有限力、形状变化、device/dtype变化都会失败；异常会使当前 buffer 无法发布，需显式 reset。正好4步结束后第五次 sample 也会拒绝。

## 数据协议

令 S=4、B=actor数、F=4、G=实际地面 collider数，脚顺序为 `L_Ankle,L_Toe,R_Ankle,R_Toe`，side顺序为左、右。

| 输出 | 形状 / 定义 |
|---|---|
| `foot_floor_normal_force_w` | `[S,B,F,G,3]`，真实读取并 clone 的 world normal-force vectors |
| `foot_normal_force_vector_mean_w` | `[B,F,3]`，先sum地面列，再取4步向量均值 |
| `foot_normal_load_mean_n/peak_n` | `[B,F]`，每步先取该脚向量和的 magnitude，再做4步 mean/max |
| `side_normal_load_substeps_n/mean_n/peak_n` | `[S,B,2]` / `[B,2]`，side内 ankle/toe 的 magnitude 相加，再做 mean/max |
| `foot_contact_substeps` | `[S,B,F]`，严格每foot load `>threshold_n` |
| `side_contact_substeps/any/all` | `[S,B,2]` / `[B,2]`，left/right 是各侧 ankle/toe 任一接触，any/all是4步约简 |
| `double_support_fraction` | `[B]`，同一子步左右都接触的比例 |
| `no_support_fraction` | `[B]`，同一子步左右都不接触的比例 |
| `actor_substep_valid_count` | `[B]`，完整控制步均为4；partial数据明确拒绝发布 |
| `foot_body_origin_pos_w/vel_w` | 可选 `[S,B,F,3]`；缺失返回 None，不伪填0 |
| `body_origin_tangential_speed_proxy_mps` | 可选 `[S,B,F]`，在声明地面法向的切平面上的脚**原点**速度 |
| `supported_body_origin_tangential_speed_proxy_mean_mps` | `[B,F]`，只对 contact 子步求平均；必须同时看 `supported_speed_proxy_valid`，无接触的0不是有效测量 |

向量法向力求和后 magnitude **不等于多种非平行 contact patch 的标量法向力总和**。side load 也不是体重归一化的支撑率。mean、peak、any/all 接触率各自保留，不用最后一个子步冒充整个控制区间。

脚原点速度不是接触点滑移速度。没有 contact-point + contact-body/ground角线速度时只报告命名明确的 proxy；不能据此称脚底 friction/slip 已验证。弯曲或多个不同法向的地面不能用单一 world-up plane 推断真实切向滑移。

## Body COM 辅助函数

- `body_name_permutation(native_names, common_names)` 要求完整命名双射；native masses 与 COM 必须按同一 body 顺序重排。
- `body_link_world_com(link_pos_w, link_quat_xyzw, local_com_pos)` 使用 `p_link + R(q_xyzw) * p_com_local`，不是直接加 local offset；检查单位四元数、匹配形状、device/dtype与 finite 结果。
- `mass_weighted_world_com(body_com_w, masses_kg)` 使用真实 world COM 与每个 actor 的正质量，不能传 body origin 并标成真 COM。
- 固定 data 的 `body_com_pos_w.torch` 已提供组合后的 world COM；`root_view.get_coms()` 是 local COM pose。ProxyArray / Warp 转 Torch 后再检查分量形状，并 clone。现有 simulator 已显式读取 `body_link_lin_vel_w`；不要用 COM alias `body_lin_vel_w` 冒充脚 link-origin velocity。

## CPU 证据与后续物理 probe

28 CPU tests PASS，2.03s；Ruff 两新文件 PASS。包含可控瞬态双支撑/无支撑、两actor与地面列不同顺序、native buffer复用、四子步约简、严格阈值、缺步/跨reset/partial fail、nonfinite/overflow拒绝、COM 90°旋转 offset、质量加权、真实 CPU Warp vec3 分量展开、真实独立 USD enabled/disabled collider解析。mock接触值和 CPU Warp frontend **不证明 PhysX后端生成的接触准确**。结果：[JUnit](foot_support_cpu_tests.xml)、[JSON合同](foot_support_contract.json)。

最小 SDK integration probe 仍待 root 执行：

1. 静止脚在真实地面有有限非零 normal load；腾空同一脚的 floor load为零。
2. 接触墙体但脚离地时，不将 wall load写入floor filter。
3. 四个120Hz读取真实counter/episode一致，dt正确；首个control与reset后无旧buffer污染。
4. source/filter metadata 与实际stage完全匹配；至少2actor验证重排无串脚。
5. `body_com_pos_w` 与本模块 link/localCOM组合数值匹配；native masses/common映射一致。
6. 保存完整raw forces、foot pos/vel、episode/reset标记和source SHA后，才可报告真正feet-floor支撑指标。当前历史 captures不能补写这些缺失通道。
