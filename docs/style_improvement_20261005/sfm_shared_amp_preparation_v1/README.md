# 同一 AMP 的 SFM 高层接口：实际 CPU 准备

本阶段只新增纯 CPU 生成器和测试，未修改旧 SFM、driver、native staging 或22项 guarded source。没有加载动作／checkpoint，没有运行 GPU、SDK、训练或 native 评估。这里实现公开方法重建，不是作者私有 SFM 参数或八场景复现。

## 完整来源与动力学

原 `/project/EXP/NavIsaaclab2.0/CrowdSim/world/sfm.py` 与本地 `CrowdSim/world/sfm.py` 完全同字节：10,471B，SHA256 `88f3274e1bfa2361dd92bb827b7b924d0d7c90d88f4a868ba76913bdfeccfd9c`。函数及相关配置的全部字节、SHA、行号记录于 [source_contract.json](source_contract.json)。[固定公开源码](https://github.com/yh83305/NavIsaaclab2.0/blob/d742f0c5b421bbc281e2112e7e50427e0336c16a/CrowdSim/world/sfm.py) 和 [公开导航调用者](https://github.com/yh83305/NavIsaaclab2.0/blob/d742f0c5b421bbc281e2112e7e50427e0336c16a/CrowdSim/nav_manager.py) 是方程来源。

生成器保留全部四项：目标吸引、EDT 梯度静态斥力、指数邻居斥力、相对速度最近接近 TTC；保留原速度球裁剪、严格邻居半径、self 排除和迎面退化的 goal-relative pass-right。原类 `fov=False`，所以全体半径内邻居均参与，无 top-k 截断。原类显式 float32 的邻居相对位置、closest-time 和 pass-right 中间量也保留；输出状态／EDT 使用 float64，没有 bitwise 等价声明。

每次内部更新先读取全部人的旧位置／速度，联合计算全部人的下一速度，再联合推进位置 `xy_next=xy+dt*v_next`。预测中不调用物理状态、不逐人覆盖邻居、不随机改变目标。生成器将 caller 给定的每人目标固定5秒，这是把公开**单步** SFM 扩展为联合预测的本地合同。实际重规划／路线推进由未来 caller 负责。

上游 SFM dt 由 `nav_manager._dt()` 的实际 env dt × 更新间隔决定，缺少 env 时才回落25Hz。本接口默认内部1/30秒、150次更新，然后每3步采样，输出 `raw_xy[N,51,2]` 的 `0,.1,..,5s` 与 `future_xy[N,10,2]` 的 `+.5,..,+5s`。内部 dt 可配置，但必须整除 .1秒；.04秒被显式拒绝，不偷偷插值或切换时钟。默认参数取公开类及 env local 配置：vmax1m/s、k5、static A/B15/.25、agent A/B25/.5、safe1.2m、neighbor4m、reach.7m、TTC gain28／threshold3s／horizon3.5s。未来同 AMP 评估可以显式登记其他参数，不能改标成作者配置。

## 边界与同 AMP 适配

`StaticDistanceGrid` 必须收到实际 metric EDT、ROS labels0/205/254、lower-left edge origin、resolution 和 source ID/SHA。occupied／unknown 的距离必须为0，known-free 必须正值。行0是最大 world Y、列0最小 world X；cell lookup 使用明确的边缘／中心合同。它不计算或认证 native 地图，不恢复上游旧 NavigationTask 的 round-to-node 坐标习惯。source ID/SHA 是 caller 的来源声明；真实数值数组及坐标框另有独立 SHA，不把声明伪装成文件验收。

当前位置、目标及每条预测线段都检查地图。线段在所有格边交点分段，并检查交点与区间中点；unknown、occupied、域外、NaN、源文件变化、参数变化、grid变化或运行中输入变化立即 fault，既不剪裁坐标，也不输出静止／重采样 fallback。严格拒绝非法预测是本地边界合同；预测失败必须由未来 runner 保留，不能删除失败评估行或制造成功率。

`SourceBinding.bind_sfm(path)` 严格接受上述公开 SHA，调用前后复核实际源文件。状态、grid 和参数都有输入身份检查，返回 raw 预测数组只读。seed 必须显式给出；本方法不消费 RNG、不重置全局 seed、不施加随机扰动。

```python
from CrowdSim.humanoid.trajectory.social_force import (
    SFMParameters, SocialForceTrajectoryGenerator, SourceBinding, StaticDistanceGrid,
)

# 下列 actual_* 必须由未来实际 caller 提供，当前没有执行 native 调用。
grid = StaticDistanceGrid(
    actual_distance_m, actual_labels,
    origin_xy=actual_origin_xy, resolution_m=actual_resolution_m,
    source_id=actual_map_source_id, source_sha256=actual_map_source_sha256,
)
generator = SocialForceTrajectoryGenerator(
    grid, parameters=SFMParameters(integration_dt_s=1/30), seed=7,
    sources=(SourceBinding.bind_sfm("CrowdSim/world/sfm.py"),),
)
prediction = generator.generate(actual_xy, actual_velocity, actual_goals)
raw_xy, raw_times = prediction.raw_xy, prediction.times_s
ten_raw_targets = prediction.future_xy
adapter_configuration = prediction.reference_adapter_kwargs()
```

`reference_adapter_kwargs()` 明确返回现有 `ReferenceAdapterConfig` 所需值：30Hz、.1秒 raw knots、5秒、v1.5m/s、a2m/s²、tracking-time.25s、initial overspeed `brake`。该函数**不执行**适配、不改变 SFM 动力学。未来 caller 将完整51帧交给既有 `CausalReferenceAdapter.build`，实际 reset 才用真实 root 初始化，普通 replan 复用上次已执行参考状态，并给 SFM／DP 两臂同一配置。再从适配后的参考取 AMP 所需目标，不能将原始十个稀疏目标声称已满足因果加速度限制。上游 MaskedMimic 的低通／加速度／yaw future 外推不用于替代这套共同 AMP 接口。

## 实际 CPU 验证

实际工具 `5788f1`，exit0：**22 passed in 0.84s**；Ruff `917314`，exit0。测试执行 unchanged 上游 `Social_Force` 完整 class AST，只注入 NumPy，不执行原模块 plotting/OpenCV/main。完整四项力与一步状态对照通过，完整150步联合上游循环与51／10帧抽样对照通过。其余覆盖对称／排列、超过五个邻居、邻居半径边界、静态 XY 斥力、TTC threshold／receding／stationary／迎面退化、coincident finite、seed无隐藏RNG、NaN、unknown／域外／斜线穿格、source／params／输入变更及无效 dt。独立 CPU 子进程 exit0，确认 `torch`、`omni*`、`isaaclab*`、`isaacsim*`、`pxr*` 均未加载。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m pytest \
  tests/test_social_force_trajectory.py -q -p no:cacheprovider
ruff check --no-cache CrowdSim/humanoid/trajectory/social_force.py tests/test_social_force_trajectory.py
```

这些是纯 CPU 算法／来源／退出证据。原始 SFM 没有加速度保证，预测也没有 articulated body clearance、真实接触或导航成功保证。新 social actor 的实际 AMP适配、native短 smoke、N3–15、200秒及八场景仍未运行；资源只有此前已核验四场景，本阶段不创建任何新场景或晋升旧结果。后续集成由 root 在 S2 与既定 owner gates 完成后串行验证。
