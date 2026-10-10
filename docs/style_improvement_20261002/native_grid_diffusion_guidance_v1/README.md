# 原生栅格 SDF 的显式 Diffusion 接口

该接口为独立部署扩展，不是作者八场景复现或物理碰撞安全证明。默认 `scene_collision_cost_fn=None` 保留既有路径，模型权重未修改。

## 接口与坐标

```python
planner.plan(..., guidance_mode="trace_clean",
    scene_collision_weight=10., scene_collision_radius=.8,
    scene_collision_cost_fn=native_scene.collision_cost)
```

回调接收 `world_xy[B,S,T,2]`（世界米坐标）及 `radius`，返回相同 dtype/device 的有限非负 `[B,S]` 成本。规划器只执行一次 local→world 变换；回调负责栅格坐标、未知/域外策略和成本单位。clean 梯度与多个候选的筛选调用同一成本。联合 actor 指导保留同场景共同 sample 编号，不按 actor 拼接不同候选。

接口显式拒绝 legacy 模式、零权重、同时提供 AABB 参数、输出形状/设备/dtype 不符、负数/NaN、断开的坐标梯度或非有限梯度。自由区连接计算图的零梯度允许通过。`autograd.grad` 不写入模型或 provider 参数 `.grad`。

NativeGridScene 成本是时间平均的 `mean_T relu(radius-SDF)`；旧 AABB 成本按时间/障碍求和。因此两者同名权重数字不表示相同总惩罚。SDF 是软指导代理，未知和域外保守阻塞由 provider 负责；没有 swept-volume、边界或人体接触保证。

## 证据与边界

真实 learned step5000 的 CPU 检查见 [actual_legacy_parity.json](actual_legacy_parity.json)：旧默认和旧 AABB 两种模式，七个输出字段各自逐位相同，共14张量比较。固定数学观测与私有 seed；参数和全局 plan RNG 不变。这证明旧路径兼容，未执行物理仿真。

最终 [tests_final.log](tests_final.log) / [XML](tests_final.xml) **89 passed**；[Ruff](ruff.log) 通过。专项覆盖回调、既有 clean/near-horizon 与同事最终 grid API。最终源码见 [source_freeze.json](source_freeze.json)。初次 [tests.log](tests.log) / [tests.xml](tests.xml) 保留同事 native fixture 中间态缺 `actual_bounds_world_m` 的失败，不能当作最终通过。

[runtime_hook_review.md](runtime_hook_review.md) / [JSON](runtime_hook_review.json) 是新单 App/单 world 接线的 CPU 只读审阅，包含地形窗口边界、报告字段与严格 CUDA 反向风险。物理资源烹饪、真实原生导航和稳定性由 root 唯一 Kit 实例验证；此产物没有运行 GPU、Kit 或碰撞模拟。
