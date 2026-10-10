# 实际策略调用前观测独立审计

passed。CPU-only，无模型/SDK/GPU。

完整采样：30条。

trajectory_prefix20: 120 sampled actorframes / 2400 elements; clipped 0 (0.000000%)。

terrain_suffix1024: 120 sampled actorframes / 122880 elements; clipped 2061 (1.677246%)。

历史 source168 与实际 source177 的 physical/plan/ground 全字段 bitwise equal=True；具体差异见JSON，不用容差替代原数组比较。

Pre-action raw clone before act; durable publication after act before step is frozen source evidence, not engine timestamp.

Additional normalizer forwards are not model forwards; no per-call runtime model counter was instrumented.

30s samples/prefix do not replace full200s stability or guarantee PhysX/Warp bitwise.

Clipping measurement does not prove idle/failure causality.

Historical/current source differ; selected source archives are not full import ledgers.

All physical failed/reset frames are retained; initial capture is post-control, not a pre-step initial GT/history oracle.
