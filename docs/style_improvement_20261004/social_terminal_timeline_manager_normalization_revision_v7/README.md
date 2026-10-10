# v7：owned manager 的 public identity 归一化，仅 terminal smoke

准备状态为 source-only，尚未执行 v7 SDK。此独立目录派生已封 v6 `c134cccce25d1fe1ebb3343e7f13f35e66607b08c4edd9a945c738fce677e49a`；不修改 v6、IsaacLab 文件、World/factory、训练代码或资产。新源尚待 ROOT/独立 peer 审阅后封存。

## 已观察与源码根因

真实 v10 owner：session75237，`4f953a` → `890bca`，退出139，owner SHA `a786c9265170622a91874fde79de0f2d1aefbccb7dd4e2b63ae15e1ead2bc67a`。exact context/cleared singleton、exact config/proxy/reference 和 canonical resolver identity 已通过；联合 Lab 状态 guard 失败，Core shutdown 没有执行。该记录没有独立观察联合 guard 的每个条件，不能说 callbacks、subscriptions、views 都未清。

固定安装源码给出具体路径：

1. `SimulationContext.__init__:174` 将 `self._physics.class_type` 原代理赋给 `self.physics_manager`；`PhysxCfg:42` 声明 `{DIR}.physx_manager:PhysxManager`，`configclass` 将其包装成固定 `ResolvableString`。
2. `PhysicsManager.initialize:376–379` 明确写 **Base** 的 `_sim=sim_context`、`_cfg=sim_context.cfg.physics`、`_sim_time=0`。本版本没有另一个 `set_cfg` 写路径。
3. `PhysxManager.close:436–438` 无条件 invalidate views、清 subscriptions，最终 `449` 调用 `super().close()`。Base 的 classmethod 此时仍绑定 canonical PhysxManager。
4. `PhysicsManager.close:443` 判断 `sim.physics_manager is cls`。实际 `sim.physics_manager` 是 string proxy，`cls` 是其解析出的 canonical class，两者不是同一对象，故 active-manager 为 False。
5. `447` 的 callback 清理仍执行；`445` 的 STOP dispatch 与 `449–451` 的 Base `_sim/_cfg/_sim_time` reset 被跳过。因此源码预期联合 guard 中至少 Base `_sim`、`_cfg` 非 None；这属于源码推导，v10 尚未逐字段实测。

Core `invalidate_physics:729–748` 处理的是自己的 view/warmup 状态，不会补清 Lab Base 的 `_sim/_cfg`。不能用它替代正确的 SDK `close()` 路径，也不能直接写 Base 私有状态掩盖问题。

## v7 唯一新增 runtime 写入

原 helper 增加显式 `phase='before_clear'|'after_clear'`，共同保留 v6 全8份 SDK whole-source、原10个方法及7个 resolver 函数的 origin gates。

**before_clear** 在原 `sim.clear_instance()` 之前调用，严格要求：canonical exact context；singleton 是当前 owned sim；Base `_sim` 是该 sim；Base `_cfg` 是它的 `_physics`；`_physics` 是 exact PhysxCfg 且与 `sim.cfg.physics` 同一对象；public manager 是该 config 的同一个 exact ResolvableString；canonical reference 字串和 resolver target 均匹配固定来源。随后仅写：

```python
owned_sim.physics_manager = resolved_manager
```

归一化完成再实查 public attribute 为 canonical class、原 config proxy 未变、Base owner/config 身份未变，记录 `owned_manager_normalization_returned=True`。此分支不执行 initialize、dispatch、step 或 Core/cache release，不替换 SDK method，也不写 `_sim/_cfg` 等 SDK 私有状态。finally 清 helper 所有新增 proxy/class/context aliases。

**after_clear** 要求实际 preflag=True、singleton=None、public raw manager 已是 canonical class，同时原 PhysxCfg 的 exact proxy/reference/resolver identity 仍吻合。报告在联合检查之前分别记录 Base `_sim/_cfg` 是否 None、callbacks 是否空、Lab subscriptions 是否空、view/warp view 是否 None、Lab inherited `_sim` 是否 None。随后保留原 Base-cleared guard、Core/interface/callback/subscription检查、Core `_shutdown()` 和仅 Lab/Core `_timeline` 的 terminal release；原 GC/App.close/失败保留路径不改。Main 成功门明确要求真实 normalization preflag。

**预期行为改变是 SDK 自身 active-manager close 分支恢复执行**，包括它本应发生的 STOP dispatch 和 Base state reset。因此不声称 teardown callbacks 全部与 v6 执行相同。原控制循环、物理步数、contact 数学、两阶段探针、力阈值与原 raw archive 逻辑未改变；实际 v7 退出是否成功必须由 ROOT 串行 SDK 运行确认。

## 派生与边界

`source_derivation.json` 给出8个精确 literal hunks、全文 inverse byte/AST 恢复 v6、9个不相关函数 byte/AST 不变。变化只有 helper 的两个阶段、`_run` 的前后调用和 Main 的新 flag 初始化/成功门。静态 source compile/inverse tool `5d39f4` exit0，Ruff `1d97ce` exit0；未 import/执行候选、SDK、Torch、NN、权重、数据或数组，无新 suite/物理 preflight。

这是固定安装版本的**进程终末兼容接缝**，保留此前 private Timeline workaround；不是公开 SDK API。进程内再次创建 App/context 或真正 resume 不受支持。没有声称消除 SIGSEGV、证明训练可用或把源码检查当真实 physics/runtime 通过。
