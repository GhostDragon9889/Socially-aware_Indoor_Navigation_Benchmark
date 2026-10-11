# Isaac Sim 6.1 stock SDK 退出崩溃定位

2026-10-10，在 `/project/EXP/IsaacLab/.venv/bin/python` 的原版 SDK 中运行 headless AppLauncher + Physx SimulationContext，1/120 s ×8 steps。未导入项目 lifecycle 包、未替换 SDK .so、未使用 os._exit、未批量删 sys.modules。

## 对照与确定对象

1. root 已确认 bare AppLauncher + full close 正常退出，加入 SimulationContext 后退出 SIGSEGV。
2. 释放 `isaaclab_physx._SIMULATION_MANAGER_ENABLE_HOOK` 后，两项 IHookHolder 弱引用均死亡，但仍在 Python `finalize_modules -> GC -> type_clear -> dictkeys_decref -> carb -> omni.timeline/_timeline` 崩溃。
3. GDB 利用实际 CPython3.12 头文件布局，只读解析 type_clear 参数和即将清空的类字典。准确对象为 **PhysxManager** 堆类型，其字典 **`_timeline` = `omni.timeline._timeline.Timeline`**。类字典中的 `_event_bus` / `_message_bus` 为 IEventDispatcher，但本次崩溃在 Timeline dealloc，不需要按猜测批量清理其他接口。
4. 在 clear_instance 完成后、App.close 前一并释放 SDK enable hook、`PhysxManager._timeline`、已加载 `omni.timeline.get_timeline_interface.timeline` 的缓存接口，再 gc.collect；同一进程只 App.close 一次。GDB 显示 **`[Inferior ... exited normally]`**，未收到 SIGSEGV。

## 文件与结果

- `native_gdb.log`: hook 释放后的第一轮崩溃堆栈。
- `native_gdb_types.log`: type_clear 只读捕获，保存完整 PhysxManager 字典。
- `native_crash_objects.json`: 上述崩溃对象类型、地址和所有类字典键名（不含凭据）。
- `gdb_hooks.py`: 只读目标内存诊断程序，不执行 inferior Python/C 函数。
- `preclose_native_release.py`: 最小成功对照脚本。
- **`native_gdb_preclose.log`: 被测 Python 进程正常退出**。外围 GDB命令 exit1 是正常退出后继续执行 `bt` 导致 `No stack`，不是 Python 错误；日志明确记录 inferior 正常退出。

## 集成约束

仅在目标 SDK 已停止物理、clear_instance 和所有依赖 Timeline 的清理都完成后释放这些 handle，并在 native App.close 前进行。不得在仿真期间将 PhysxManager._timeline 清空。未改动 SDK 源文件。此结果覆盖最小 Physx sim，不代表所有摄像机 / contact / 含 articulation 的主场景关闭已验证；root 随后确认同样的 stock framework teardown 最小 PhysX进程真实退出码0；后续主场景仍须分别引用其实际验证结果。


## 真实 SecondFloor 行人场景：另一个 tensor borrow 残留

2026-10-10 16:04 UTC（本地 2026-10-11 00:04），使用项目目标 Python `.runtime/isaaclab3_ea_sim61/.venv/bin/python`、原版 SDK 和 `scripts/run_trace_pace.py --config configs/reproduction/four_scene/SecondFloor_profile.json --scene configs/reproduction/four_scene/SecondFloor_scene.yaml --steps 1 --measure-contacts` 做 GDB。已采用前述 hook/timeline 释放和 stock framework teardown；仍在 Python finalization 崩溃。这不是同一个 Timeline 对象问题。

单步运行完成、0 falls、cleanup_succeeded=true；`failed_progress` 是1步不能达到路线进度门槛的预期主程序结果，不能当作60秒性能评估。随后收到 SIGSEGV，确切退出链为：

```text
Articulation subtype_clear
  -> Articulation.__dict__ dealloc
  -> ArticulationData subtype_dealloc
  -> ArticulationData.__dict__ dealloc
  -> SimulationView subtype_dealloc
  -> omni.physics.tensors.bindings._physicsTensors
  -> libomni.physx.tensors.plugin.so
  -> SIGSEGV
```

GDB 只读解析 CPython3.12 实例字典，在即将清空 `Articulation` 之前保存快照，并从崩溃栈还原已保存的 `self`。实际对象如下：

| 对象 | 地址 | 已确认的关联 / 状态 |
|---|---|---|
| Articulation | `0x7ffc9eeafe00` | `_is_initialized=False`；`_root_view=None`；初始化/失效/prim deletion/debug callback handles 均 None |
| Articulation._data | `0x7ffc481709b0` | ArticulationData；`_root_view` 是 weakref.ProxyType |
| 上述两个对象的 `_physics_sim_view` | **`0x7ffc4811fe90`** | 同一个强引用 SimulationView，仍在两个实例字典里 |
| SimulationView._backend | `0x7ffc48158ab0` | `omni.physics.tensors.bindings._physicsTensors.SimulationView` native wrapper |
| SimulationView._frontend | `0x7ffc4820a120` | FrontendWarp |

`scene_native_crash_objects.json` 保存具体 frames/对象/字段/类型；`scene_native_gdb_instances.log` 保存完整执行与 GDB 报告。地址仅用于此进程的精确同一性证明，不能硬编码进项目。

### SDK 源码与 caller 清理约束

目标 SDK `isaaclab_physx/assets/articulation/articulation.py:4077` 的 `_invalidate_initialize_callback()` 只调用父类并清 `_root_view`；`articulation_data.py:88` 将 root view 存为 weak proxy，但 `:97` 将 `SimulationManager.get_physics_sim_view()` 返回值存入 `_physics_sim_view` 强引用。`contact_sensor/contact_sensor.py:329` 有相同全局 simulation view 强引用，`:684` 的 stop callback 会清 body/contact views 和 graph/buffers，未清 `_physics_sim_view`。GDB 本次直接触发的是 Articulation/Data；sensor 关联是源码证据，不将其标为已观测的崩溃对象。

据此建议由项目 lifecycle 在**终端关闭路径**释放 caller-owned scene 的 borrow：

1. stop / clear_instance 已成功，目标 SDK family 已确认，场景不会再次更新。
2. 对该场景实际持有的 asset/data/sensor 作限定遍历；拒绝 `_is_initialized=True`、asset root view 或 sensor body/contact view 尚未失效的情形。
3. 借用字段必须与已捕获的 public backend `simulation_view` **对象同一**。仅清匹配的 `_physics_sim_view`，不随意清未知 owner/native wrapper `_backend`，不通过全局模块扫描隐藏未清引用。
4. 释放 backend 的 public simulation_view 后，在 App.close 前检查该 wrapper 弱引用已死亡；若存活则明确失败，保留可定位错误。
5. 同一程序只关闭 App 一次，使用 target SDK stock framework teardown；不得用 os._exit 掩盖退出错误。

这是已终止且由项目创建的 SDK scene 实例清理，不修改 SDK 源码。第二轮 GDB 是**修复前失败证据**；root 随后完成补丁集成与真实 N1 验收，见下节。GDB 诊断进程已退出并释放 GPU。


## 集成后的真实退出验收

root 完成 target lifecycle caller-owned borrow清理并执行 `SecondFloor_N1_60_07`：真实父进程 `subprocess.Popen.wait` 返回 **0**，60 controls、0 falls、位移1.1426m、平均速度0.5920m/s、cleanup_succeeded=true、cleanup_errors=[]。本次为**60个控制步**、65.82秒墙钟时间，不是60秒仿真场景完整路线评估。

子结果明确记录：释放26个SDK-owned borrow aliases（articulation + articulation_data +24个contact sensors），全部与 public backend view 同一性核对通过；`public_backend_view_live_before_App_close=false`；Timeline两种Python wrapper分别释放，`same_python_wrapper=false`是可接受的接口包装差异。源文件与输入SHA在执行前后均一致。

验收文件位于父目录 `physics/SecondFloor_N1_60_07_execution.json`、`physics/SecondFloor_N1_60_07/result.json`；GDB诊断与此次成功执行应共同保留，前者解释故障对象，后者证明修复路径的真实正常进程退出。没有修改SDK源文件、没有native插件替换、没有os._exit或批量sys.modules删除。其他场景、N4、RTX录像、训练退出仍按其各自实际receipt报告，不由这一N1路径推定通过。


---

打包说明：本页保留审查当时的阶段结论。最新证据与复制范围见 [审计索引](README.md)；未随包复制的大清单、源码快照与完整日志见 [来源索引](SOURCE_INDEX.md)。项目源码位置以仓库相对路径表述；SDK 绝对路径仅记录被审查机器。
