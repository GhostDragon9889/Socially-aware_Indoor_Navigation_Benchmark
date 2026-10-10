# Foot-support probe v1 失败与 v2 SimulationView 修正

日期：2026-10-02。v1由root实际启动native GPU，但在初始化factory lookup处exit1，尚未采集三个阶段；不是“地面载荷为零”或步态失败。保留 `output/diffusion_amp/foot_support_native_qa_v1/result.json`、`/tmp/foot_support_native_qa_v1.log`，及当时完整脚本 `foot_support_probe_v1_source.py.txt`；没有改写它们。

## 具体原因与固定源码

v1脚本导入 `isaacsim.core.simulation_manager.SimulationManager`。它的getter返回None，尽管当前Lab3 SMPL articulation已成功初始化。固定Lab使用另一个owner：

- [SimulationContext](https://github.com/isaac-sim/IsaacLab/blob/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/source/isaaclab/isaaclab/sim/simulation_context.py)：174行从配置注册实际physics_manager，266行公开 `physics_sim_view` 委托它的 `get_physics_sim_view()`。
- [PhysxManager](https://github.com/isaac-sim/IsaacLab/blob/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py)：452行getter返回其 `cls._view`，795行以当前stage创建Warp SimulationView。
- [PhysX Articulation](https://github.com/isaac-sim/IsaacLab/blob/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py)：39行导入 `PhysxManager as SimulationManager`，3859/3871行由它取得view并创建root articulation。

安装tensor API的 `SimulationView.create_rigid_contact_view(pattern, filter_patterns, max_contact_data_count)` 是已确认的公开native工厂，支持exact foot paths和每sensor相同的exact floor列表。它返回Warp frontend force矩阵，由既有模块先转Torch再检查分量形状。这里仅检查固定源码和CPU contracts，没有启动新的SDK probe。

## v2 最小修改

新增纯lookup helper `resolve_native_contact_factory(simulation_context)`：

1. 要求实际 `physics_manager.get_physics_sim_view` callable。
2. 从现有公开 `world.simulator._sim.physics_sim_view` 取view，None明确失败。
3. 核对该view与实际active manager getter返回的对象一致。
4. 核对 `create_rigid_contact_view` callable，随后沿用exact脚/地面/墙 factory参数与metadata门禁。

不读猜测的 `_physics_sim_view`，不新建SimulationView，不回退legacy singleton；不会改动AMPWorld、功率循环、训练器、通用contact monitor或物理参数。source SHA与actual resolved path diagnostics现在在factory初始化之前写入report，后续失败仍能保留对应来源。SDK public physics counter逻辑未改，仍验证每control恰4个counter increment且读取前后不推进。

## CPU 验证与新运行

20 focused CPU tests PASS，0.88s；Ruff两文件PASS。新增测试重现“已存在active Lab view”路径、manager对象一致、未知backend无contact factory/None/owner错配fail-closed、拒绝私有lookup，以及将真实调用格式的exact foot/floor列表交给注入工厂。CPU mocks不是实际PhysX通过证明。

新冻结SHA：

- script：`009dde3a5f21af1a62e01d590b15edcf11eb067c1bc4ecaf180e064ec6cf934c`
- focused tests：`6ed415a155fb56a3b44c0b2b1b108499dd1b563f8384bd1e7b8fa9f77a0e91e4`
- 保留v1 script：`e0c92e8ba0b2d33e0bc9f3e73590e4a4a6922dd089708ad442f67eb11cc0a6d8`
- 保留v1 failed result：`9c83df914d1bded3575a438ba90294a3c75de5acd1d35cbd101e1abb7f9516d0`

root在GPU独占窗口用新目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
../IsaacLab-Protomotions/.venv/bin/python scripts/probe_paper_foot_support.py \
  --deterministic-evaluation \
  --output output/diffusion_amp/foot_support_native_qa_v2
```

v2实际SDK与三个物理阶段尚待运行。旧v1失败保留；不在取得完整原生result前称native ground/COM gate通过。[原协议与数据字段](foot_support_probe.md)、[v2 CPU JUnit](foot_support_probe_v2_cpu_tests.xml)、[证据JSON](foot_support_probe_v2_lookup_fix.json)。
