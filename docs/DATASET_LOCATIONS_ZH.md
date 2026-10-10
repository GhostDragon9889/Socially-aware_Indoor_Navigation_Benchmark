# 数据集迁移与当前位置

更新：2026-09-30。此记录更新此前审查中“配置中的场景路径不存在”的状态。

## 已完成

将 `ProtoMotions/FLUX/assets/dynbench` 整体移动到：

```text
/project/EXP/SangforAndSaut/Assets/
├── DynBench/                              # 数据实际存储位置
│   ├── Materials/
│   ├── SkyTexture/
│   ├── cluttered_scenes/
│   ├── dynbench/
│   └── isaacsim_scene/
│       ├── Full_Warehouse/
│       ├── Hospital/
│       ├── Jetracer/
│       ├── Office/
│       ├── Warehouse/
│       └── Warehouse_multiple_shelves/
└── Simple_Warehouse -> DynBench/isaacsim_scene/Warehouse_multiple_shelves
```

`ProtoMotions/FLUX/assets/dynbench` 保留为指向新位置的符号链接，原 FLUX 路径仍可访问数据。采用同一文件系统内目录重命名，无额外复制整套数据。

四份 `CrowdSim/config/scenes/dynabench_*.yaml` 中 `/data/DynBench/` 改为 `../Assets/DynBench/`。两个默认仓库配置通过 `../Assets/Simple_Warehouse` 目录链接找到场景；完整场景目录与模型、材质保持在一起。未创建系统根目录 `/data`。

## 验证结果

- 全部 108,178 个文件的相对路径和尺寸一致，合计 59,911,351,479 字节；计数包含下载缓存/元数据。
- 六个入口 USD 的 SHA-256 与移动前一致。
- 六个入口 USD 均能由 Sdf 打开；直接外部引用的缺失集合与移动前一致。
- 六份 CrowdSim 场景 YAML 引用的 USD 均存在。
- 尚未运行完整物理/渲染仿真，未递归验证全部资产引用。

## 仍待补齐

以下地图在移动前即不存在：

```text
Assets/DynBench/isaacsim_scene/Full_Warehouse/map/occupancy.yaml
Assets/DynBench/isaacsim_scene/Hospital/map/occupancy.yaml
Assets/DynBench/isaacsim_scene/Office/map/occupancy.yaml
Assets/DynBench/isaacsim_scene/Warehouse/map/occupancy.yaml
```

地图需要从对应场景生成并校准。当前 Warehouse 配置使用的 `CrowdSim/maps/newmap.yaml` 与 `World1.png` 虽然存在，尚未验证与新接入 USD 的坐标/障碍物一致性。

入口 USD 原有的本地资源缺失（移动未增加缺失）：

- `Assets/DynBench/Props/Forklift/forklift.usd`，被 Full_Warehouse 引用。
- `Assets/DynBench/Props/KLT_Bin/small_KLT_visual_collision.usd`，被 Warehouse 和 Warehouse_multiple_shelves 引用。

AMASS MotionLib 不属于 DynBench；现已从独立 AMASS 原始包生成默认验证库、测试库与训练库，位于 `../Assets/motion/`，详见 [AMASS 数据准备记录](AMASS_PREPARATION_ZH.md)。SMPL 可视网格参数模型仍未补齐。FLUX 导航权重仍保留原位置；它不属于 NavIsaacLab 人体模型数据。

[迁移校验记录、场景路径及原配置备份](dependency_audit/dataset_migration.json)。
