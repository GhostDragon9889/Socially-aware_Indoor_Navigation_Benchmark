# Isaac Sim6.1 headless IOMMU modal提示

目标官方经验文件已经设置避免启动IOMMU弹窗的值：Lab `apps/isaaclab.python.headless.kit:155`（section从129行开始）、`isaaclab.python.kit:203`（section从184行开始）、Sim `apps/isaacsim.exp.base.kit:268`（section从247行开始）。均为 **[settings.persistent]** 下 `renderer.startupMessageDisplayed=true`。

实际key是 **`/persistent/renderer/startupMessageDisplayed`**。自定义headless经验文件可在 `[settings.persistent]` 显式添加此值，或启动CLI `--/persistent/renderer/startupMessageDisplayed=true`；必须在AppLauncher/SimulationApp启动之前生效。无persistent前缀是另一配置路径，不能从这些源码行推断它能禁掉modal。`headless=True` / `--no-window`与该独立modal提示setting不是同一个设置。

该项只让提醒被视作已显示，**不关闭硬件IOMMU或改变实际物理/渲染安全性**，日志硬件状态仍可保留。官方[6.1性能文档](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/sim_performance_optimization_handbook.html)说明IOMMU的multiGPU性能影响；本任务无需改变BIOS/kernel。未执行任何系统修改。精确源码位置见 `headless_iommu_setting_sources.json`。

本子任务只读查证该设置；root负责项目headless experience配置与真实RTX验证。


---

打包说明：本页保留审查当时的阶段结论。最新证据与复制范围见 [审计索引](README.md)；未随包复制的大清单、源码快照与完整日志见 [来源索引](SOURCE_INDEX.md)。项目源码位置以仓库相对路径表述；SDK 绝对路径仅记录被审查机器。
