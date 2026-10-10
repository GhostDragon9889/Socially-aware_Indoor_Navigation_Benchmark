# Isaac Sim 6.0 原生退出修复：实际 v18

2026-10-04。八人、两组、相同物理配置的直接 SDK 进程正常退出 **0**：初始工具 c8b514 / session93877，完成 e8ad50。没有使用调试器、快速退出或跳过退出处理器。

[实际结果与来源](../social_actual_runtime_v1/group_physics_native_overlay_v18_actual_root_owner.json)已只读保存；[原 v17 崩溃](../social_actual_runtime_v1/group_physics_native_exit_mapping_v17_actual_root_owner.json)保留。

- 官方 v6.0.0 的四个 C++ 实现文件使用 GCC11/C++17 编译并链接。只有三个源码改动：保存 StageUpdate 节点句柄、接收创建返回值、在原 onShutdown 删除自身节点并释放自身订阅及 StageUpdatePtr。接口和物理计算保持。
- 独立扩展完整复制了当前安装版，排除 pycache，只替换原生插件。运行实核 development path、扩展路径、两个 Python 来源、未改动的 cp312 绑定及进程库映射；插件 SHA 为 cc21b768…。
- 原64物理子步完成，同组接触正探针1355.844238N，跨组原始力严格0。八个contact弱引用及后端SimulationView弱引用均已销毁，4个Articulation和4个Data字段清空，cleanup_errors为空。
- 35个安装扩展文件前后SHA相同。独立元数据审阅也确认来源、进程结果和释放计数。
- 与此前 v15 相比，四个物理NPZ的文件字节数及完整SHA均相同；这是已归档文件身份对照，没有重新读取数组。退出结果从139变为0。

[构建源码和参数依据](../native_simulation_manager_overlay_revision_v1/README.md)与[实际运行入口](../social_native_overlay_entry_revision_v10/smoke_group_physics.py)分开保存。build.py已审阅但未执行；本次实际使用root记录的四单元编译和链接命令，见overlay的build_copy_manifest.json。下载的开发依赖使用当前Kit110.1.1、USD25.11和Physics110.1.11，库链接使用当前SDK版本。

本结果覆盖当前八人物理探针。此前C退出的具体C++对象尚未被符号直接证明；本次正常退出也不证明所有训练配置、长时间运行或论文动作效果。下一步复用已加载验证的扩展，运行固定train2848的共享World和短训练，然后执行完整训练及统一物理／风格评价。
