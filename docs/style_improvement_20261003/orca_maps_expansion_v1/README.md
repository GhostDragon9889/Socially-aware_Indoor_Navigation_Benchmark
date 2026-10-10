# 新增ORCA地图轨迹数据（CPU生成）

TRACE作者附录C.1描述了ORCA-Maps与ORCA-Interact各1,000个10秒场景、80/10/10场景划分。当前本地Diffusion使用24种布局、48个20秒episode。本轮使用已冻结且未修改的生成器，另建1,000种非空障碍布局、各一个10秒episode，800/100/100场景划分，seed=20261003，障碍上限20。CPU任务不修改现有AMASS、Diffusion权重或正在训练的AMP输入。

这份数据是**本地ORCA-Maps扩展**：沿用4–15代理、真实15m邻居与可见障碍窗口筛选、0.4m圆盘半径、速度方向代理。未生成ORCA-Interact，代理数量、筛选、随机种子及初始采样与作者数据不同，不能宣称作者数据复原。生成器在60Hz检查每条完整episode的圆盘间／障碍间隙，并记录拒绝原因；输出只有真实10Hz五维平面轨迹，没有SMPL姿态或力矩。

所有旧目录保留；新目录仅在整批成功后原子发布。`generation_actual_launch.json`与`generation_CPU.log`记录实际进程，完成之前不能把预定1,000场景报告为已生成。后续需独立核验文件、分割与轨迹统计，再登记新的Diffusion训练／验证；生成数据不证明模型或物理导航改善。

依据：[TRACE/PACE作者附录](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf)，[固定TRACE数据生成源](https://github.com/nv-tlabs/trace/tree/594b2af0824c5853e1a1884e167cd5ed097dd81c/orca_sim)。

## 实际生成完成

Root已观察CPU任务真实退出0；生成用时202.84秒，1,000种布局已原子发布到 `data/diffusion_amp/orca_maps_expanded_1000_v1`。场景与episode数量分别为800／100／100，合法条件窗口137,689／17,359／16,698；158次拒绝尝试及其原因保留在manifest中。manifest SHA256为 `97e1d6390b15aa9e72adddcb6cf57a843d4a1c8440d1fb0ed88d7daf1bf97bdd`，实际进程记录见[生成记录](generation_actual_launch.json)。独立完整性与10Hz几何QC尚待完成；尚未以本数据训练新权重。

独立QC随后实际完成，退出0／21.69秒。[报告](../orca_maps_expansion_independent_QC_v1/actual_QC.json)核验全部3,003个文件、957,884个保存的actor-frame和171,746个条件窗口：文件字节、非object有限数组、10Hz时序、布局／scene／seed隔离、窗口边界、真实邻居、可见障碍以及圆盘／多边形间隙均通过。最小pair／map间隙为−0.000098095／−0.000029065m，落在原0.0001m容差内；负值保留。60Hz质量仅绑定实际生成源和日志，未独立重算未保存的60Hz状态；全局raster几何检查使用16像素格点，没有冒称逐像素重生成。Root随后将36,973,374字节新数据全部[只读封存](dataset_root_seal.json)。这仍是数据质量证据，尚无新模型效果结论。
