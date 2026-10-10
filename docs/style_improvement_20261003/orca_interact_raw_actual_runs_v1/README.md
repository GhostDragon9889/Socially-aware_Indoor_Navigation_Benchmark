# ORCA-Interact 官方格式原始数据

Root已对[全量实际QC](../orca_interact_raw_QC_v1/actual_QC.json)列出的1,000个`sim.npz`重新核验完整SHA，均为独立inode、nlink1，封存0444：[根封存记录](raw_root_seal.json) SHA `337577c8…`，总39,415,424字节。原数据保留原位置，尚未创建训练窗口。

依据[事前参数与源码登记](../orca_interact_raw_preparation_v1/README.md)，本轮生成1,000场景、最多20人、零障碍的独立原始数据。固定seed20261004，提前分配800／100／100场景；只使用CPU，原TRACE算法不改。

实际生成已退出0（root工具进程84128），[真实生成记录](generation_actual_launch.json)绑定原参数、源码、seed和日志；没有GPU运行。

[实际全部1000场景QC](../orca_interact_raw_QC_v1/README.md)随后CPU退出0（约.879秒）：场景与800／100／100分割一致，共10,608名代理／1,071,408个agent-frames，每场101帧，1–20人；13个官方字段全部finite、形状／dtype有效，各目录仅sim.npz，无地图或render文件。官方及本地比较源码在生成后逐字节复核，未调用空输出guard。

实际时间从约.016666668秒到10.000019073秒，间隔约.083332062–.100001335秒；最后间隔较短，不能称严格均匀10Hz。原时间戳未改，尚需显式时序／无图条件／窗口适配；此格式尚未接入本项目ORCACorpus或训练，也不称作者提供的数据。
