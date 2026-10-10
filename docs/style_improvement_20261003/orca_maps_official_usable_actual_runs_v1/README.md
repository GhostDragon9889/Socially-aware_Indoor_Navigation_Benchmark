# 官方 ORCA Maps 流程：已生成并适配的本地语料

已完成 1,000 场景生成、全量转换、CPU 检查及封存。生成器使用固定版本的 [TRACE 官方 ORCA 实现](https://github.com/nv-tlabs/trace/tree/594b2af0824c5853e1a1884e167cd5ed097dd81c/orca_sim)，最多 10 人、20 个障碍物、10 秒，随机种子 20261005。这是本地生成的数据，未获得作者实际训练集。

数据位于 `data/diffusion_amp/orca_maps_official_usable_1000_v1`，含 1,000 个动作文件、1,000 个地图文件和一份 manifest；全部 2,001 文件已核对身份并设为只读，各文件为独立 inode。原始文件保留在 `orca_maps_official_raw_1000_v1`。

| 分割 | 场景 | 有效条件窗口 | 无有效窗口场景 | 空地图 |
| --- | ---: | ---: | ---: | ---: |
| train | 800 | 78,149 | 91 | 41 |
| validation | 100 | 10,211 | 10 | 3 |
| test | 100 | 9,129 | 13 | 1 |

所有原始场景均保留。空地图表示已知没有障碍物，地图条件仍为 True；Interact 的缺失地图则表示未知。窗口要求当前有真实邻居，不额外根据可见障碍物、碰撞或地图是否重复筛选。

时间轴用真实初始状态补入 t=0，再对原始非均匀时间戳插值到 0..9.9 秒的 100 帧；历史为 31 帧，未来为 52 帧。方向是速度方向代理，静止时不作为真实身体朝向。栅格为两通道 1080×1080，12 像素/米；使用本地 Shapely 栅格化，未声称与作者 cv2 的边界像素完全一致。

原始检查发现 114 个行人对重叠记录和 25 个障碍物重叠记录，均已保留；这份语料不标记为无碰撞。转换检查核对全部场景、文件、时间轴、窗口及地图语义，并对所有非空地图进行几何像素探测。

## 实际执行证据

- 转换：root session 45405，completion `e60c2a`，实际退出码 0；[实际记录](conversion_actual_launch.json)。
- CPU 检查：session 36128，completion `b0bd5b`，实际退出码 0；[完整报告](../orca_maps_official_usable_QC_v1/actual_QC.json)和[退出记录](../orca_maps_official_usable_QC_v1/actual_CPU_closure.json)。首次检查的 tuple/list 表示差异失败也保留，修正检查器后通过，未重写数据。
- [root 封存记录](usable_root_seal.json)：2,001 文件，SHA256 `2697aea244cd01be86b2a48c9a9aa8b5027fc316f5c192d9f765d66f23e31bdd`。

与已适配的 Interact 合并后，train / validation / test 各有 228,588 / 29,448 / 28,821 窗口。正式混合 Diffusion 训练尚未完成；数据完整性检查不代表模型效果或论文复现已经达标。
