# NavIsaacLab 行人资源补全与运行记录

日期：2026-10-01。目标环境：Isaac Sim **6.0.0.1**、Isaac Lab 源码产品版
**3.0.0**（Python package version 为 **12.0.0**）、Python 3.12.13、Torch 2.11。
Isaac Lab 固定 revision：`4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`。

## 1. 当前完成范围

已从正式来源实际下载人体 UV/纹理、两份场景 HDR、七份权重和相关源码，
保留来源、固定 revision、许可、大小与 SHA256。既有 SMPL、AMASS、
MaskedMimic 模型继续使用。

当前可运行的链路为 **SFM → SMPL MaskedMimic → Isaac Lab PhysX → SMPL LBS 纹理网格**。
四个真实物理行人在同一世界完成 **200 秒带纹理验证**。
论文描述的 **Diffusion → SMPL 导航 AMP** 仍需要匹配的模型与接口；
本轮下载的 TRACE/PACER 是原项目公开参考模型，尚不能证明等同于论文定制产物。
[NavIsaacLab 论文](https://arxiv.org/abs/2606.26265)、
[官方公开仓库](https://github.com/yh83305/NavIsaaclab2.0)

详细公开发布范围与查询结果见
[论文资源公开核查](resource_completion_20261001/paper_publication_sources.md)。
未找到匹配公开产物是本轮查询结论，不表示作者从未发布。
本轮总清单与最终状态见
[resource_completion_summary.json](resource_completion_20261001/resource_completion_summary.json)。

## 2. 下载及接入位置

| 资源 | 本地位置 | 校验与接入 |
| --- | --- | --- |
| SMPL 原始纹理样本、OBJ、MTL、许可证 | `data/smpl/meshcapade_source/` | 固定 Meshcapade revision；与官方 Git blob SHA-1 相符 |
| 两张 4K RGB albedo | `data/smpl/textures/f_01_alb.002.png`、`m_01_alb.002.png` | 每张 4096×4096；实际 RTX 仿真已启用 |
| 对应 SMPL UV | `data/smpl/smpl_uv.npz` | 266,068 B；`uvs=(28944,2)`、`uv_faces=(13776,3)`；全部面唯一匹配 |
| Parking HDR | `data/scene_resources/hdri/abandoned_parking_4k.hdr` | 25,661,358 B；官方 MD5、SHA256、HDR/RGBE 文件头、4096×2048 均通过 |
| Sunflowers HDR | `data/scene_resources/hdri/sunflowers_4k.hdr` | 26,582,747 B；同上 |
| TRACE 原始权重包 | `data/pretrained_models/paper_references/trace/trace.zip` | 1,024,117,268 B；已按白名单解压并验证 ZIP CRC |
| TRACE ORCA 权重及配置 | `…/trace/ckpt/trace/orca_mixed/` | `iter40000.ckpt` 376,736,841 B；配套 `config.json` |
| TRACE 混合数据权重及配置 | `…/trace/ckpt/trace/nusc_eth_ucy_mixed/` | `iter40000.ckpt` 376,987,721 B；配套 `config.json` |
| TRACE nuScenes 权重及配置 | `…/trace/ckpt/trace/nusc_only/` | `iter40000.ckpt` 376,987,721 B；配套 `config.json` |
| 原 TRACE 源码 | `third_party/trace/` | `594b2af0824c5853e1a1884e167cd5ed097dd81c` |
| 原 PACER 源码 | `third_party/pacer/` | `baeb6241f03fc9d78370a64e3c2dcef9d830d97f` |
| PACER no_shape | `…/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth` | 119,755,239 B；ZIP CRC、SHA256、安全读取与 CPU actor 前向通过 |
| PACER shape | `…/paper_references/pacer/output/release/pacer/Humanoid.pth` | 121,648,127 B；同上 |
| PACER 站立、体型和遮挡样例 | `…/paper_references/pacer/sample_data/` | 3 文件共 4,963,954 B；与服务端大小一致，SHA256 已记录 |
| 第一作者 MaskedMimic SMPL | `…/paper_references/navisaaclab_public/masked_mimic/smpl/` | `last.ckpt` 229,160,269 B；完整五个配套文件 |
| 第一作者 SMPL terrain tracker | `…/paper_references/navisaaclab_public/motion_tracker/smpl-terrains/` | `last.ckpt` 132,661,223 B；完整五个配套文件 |

### 人体 UV 的实际转换

[Meshcapade 官方样本](https://github.com/Meshcapade/SMPL_texture_samples)中的 SMPL OBJ
是细分网格，不能直接截取其面索引作为当前 6,890 顶点网格的 UV。
转换工具按共享 face center 恢复原始 parent face，再按本地 neutral 模型的面顺序
逐角匹配，保留 UV 接缝。13,776 个三角面全部唯一对应，边环、绕序均通过；
男女样本网格的逐角 UV 最大差为 0。

转换源和归属说明保存在
[UV_TEXTURE_ATTRIBUTION.md](../data/smpl/UV_TEXTURE_ATTRIBUTION.md)，
转换工具为
[install_meshcapade_smpl_uv.py](resource_completion_20261001/install_meshcapade_smpl_uv.py)。
原样本适用 SMPL-Body CC-BY-4.0；当前真实 SMPL 参数模型继续沿用既有许可。
[官方人体许可](https://smpl.is.tue.mpg.de/bodylicense.html)

### 场景 HDR 的接入

两份 HDR 来自 [Poly Haven Parking](https://polyhaven.com/a/abandoned_parking) 与
[Sunflowers](https://polyhaven.com/a/sunflowers)，适用 CC0。
为医院和办公室 USD 原引用位置新增了相对软链接：

```text
../Assets/NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr
../Assets/NVIDIA/Assets/Skies/Clear/sunflowers_4k.hdr
```

它们解析到本仓库下载文件。未覆盖原 USD。
下载和安装证据分别在
[hdri_download_manifest.json](resource_completion_20261001/hdri_download_manifest.json)、
[hdri_installed_references.json](resource_completion_20261001/hdri_installed_references.json)。
这确认两处缺失纹理引用已补齐，没有证明整个场景渲染或占据地图已完成。

### TRACE 的可读取性与接口

三个 checkpoint 均用当前 Torch 的 `weights_only=True` 在 CPU 安全读取，
每份 state dict 有 812 项，全部浮点权重有限。
参数/缓冲元素量：ORCA 47,040,528，另外两份各 47,071,888。
校验不包含完整 TRACE 轨迹推理或其旧版训练环境。

配置保留了完整输入归一化：30 帧历史、0.1 秒采样、52 帧未来、100 步扩散；
ORCA 地图是 2 通道，nuScenes 两个模型是 7 通道，raster 为 224×224。
接入 CrowdSim 时必须按相应配置构建历史、邻居、地图、有效性掩码、局部坐标与目标引导。
当前 SFM 规划器没有自动切换成 TRACE。
[TRACE 官方说明及权重入口](https://github.com/nv-tlabs/trace)

完整记录：
[trace_download_manifest.json](resource_completion_20261001/trace_download_manifest.json)、
[trace_weight_verification.json](resource_completion_20261001/trace_weight_verification.json)。
模型许可 CC-BY-NC-SA-4.0 的原文件随解压目录保存。
原始代码的依赖栈为 Python 3.8 / Torch 1.10.2 / CUDA 11.1；
本轮只保管其源码与资源，没有在 Sim 6 环境安装旧依赖。

### PACER 权重与示例资源

已按[官方脚本](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/download_data.sh)
下载 `pacer` 和 `pacer_no_shape`，另补齐
`amass_isaac_standing_upright_slim.pkl`、`amass_isaac_gender_betas_unique.pkl`、
`amass_copycat_occlusion_v2.pkl`。完整实际文件名与哈希以
[pacer_download_manifest.json](resource_completion_20261001/pacer_download_manifest.json)为准。
示例 pickle 只保管并校验字节，没有执行反序列化。
原 LICENSE 与归属说明在隔离目录保存。

当前 Torch 安全读取两份模型和 normalizer 均通过，浮点值有限、方差非负。
按照官方 ReLU MLP 结构完成四行合成归一化输入的 CPU actor 前向，输出均为有限的
`(4,69)`。这是网络与字节兼容检查，尚未验证 PACER 在 Isaac Lab 的物理控制。

| PACER 变体 | 原始观测 | self / task | 编码后 actor 输入 | AMP 历史观测 | 动作 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_shape | 1401 | 357 / 1044 | 613 | 1950 | 69 |
| shape | 1412 | 368 / 1044 | 624 | 2060 | 69 |

1044 维 task 包括局部未来轨迹和高度图。这些输入、normalizer、PD 映射与 MaskedMimic
的条件 VAE/Transformer 输入不同。两者动作同为 69 维，不能据此直接换 checkpoint。
证据：[pacer_weight_verification.json](resource_completion_20261001/pacer_weight_verification.json)。

可复用下载器：[fetch_pacer_reference.py](../scripts/fetch_pacer_reference.py)，
支持有界下载、严格 Range 续传、已校验文件复用与原子发布；不会安装上游依赖。
`third_party/pacer/output` 和 `sample_data` 已相对链接到对应资源。

### 第一作者的公开模型快照

从第一作者 [ProtoMotions fork](https://github.com/leele-bing/ProtoMotions/tree/8702e7919d33d7f264682a28c5c120934c8cc267/data/pretrained_models)
固定 revision `8702e7919d33d7f264682a28c5c120934c8cc267` 下载两套模型。
每套包含 `last.ckpt`、`experiment_config.py`、`resolved_configs.pt/.yaml`、
`resolved_configs_inference.pt/.yaml`，共 12 文件、362,205,589 B；
每个 LFS 对象均与官方精确大小及 SHA256 OID 对齐。

两份 checkpoint 均安全读取，全部模型浮点值有限；模型 state dict 的键及形状
与当前对应模型一致。但旧配置仍引用 `protomotions.agents.masked_mimic.config`
等旧 API，而当前模块使用 `agents.supervised.masked_mimic_config`。
本轮未执行旧配置、切换当前权重或验证这些快照的物理控制；
相同张量形状不能单独证明配置和观测语义一致。

这些公开对象属于 MaskedMimic / motion tracking，不能改名当作论文导航 AMP。
证据：[下载清单](resource_completion_20261001/navisaaclab_public_download_manifest.json)、
[权重读取检查](resource_completion_20261001/navisaaclab_public_weight_verification.json)、
[张量形状对照](resource_completion_20261001/author_weight_shape_compatibility.json)。

## 3. 已有核心资源

| 资源 | 当前使用位置 |
| --- | --- |
| Neutral SMPL 参数 | `data/smpl/SMPL_NEUTRAL.pkl`，现有本地模型软链接 |
| Male / Female SMPL 参数 | `data/smpl/SMPL_MALE.pkl`、`SMPL_FEMALE.pkl`；本轮验证既有 licensed 模型后补建软链接 |
| 当前控制器 | `data/pretrained_models/masked_mimic/smpl/last.ckpt` 及配套配置 |
| AMASS 验证动作 | `../Assets/motion/amass_smpl_validation.pt` |
| AMASS train/test | 同目录 `amass_smpl_train.pt`、`amass_smpl_test.pt` |
| 物理人体 | `protomotions/data/assets/mjcf/smpl_humanoid.xml`，24 bodies / 69 DOF |
| 独立行人场景与地图 | `CrowdSim/config/pedestrian.yaml`、`CrowdSim/maps/pedestrian_open.*` |

原始 AMASS 和三个已处理 split 已存在；通用 11,786 段数据不等于论文的
2,848 段精确 locomotion 子集。本轮没有重复下载大型 AMASS 或当前预训练模型。

## 4. 运行已验证的带纹理行人

在仓库根执行：

```bash
source /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/activate
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m CrowdSim.humanoid.simulate \
  --headless --num-humanoids 4 --steps 240 --reset-at 120 --capture \
  --output-dir output/my_textured_pedestrians
```

GUI 入口：

```bash
python -m CrowdSim.humanoid.simulate --num-humanoids 4
```

默认自动发现 `data/smpl/smpl_uv.npz` 和 `data/smpl/textures/`。
运行结果会报告 `mesh_appearance`、纹理池和逐人分配，输出 RGB 帧、MP4、
真实 root/joint 状态和 `result.json`。

本轮 8 秒实跑结果：4 人、240 步、30Hz、80 帧、10fps，`status=passed`，
`mesh_appearance=textured`，一次子集重置，最低骨盆高度 0.734768 m；
30 个观测归一化缓冲区保持冻结，渲染没有推进物理时间。
四人实际行走距离分别约 3.586、4.821、4.140、2.651 m。

- [运行结果](../output/resource_validation/textured_pedestrians/result.json)
- [视频](../output/resource_validation/textured_pedestrians/pedestrian_motion.mp4)
- [渲染帧](../output/resource_validation/textured_pedestrians/frame_000120.png)

### 200 秒验证与修复

首次 6,000 步验证在第 862 个物理步附近、约 28.7 秒触发骨盆低于 0.35 m 的检查；
当时已发生 6 次重置。原始失败结果与帧已保留：
[失败记录](../output/resource_validation/pedestrians_200s/result.json)。
已确认并修复以下三个问题：

1. **骨盆目标随实际下落高度下移**：显式 pedestrian 配置改为未来目标处地面上方
   0.93 m，避免垂直恢复误差恒为零。
2. **通用 AMASS 随机动作用于行人重置**：配置按真实路径选取 9 条 Walking 动作，
   检查首帧高度、向上方向、垂直速度、有限值，再与原排除规则取交集。
3. **重置返回旧路径的首帧观测**：在新导航状态准备好后，只刷新重置行的 context
   与观测，再返回当前观测。

原 AMASS、checkpoint 与 0.35 m 失败检查保留。
9 条动作只是当前演示的已验证重置集合，不等于论文 2,848 段训练子集。
分析和回归见
[pedestrian_fall_analysis.md](resource_completion_20261001/pedestrian_fall_analysis.md)。
修复回归 28 项及整个 CrowdSim 回归 **154 项**均通过；28 项包含在 154 项内。

随后在相同 seed=7、四人共享世界下完成 **6,000 步 / 200 秒**实跑：

| 指标 | 结果 |
| --- | --- |
| 状态 | `passed`，进程 exit 0 |
| 控制 / 渲染采样 | 30Hz / 2Hz |
| 最低骨盆高度 | 0.826277 m |
| 重置 | 11 次 |
| 四人行走距离（扣除重置瞬移） | 114.838 / 147.093 / 117.937 / 106.289 m |
| 图像 / MP4 | 400 帧 / 1,260,510 B |
| 归一化统计 | 30 个缓冲区保持冻结 |
| 真实状态 | root position、XYZW quaternion、69 DOF、motion ID/time、reset mask 全部有限 |

- [最终运行结果](../output/resource_validation/pedestrians_200s_repaired/result.json)
- [完整 200 秒视频](../output/resource_validation/pedestrians_200s_repaired/pedestrian_motion.mp4)
- [最终状态数据](../output/resource_validation/pedestrians_200s_repaired/physical_states.npz)

复跑命令：

```bash
python -m CrowdSim.humanoid.simulate \
  --headless --num-humanoids 4 --steps 6000 --capture --capture-every 15 \
  --output-dir output/my_pedestrians_200s
```

该结果证明当前配置、seed、四人开阔场景的长期运行通过。
没有验证论文八场景、3–15 人、Diffusion+AMP 的完整评估指标。

## 5. 严格论文复现仍需补齐的内容

1. **定制 trajectory diffusion**：matching checkpoint、代码、归一化、地图/邻居协议、
   目标引导和 5 秒轨迹采样。已下载 TRACE 可以作为迁移参考。
2. **SMPL 导航 AMP**：matching actor/critic/discriminator、观测统计、24 bodies / 69 DOF
   排序、PD offset/scale、奖励和时间协议。PACER 公共权重需要独立适配，
   不能装入当前 MaskedMimic 网络。作者另一个 AMP fork 的 105 维输入 / 28 维动作
   属于不同人体任务。
3. **2,848 条 AMASS 清单**：clip ID、截取区间、过滤、采样和预处理版本；
   现有完整数据不能单独证明该子集一致。
4. **八场景和评估 manifest**：论文场景 IDs、USD 导出、占据地图、episode/seed、
   3–15 人配置与 200 秒评估协议。当前六个 DynBench 目录不能直接当成论文八场景。
5. **剩余仓库场景引用与地图**：KLT `small_KLT_visual_collision.usd`、Forklift 完整子树，
   以及 Full_Warehouse、Hospital、Office、Warehouse 四份对应场景的 occupancy YAML/PNG。
   地图需要按真实碰撞场景生成。
6. **更广泛控制评估**：当前四人开阔场景 200 秒已通过；其他 seed、密集人群、
   目标真实场景及 PACER/TRACE 的物理接入仍需对应测试。

以上资源与工程工作决定能否复现论文方法及指标。
当前带纹理短时行人演示、下载成功与 checkpoint 可读取分别有独立证据。
