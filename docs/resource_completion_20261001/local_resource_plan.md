# 行人资源补全计划（2026-10-01）

**最新结果以 [资源补全与运行记录](../NAVISAACLAB_RESOURCES_AND_REPRODUCTION_20261001_ZH.md) 为准。**
后续已补齐 UV、两张纹理、七份公开参考权重、完整配套配置和两份 HDR；
HDR 已建立到原场景引用的相对链接，四人带纹理 200 秒实跑通过。
本文其余部分保留最初资源计划与中间时点状态，不用于断言这些文件现在仍缺失。

**后续下载已完成：** 两份官方CC0 4K HDR已保存到`data/scene_resources/hdri/`，大小/MD5/Radiance头验证通过；详见[独立下载清单](hdri_download_manifest.json)。场景原引用路径尚未接入。以下原只读审查记录已补充这一更新。

本轮只读检查本地文件、当前配置和官方来源；未下载、移动资源、修改共享配置或启动 SDK/Kit。机器证据及每个资源的绝对路径、大小、mtime、依赖、来源见 [local_resource_plan.json](local_resource_plan.json)。大文件没有重复哈希或反序列化。

## 1. 当前行人已能运行，核心资源无需再下载

当前独立入口 `CrowdSim/config/pedestrian.yaml` 使用本地 `pedestrian_open`，控制链是 **SFM + SMPL MaskedMimic + IsaacLab PhysX**。最新既有 [真实运行结果](../../output/completion_validation/smpl_render_final/result.json) 为四人、240步、8秒、80帧，`status=passed` 且 `mesh_created=true`。这验证了现有纯色蒙皮、动作和控制链；论文精确方法与场景还没有完成。

| 本地目标 | 当前字节数 | 当前核对与动作 |
| --- | ---: | --- |
| `data/smpl/SMPL_NEUTRAL.pkl` | 247,186,228 | 软链接解析到 arena 原始 v1.1.0；真实 CPU LBS/PhysX 网格已通过，复用 |
| `data/pretrained_models/masked_mimic/smpl/last.ckpt` | 229,131,184 | 配套 `.pt` 配置46,693字节；此前严格权重加载及当前实跑均通过，复用 |
| `../Assets/motion/amass_smpl_train.pt` | 9,565,828,107 | 当前stat与先前全面验证一致，复用 |
| `../Assets/motion/amass_smpl_validation.pt` | 678,009,566 | 当前首/中/末动作与资产最大误差约0.000526毫米，复用 |
| `../Assets/motion/amass_smpl_test.pt` | 65,614,188 | 当前stat与先前全面验证一致，复用 |
| `protomotions/data/assets/mjcf/smpl_humanoid.xml` | 23,780 | 24 bodies/69 DOF exp_map；真实Lab3 FK亚微米一致，复用 |
| `CrowdSim/maps/pedestrian_open.yaml` / `.png` | 121 / 693 | 本地入口已验证，无外部场景/HDR要求 |

8套现有预训练的 `last.ckpt`、配套配置仍在；此前8/8严格加载通过，此轮复核stat。**不用重拉 AMASS、SMPL neutral 或预训练。** 原始AMASS22包共 22,903,743,464 字节仍保留。通用11,786段split不能当作论文2,848段locomotion精确清单。

male/female模型源文件也已在 `arena_sangfor/arena_ws/arena_assets/smpl/SMPL_python_v.1.1.0/smpl/models/`，约247MB/份；`data/smpl/SMPL_MALE.pkl`、`SMPL_FEMALE.pkl` 尚未接入。它们只在重新转换相应性别AMASS等工作中需要，当前neutral渲染无需补下载。[项目AMASS说明](https://protomotions.github.io/getting_started/amass_preparation.html)要求相应命名。

## 2. 人体外观补全：UV 与图片需要配套

| 缺失目标 | 正式来源 | 大小、门槛与下一步 |
| --- | --- | --- |
| `data/smpl/smpl_uv.npz` | [SMPL官方 Downloads](https://smpl.is.tue.mpg.de/) | 官方提供UV OBJ，需要注册、登录与license；当前没有授权二进制直链或大小。拿到OBJ后转`uvs`/`uv_faces`，核对与6890顶点、13776面SMPL拓扑/face角顺序一致 |
| `data/smpl/SMPLitex-textures/` 内图片 | [作者数据集页](https://dancasas.github.io/projects/SMPLitex/SMPLitex-dataset.html)、[官方ZIP](https://drive.google.com/file/d/1y7isceC55NZRlNr9KdDLczpaKLXa3nyl/view?usp=sharing) | 官方页面还提供逐张图片；可先选几张做有限预览。远程字节数未验证，网页/README未找到明确dataset许可，Drive可能有限流/登录 |
| 可选单图`data/smpl/smpl_body_texture.png` | 同上，或明确授权的相同UV布局图片 | 当前加载器也接受PNG/JPG/JPEG/TIF/TIFF；单图不能代替UV |

[一张真实作者PNG链接](https://dancasas.github.io/projects/SMPLitex/SMPLitex-dataset/textures/SMPLitex-texture-00000.png)已从页面确认，HEAD因TLS超时没有取得大小。运行时自动选择 `data/smpl/SMPLitex-textures`，也支持 `CROWDSIM_SMPL_TEXTURE_DIR` / `CROWDSIM_SMPL_TEXTURE_PATH`。**仅图片不够；没有UV时仍使用已验收的纯色材质。** 仓库没有可直接执行的`tools/generate_smpl_uv.py`；UV OBJ→NPZ转换与拓扑验收是后续接入工作。

[SMPLitex生成模型](https://github.com/dancasas/SMPLitex)用于生成新图片，已经提供好的纹理不需要下载它，也不需要其SD/Diffusers环境。SMPL模型下载要求注册和许可，商业用途另有官方许可入口。[SMPL官方说明](https://smpl.is.tue.mpg.de/)

## 3. 现有场景仍有4个外部引用缺口

以下精确路径均在本轮重新stat，仍不存在；依赖链依据之前的递归Sdf检查，本轮没有打开完整Stage。

| 精确目标（相对`../Assets/`） | 使用场景 | 正式来源与处理 |
| --- | --- | --- |
| `DynBench/Props/KLT_Bin/small_KLT_visual_collision.usd` | warehouse、bidirectional_open、dynabench_warehouse | [Isaac资产访问](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/accessing_assets.html)；先确认6.0目录确实有所需`visual_collision`变体，不能用文档的`small_KLT_visual.usd`当同一文件 |
| `DynBench/Props/Forklift/forklift.usd` | dynabench_full_warehouse | [6.0官方SDG示例](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_scene_based_sdg.html)明确`Isaac/Props/Forklift/forklift.usd`；通过配置的资产根API定位，再补完整子树 |
| `NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr` | dynabench_hospital | [Poly Haven原资产](https://polyhaven.com/a/abandoned_parking)，CC0；选择4K **HDR** |
| `NVIDIA/Assets/Skies/Clear/sunflowers_4k.hdr` | dynabench_office | [Poly Haven原资产](https://polyhaven.com/a/sunflowers)，CC0；选择4K **HDR**，不是`sunflowers_puresky` |

后续通过现有代理读取官方API并完成下载：Abandoned Parking **25,661,358字节**，Sunflowers **26,582,747字节**；两者MD5均与官方一致，SHA256和4096×2048 RGBE头信息见[下载清单](hdri_download_manifest.json)。文件在`data/scene_resources/hdri/`，上表原场景引用目标仍缺；尚未创建共享Assets链接或改USD。KLT/Forklift远程字节数仍未知。

本地已有 `DynBench/Materials` **1,066文件 / 873,966,832字节**，`DynBench/SkyTexture` **10文件 / 172,015,477字节**。这10张天空不是所需两张的同名副本；其他天空可以在明确的覆盖层中作为演示选择，但不能冒充严格场景恢复。整包重复下载不会自动补齐场景越目录引用。

NVIDIA资产只通过 `isaacsim.storage.native.get_assets_root_path()` 和 `path_join()`解析当前根。**不拼CDN或bucket对象地址。** 离线选择[6.0.0官方完整资产包表](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/download.html)，5分卷MD5已记录JSON；当前无需为了`pedestrian_open`下载整个包。完整包每卷须校验，解压后根应有`Isaac`和`NVIDIA`；材料适用NVIDIA资产许可。最新版已是6.1，不能据此替换本项目6.0目标。

## 4. 四份地图需要生成，未发现正式成品下载

下面4份YAML目前不存在，其引用的占据图片也需要同时生成：

```text
/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Full_Warehouse/map/occupancy.yaml
/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Hospital/map/occupancy.yaml
/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Office/map/occupancy.yaml
/project/EXP/SangforAndSaut/Assets/DynBench/isaacsim_scene/Warehouse/map/occupancy.yaml
```

先修复目标场景引用、确认实际碰撞几何，再生成栅格和YAML，核对`image/resolution/origin/yaw/negate/thresholds`及world↔grid变换。`newmap.yaml/World1.png`和`pedestrian_open`属于别的布局，不能直接复制去填这4个路径。对应USD主文件存在不能证明地图或完整材质已到位。[6.0官方Occupancy Map说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/digital_twin/ext_isaacsim_asset_generator_occupancy_map.html)明确从场景碰撞几何生成。

## 5. 论文8场景仍缺精确manifest

[论文](https://arxiv.org/abs/2606.26265) III-A说场景来自OmniGibson/BEHAVIOR-1k；IV-D在8场景、3–15行人、每场景200秒评估。Fig.3只展示Grocery、Residence、Cafe、Hall、School五个例子。当前本地DynBench是Hospital、Full_Warehouse、Warehouse_multiple_shelves、Jetracer、Warehouse、Office六目录，无法据此映射到论文八场景。

正式DynBench来源可由[发布者FLUX README](https://github.com/Zeying-Gong/FLUX)定位到[zgong313/DynBench](https://huggingface.co/datasets/zgong313/DynBench)；这是共享场景来源核对，本计划不处理FLUX/NavDP内部模型。本地数据集card声明Apache-2.0，但其中NVIDIA/第三方资产各自许可仍需保留。HF页/API当前访问失败，不能确认完整remote大小、是否含这些缺失Props或地图。不要改用搜索得到的其他人同名镜像就宣称来自作者。

[BEHAVIOR官方安装](https://behavior.stanford.edu/getting_started/installation.html)明确需要dataset许可接受。原始BEHAVIOR整套下载也不等于作者挑选/导出的八场景，需要作者场景IDs、USD exports、maps、episode/seed清单；目前没有已核实的对应NavIsaacLab资源包链接或大小。[作者项目入口](https://broln7.github.io/NavIsaacLab-web/)本工具访问失败，未尝试登录或绕过。

严格论文方法还没有对应的：轨迹扩散checkpoint、SMPL导航AMP checkpoint/观测统计、2,848 locomotion clip清单、轨迹训练与评估episode manifest。未确认正式可下载产物时，不猜权重文件名/URL；现有SOMA/G1 AMP、GPC prior、MaskedMimic也不能替代对应论文模型。

## 6. LFS资源按实际使用补齐

49个ProtoMotions USD指针仍在，准确`oid`及目标大小见JSON。当前行人走MJCF转USD并建立自己的SMPL网格，不要求全部49个二进制；若选择直接USD/marker/projectile等分支，则从[官方ProtoMotions](https://github.com/NVlabs/ProtoMotions)对应版本取所需LFS对象，验证`sha256=oid`和目标长度，并再查USD子依赖。例：`smpl_humanoid.usda`目标114,283字节、`terrain_material.usda`5,908字节、`facing_direction_marker.usd`5,573字节。安装Python wheel不包含全部SMPL参数/权重/AMASS资源。

默认`CrowdSim/config/env.yaml`还开启Nova Carter和机器人PPO：`output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt`及本地`../Assets/Robots/NovaCarter/nova_carter.usd`候选仍缺。Nova Carter可走[官方资产目录](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/nova_carter_landing_page.html)远程回退，但本轮没有验证可达性；机器人PPO未确认正式外部下载入口，需要匹配训练产物。独立`pedestrian.yaml`没有机器人，因此不受这两项阻挡。

## 7. 后续优先顺序

1. **现有四人纯色模拟**：复用现有资源，已具备，无需下载。
2. **人体真实贴图**：先获取已有许可的SMPL UV OBJ并转换验收，再获取少量作者纹理；确认纹理条款与source/hash后接入。
3. **选择目标真实场景**：医院/办公室先补两份CC0 HDR；仓库通过官方资产根确认精确Props和子依赖；生成所选场景地图。
4. **论文精确复现**：取得八场景manifest和matching diffusion/SMPL AMP/动作清单，再按manifest补齐。不要先拉取整套同名数据当作完成。

最初只读审查仅写计划；后续经授权增加了两份官方HDR和独立来源/校验记录。其余远程大小未知和访问失败均如实标注，不代表源不存在。最新真实验收证明当前控制链短时运行，没有证明论文200秒密集人群指标。
