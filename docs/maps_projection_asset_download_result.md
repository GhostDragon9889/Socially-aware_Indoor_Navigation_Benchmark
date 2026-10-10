# DynBench 缺模型：下载、本地接入与校验结果

当前更新：父进程已经按审批创建并实核两个外部相对链接，四个场景 composition/纹理/MDL 路径闭包全部可用，四份新 CPU 地图已生成到 `data/diffusion_amp/maps_resource_complete/`；[当前地图清单](maps_projection_resource_complete_inventory.md)。尚待 PhysX/RTX 实际验收。

下载依据为父进程真实 Kit 写入的 `data/scene_resources/dynbench_missing/official_api_request.json`，通过当前配置的 public storage API 解析。叉车的唯一剩余 normal texture 已从原本地 Full_Warehouse 场景包接入，属于 mixed source。未硬编码替代 endpoint，未启动第二 Kit/GPU，本 agent 未修改外部 Assets/SDK。[逐文件完整 SHA、实际原始错误与链接计划](maps_projection_asset_download_result.json)；原始失败传输 manifest 仍保留在 `data/scene_resources/dynbench_missing/download_manifest.json`，当前完整本地状态另存 `local_resource_manifest.json`。

## 结果

| 模型 | 文件数 / 实际字节 | 状态 | 尚缺 |
|---|---:|---|---|
| KLT_Bin | 5 / 22,217,132 | `downloaded_local_dependencies_verified` | 无 |
| Forklift | 6 / 60,302,676 | `mixed_source_local_dependencies_verified` | 无 |

KLT 的 root、visual USD 与三个真实 4096×4096 PNG 完整到位；Forklift root、fork、materials USD 与三个 PNG 完整到位。两者完整本地 USD composition/资产属性路径递归无未解析项，OmniPBR 15 文件的已安装静态 MDL source closure 也通过。全部十一个下载/接入文件再次核对大小/SHA；六张 PNG 的 magic 和 Pillow verify 通过。没有把此前未下载纹理当作 HTTP 成功。

root USD：

- `Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd`，226,995 B，SHA256 `2e6172432c1b5ee2b7ea047229e070ccf1d49b69d01b4796c0445531af392bb9`。
- `Isaac/Props/Forklift/forklift.usd`，19,356,891 B，SHA256 `12765bfe269089369fe5541993cfd2768de0d70163a5aa03874073e767ad1feb`。

独立 CPU USD26.8 Stage 实测两者各 3 个 Gprim、default `/Root`、metersPerUnit=1、Z-up。KLT bounds m：`[-0.09892139,-0.14831734,-0.07317996]` 到 `[0.09892139,0.14831736,0.07317996]`。Forklift bounds m：`[-0.60690341,-1.74734005,-0.00190668]` 到 `[0.60690344,1.74734005,2.15303798]`。本地依赖/geometry gate 已通过，尚未执行 MDL compiler/RTX 或 PhysX collider 验收。

## 原本地 normal texture 接入

原有 `Assets/DynBench/isaacsim_scene/{Warehouse,Warehouse_multiple_shelves,Full_Warehouse}/Materials/Textures/T_Forklift_N.png` 三份文件全部为 18,120,694 B、4096×4096 RGBA、真实 PNG，Pillow verify 通过，SHA256 完全相同：

`22044c7b9ce49e1d8aaa34f36839c9953eabd8872dffdef3caaad7f673586266`

选择 Full_Warehouse 原场景包文件，非破坏复制到 workspace `Isaac/Props/Forklift/Materials/Textures/T_Forklift_N.png`；basename 与 downloaded materials USD 中唯一未解析的 authored path 一致。复制后再次核对源/目的 SHA、源 mtime 未改及完整本地闭包。逐源路径与证明见 `data/scene_resources/dynbench_missing/local_texture_completion.json`。

该文件来源为已有本地场景包，HTTP status 为 null，不能宣称是官方 6.0 远端同一字节，也不能宣称作者版本字节一致。源文件未修改，材质 authored path 和 USD 业务配置未修改。

## 保留的远端传输错误

四轮有界 dependency passes：HTTP localhost7897、SOCKS5h localhost7897、HTTP、显式直连。每次 curl 最多一次 retry、connect10s/max45s；后续只下载失败文件，其余文件严格 hash 复用。前三轮原始 manifest 分别封存为 `download_attempt_1_manifest.json` 到 `download_attempt_3_manifest.json`。

最后 Forklift normal texture 远端返回 `curl (28) Connection timed out after 10002 milliseconds`，HTTP `000`，没有任何 HTTP response。因此是连接失败，不能写成 HTTP404、对象缺失或许可拒绝。原 `download_manifest.json` 原始失败记录与 SHA 保持不变；已停止网络重试，通过上述确切本地源补全，而非猜其他名字或造纹理。目前本地资源已足够，不需要重复下载。

本轮还修正 `--proxy ''` 的直连语义：命令显式传给 curl 空 proxy 与 `--noproxy '*'`，防止环境里的代理变量被意外继承；没有改变用户代理配置。Ruff PASS。

## 外部链接：父进程已创建并验证

创建前原外部目录不存在且不是 symlink。父进程已在 `require_escalated` 审批下创建以下相对链接，无覆盖已有内容；本 agent 随后复核实际 symlink target 与两个 root USD SHA 全部匹配。本 agent 没有外部写入，记录见 `docs/diffusion_amp_20261001/scene_resource_links.json`：

```text
/project/EXP/SangforAndSaut/Assets/DynBench/Props/KLT_Bin
  -> ../../../ProtoMotions/data/scene_resources/dynbench_missing/Isaac/Props/KLT_Bin
/project/EXP/SangforAndSaut/Assets/DynBench/Props/Forklift
  -> ../../../ProtoMotions/data/scene_resources/dynbench_missing/Isaac/Props/Forklift
```

链接后原 Warehouse/Full_Warehouse USD 重新递归检查通过，新四张地图的结果及 SHA 见上述当前清单。此前两张仓库图作为历史快照保留没有覆盖。所有新地图仍待单实例 Kit 的实际 PhysX omap 验收。

这些是当前官方资源与已有本地场景包中匹配 exact relative filenames 的文件；作者资产版本/原始 hash 未公开，不能宣称与作者字节完全相同，不能据此宣称论文八场景评估已完成。NVIDIA 与原场景包的分发/source notices 应保留，没有 CC0 声明。
