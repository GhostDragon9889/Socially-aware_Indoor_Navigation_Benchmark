# 四场景地图与缺模型：交接状态

更新：父进程真实 Kit 已完成 public API 解析，随后执行了实际下载。KLT 五文件全部验证通过；Forklift 五文件下载后，唯一 normal texture 已从原 Full_Warehouse 本地场景包接入（三份原包同 SHA），全部六文件及完整本地闭包验证通过，属于 mixed source。原远端 timeout/HTTP000 证据保留，没有假造该纹理 HTTP200。两个相对链接已由父进程创建，四张资源完整的新 CPU 图已生成，见[当前地图清单](maps_projection_resource_complete_inventory.md)和[实际接入报告](maps_projection_asset_download_result.md)。下文保留下载前的准备流程；旧地图/清单未覆盖，PhysX 验收待执行。

四份 CPU USD 投影地图已冻结，结果见[地图清单](maps_projection_inventory.md)；完整源文件/测试 SHA 与待补模型计划见[机器清单](maps_projection_missing_asset_plan.json)。未启动 Kit/GPU，未修改外部 Assets、SDK 或共享场景配置，尚未执行模型下载。

## 精确缺口及下载入口

| 缺资源 | 原 USD 需要的位置 | 官方根目录下相对路径 | 当前证据 |
|---|---|---|---|
| 叉车 | `../Assets/DynBench/Props/Forklift/forklift.usd` | `Isaac/Props/Forklift/forklift.usd` | Full_Warehouse 1 个引用；[Sim6 官方 SDG 示例](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_scene_based_sdg.html)使用该名称 |
| KLT 箱 | `../Assets/DynBench/Props/KLT_Bin/small_KLT_visual_collision.usd` | `Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd` | Warehouse 138 个引用；[NVIDIA 论坛原始帖](https://forums.developer.nvidia.com/t/why-physicsmaterial-destroys-the-klt-bin/248866)提供旧版精确名称，当前版本可用性未确认 |

模型大小、SHA 和下载状态均尚未知，不能凭路径名称宣称已补全，也不能宣称当前官方版本与作者使用的原始字节相同。Office/Hospital 的本地 composition 完整；两份 warehouse 因缺模型将原 free 候选保留为 unknown。四份都尚未做 PhysX occupancy 复核，也不是论文八场景地图。

遵照[官方资产访问文档](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/accessing_assets.html)，使用实际配置的 public storage API 根目录；没有硬编码或推导 CDN URL，没有改 profile、代理配置或用户存储设置。

## 父进程下一次已有 Kit 中调用

`scripts/resolve_paper_scene_assets.py` 模块导入不会启动 SDK；函数先取得当前 `omni.kit.app.get_app()`，调用已有 extension manager 的 `set_extension_enabled_immediate("isaacsim.storage.native", True)`，确认启用成功后才导入 storage public API。physics headless experience 默认未启用该扩展，直接 import 会失败。下面调用只使用父进程已运行的应用：

```python
import runpy
resolution = runpy.run_path("scripts/resolve_paper_scene_assets.py")["resolve_requests"]()
```

无参输出：`data/scene_resources/dynbench_missing/official_api_request.json`。返回值仅含 JSON-safe 路径、两个相对名称及 `availability_asserted=False`；不会把整份 URI 请求打印出来。`get_assets_root_path(skip_check=True)` 只解析当前配置，不证明文件可下载。独立 Python 进程不能调用该 resolver 来抢占当前 Kit。

实际 request 文件落盘后，纯 CPU 下载命令为：

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/download_dynbench_missing.py
```

输出保留原 `Isaac/Props/...` 层级于 workspace `data/scene_resources/dynbench_missing/`。脚本拒绝缺 request、错名称/实际路径、根目录外依赖、含认证或 query 的 URI；每次 curl 至多一次 retry，默认单文件 64 MiB、每模型累计 512 MiB、256 个依赖，connect 10 秒、单次 45 秒。仅使用本命令的 localhost7897 代理参数，可 `--proxy ''` 使用直连，不修改机器配置。

下载后记录 HTTP 状态、实际 redirect URI、magic、大小与 SHA256，递归解析真实 USD composition、asset arrays/time samples 与 MDL 静态资源；已存在文件只有旧 manifest URI/hash 与实际文件一致才复用。USD composition 检验在单独纯 CPU 进程中限时 60 秒；缺失、绝对远程 composition、未知 MDL 模块、超限及访问错误都保留 incomplete，不改写原场景，不自动换资源版本。

当前安装 OmniPBR/OmniGlass/OmniSurface 已实查 15/9/14 文件的静态 MDL source closure；这不等于 MDL compiler/RTX 成功。编辑器 thumbnail metadata 不算渲染纹理依赖；动态 MDL 资源仍需后续渲染核验。下载资产没有 CC0 声明，实际资源须保留其分发许可/通知。

## 外部 Assets 链接计划与验收

仅当下载 manifest 对该模型为 `downloaded_local_dependencies_verified`、完整本地 composition 可用时，父进程才能按审批流程创建以下外部链接；目前未执行：

- `../Assets/DynBench/Props/Forklift` → 当前 repo `data/scene_resources/dynbench_missing/Isaac/Props/Forklift`
- `../Assets/DynBench/Props/KLT_Bin` → 当前 repo `data/scene_resources/dynbench_missing/Isaac/Props/KLT_Bin`

不得覆盖已有内容。链接后重新核对原 USD 的真实解析路径、重生成两份 warehouse 地图；GPU 空闲后由单实例 Kit 做实际 PhysX collider/omap 和 world/grid/Y-flip 验收。若当前官方根下 exact KLT 不存在，保留 unknown 并另找可核实版本，不造相似箱子。

## CPU 验证

地图 6 项 CPU 回归已通过；资产 helper 9 项 CPU 回归通过（0.42 秒），[JUnit](maps_projection_asset_helpers_tests.xml)，Ruff `--target-version py312` PASS。包括真实 USD composition+texture arrays/time samples、URL scope/encoded traversal、MDL namespaced/relative imports、实际 SDK MDL source closure、厘米/Y-up 真几何和缺 composition 拒绝。另三个隔离测试只验证“已有 app→立即启用→导入 API”顺序及启用失败不写 request；不证明实际 API 或模型可用，实际 Kit 验收仍由父进程执行。没有用假下载结果作资源可用性验收。

扩展身份已读本地 Sim6 官方 `isaacsim.storage.native/config/extension.toml`：版本 1.9.2，python module `isaacsim.storage.native`，依赖 `isaacsim.core.version`、`omni.client`。官方 `isaacsim.core.experimental.utils/impl/app.py:245–270` 的 `enable_extension()` 使用相同的 immediate extension manager API，并将返回值定义为是否成功。没有新增 SDK bootstrap，也没有回退到硬编码 endpoint。
