# 2026-10-10 资源与依赖审查过程

## 范围

只读源码/文件、完整 SHA256、AST、发行包元数据和 SDK-free 门禁。文档写入是本次唯一项目修改。未安装依赖、未获取/上传大文件、未反序列化动作库或权重、未启动 Kit/物理/训练。

## 执行记录

1. acquire-codebase-knowledge scanner 从项目根生成 docs/codebase/.codebase-scan.txt；宽范围包含第三方/历史文档/巨型JSON，不作为生产代码规模。
2. collect_inventory.py 读取三个 pyproject、SDK273项 lock、工作/目标环境metadata、AST983文件、目录逻辑大小、固定v7及早期manifest动作库完整SHA、四场景模型/地图/probe SHA、Git源revision与视频头信息，写入inventory.json。
3. 清除继承ROS PYTHONPATH后再次采集两环境metadata。uv管理272包；Python搜索metadata额外包含src egg-info同版重复和独立crowdsim-ppo，故记录数不是SDK包数量。以规范发行名检查18core和272非virtual锁版本。
4. 默认sandbox的uv缓存锁无法写入/home/lfm/.cache/uv；设置UV_CACHE_DIR=/tmp/social_navigation_resource_dependency_uv后只读重试成功执行，得到明确3项metadata覆盖冲突。该缓存错误已解决，不作为环境依赖缺失。
5. sandbox内nvidia-smi无法访问设备；允许只读设备检查后返回RTX4080SUPER、driver595.99.02、16376MiB。没有CUDA计算。
6. 在SDK Python执行 social_navigation.preflight.verify_native_scene_config，YOHO/NewMetro/SecondFloor/ThirdFloor原配置全部通过，人数4/4/1/1；sdk_started和live_physx_execution_verified均为false。
7. social_navigation.asset_release.verify校验467文件尺寸/完整SHA及14份原USD gzip解压SHA，status=verified。未重复--check-usd依赖闭合；使用同日已发布的USD闭合/远程独立取回凭据。
8. SDK Python -B -I执行 scripts/isaaclab3_environment.py check：包版本、234 USD RECORD文件、USD26.8 round-trip通过；Kit未启动。
9. 只读调用 require_checkout：SDK HEAD5951d48与profile4ecd0b不同，gate失败。git diff两提交仅pyproject.toml/uv.lock变化，SDK工作树无未提交改动；未执行apply/sync。
10. 对可选模块只做find_spec，不导入/运行；确认ORCA生成、重定向、下载器、文档构建等缺包。4个成片做ffprobe读取时长/帧数/FPS/尺寸，不重复完整解码。
11. HF本地publication_status仍待有效CLI身份，记录上传0文件。Hugging Face只读插件对指定私有模型ID返回not found；无法区分不存在与访问权限，没有再次登录、读取令牌或上传。
12. 六份ORCA语料manifest与10,072条文件引用的尺寸/SHA通过；另核对旧/补齐版CPU投影地图状态和外部DynBench/NVIDIA/场景树。未加载窗口张量，未重跑ORCA。
13. 生成七份codebase文档、中央资源报告，修正双语指南中的发布权重状态、SDK revision补充和实际AMASS位置；另做七份核心章节、相对文件链接与JSON证据断言检查。

## 关键命令（仓库根）

```bash
NAV_PY=/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" -B -I scripts/isaaclab3_environment.py check
UV_CACHE_DIR=/tmp/social_navigation_resource_dependency_uv uv pip check --python "$NAV_PY"
python scripts/manage_release_assets.py verify
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

collector为本机路径的审计采集器，未实现跨主机路径迁移。用新的输出目录重跑，避免覆盖封存清单：

```bash
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" -B \
  docs/resource_audit_20261010/collect_inventory.py \
  --output /tmp/social_navigation_resource_audit_new
```

输出目录必须不存在。采集可能读取约数十GB进行SHA与文件清点，结果对应运行时本机状态，不保证与旧snapshot相同。此collector整合了本轮分别运行的本地检查；HF插件查询和sandbox外设备查询由工具单独执行，原始结果保存在本次inventory。轮次的审查日期固定为2026-10-10，不把重跑当作原轮次物理证据。

## 输出

- inventory.json：依赖声明、实际metadata、选定资源全SHA、逻辑目录量、外部仓库revision、视频头信息及缺口。
- resource_checks.json：4/4 source-bound原生地图/路线预检、发布文件完整性。
- environment_checks.json：metadata覆盖冲突、USD环境检查、manifest/lock身份、revision gate和设备/系统工具记录。
- ../codebase/*.md：七份结构与依赖文档。
- ../RESOURCE_DEPENDENCY_MAP_20261010_ZH.md：中央总览。

完整wheel/sdist URL与hash位于env/isaaclab3_sim6/uv.lock；inventory避免重复整个wheel列表，保留lock文件SHA和273条包来源/依赖/版本。
