# Testing Patterns
审查：2026-10-10；基于本机文件、安装元数据、AST 和 SHA。此次未启动 Kit、训练或物理重测。完整机器清单见 [inventory.json](../resource_audit_20261010/inventory.json)。

## Core Sections (Required)

### 1) Test Stack and Commands

根pytest testpaths为tests、src/CrowdSim/tests、src/protomotions/tests；pythonpath为src与根，strict-markers同时兼容pytest8/9。当前安装9.0.3。公共CI仅明确检查项目布局/映射/AST。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:. .venv/bin/python -m pytest --import-mode=importlib \
  -p no:cacheprovider -q tests/test_project_layout.py
python scripts/manage_release_assets.py verify
.venv/bin/python -B -I scripts/isaaclab3_environment.py check
.venv/bin/python -m social_navigation preflight --scene YOHO
```

第一条为CI示例，此次未重跑；后三类检查此次有实际记录。写入输出的CLI预检应指定新目录，库级verify_native_scene_config此次仅只读核验。

### 2) Test Layout

本轮统计src内169个测试文件、根tests154个；data/scripts/examples另列。AST共983个文件均可解析。需要LFS、Torch或SDK的测试不能由最小CI依赖全部执行。[静态范围](../resource_audit_20261010/inventory.json)

### 3) Test Scope Matrix

| 层次 | 本轮结果 | 不能证明 |
| --- | --- | --- |
| 源AST | 983文件，0语法失败 | 算法/API/原生内存行为 |
| core/lock元数据 | 18core满足；272非virtual锁版本一致 | Sim全部metadata无冲突、GPU执行 |
| USD环境check | 234 RECORD文件通过；standalone26.8 round-trip PASS | Kit运行时USD版本或场景物理 |
| 动作清单 | v7 2848/22/34，文件SHA与父来源隔离通过 | 内容级去重、教师预训练无泄漏、作者原ID |
| 四场景地图路线 | 原source-bound配置4/4通过 | release新位置准入、新PhysX效果 |
| 发布资源 | 467文件SHA、14原USD gzip通过 | 重新搬迁后的接触/碰撞表现 |
| 成片媒体 | 4个视频ffprobe均60秒/1800帧/30FPS | 连续主观步态评价；本轮未重新全量解码 |
| 原生物理/训练 | 使用已有历史证据；本轮未重跑 | 新SDK、新模型或任意种子保证 |

### 4) Mocking and Isolation Strategy

CPU门禁不反序列化模型、不启动SDK，原生真实接触须另行实验。本轮P metadata查询移除ROS PYTHONPATH；device查询采用只读nvidia-smi。SDK自身源码不修改，pytest插件可关闭以减少环境污染。mock不替代实际physics/returncode。[资源预检实现](../../src/CrowdSim/humanoid/application/resources.py)

### 5) Coverage and Quality Signals

没有统一全套绿灯：同日兼容性审查记录全量测试存在历史fixture/门禁失败；此次没有重复全量pytest，也未测覆盖百分比。固定验证22条已用于选型，不是新的未见集；测试34条与公开PACER预训练是否重叠未知。已有四片完整解码和物理子步证据见交付文档，本轮仅核对身份/头信息。[交付范围](../FOUR_SCENE_SPEED_20261010_ZH.md)

### 6) Evidence

- [实际环境、文件身份、依赖和目录清单](../resource_audit_20261010/inventory.json)
- [本轮四场景预检与发布资源校验](../resource_audit_20261010/resource_checks.json)
- [环境/USD 检查与版本覆盖诊断](../resource_audit_20261010/environment_checks.json)

- [pytest设置](../../pyproject.toml)、[CI](../../.github/workflows/public-ci.yml)、[最新交付索引](../four_scene_speed_final_delivery_20261010.json)
- [兼容性审查原始记录](</home/lfm/Documents/ChatGPT/Social Navigation/isaaclab_compatibility_audit_20261010/ISAACLAB_COMPATIBILITY_REVIEW_ZH.md>)

## Clone portability update — 2026-10-10

The current four-scene flow uses `configs/reproduction/four_scene`, exact in-memory resource bindings, the committed SDK bootstrap/native shutdown bundle, fixed-bank rebuild templates and `scripts/run_four_scene_benchmark.py`. See [clone reproduction](../../CLONE_REPRODUCTION.en.md) and [validation](../CLONE_VALIDATION_20261010.json). Historical experiment receipts remain provenance, not new-run configuration.
