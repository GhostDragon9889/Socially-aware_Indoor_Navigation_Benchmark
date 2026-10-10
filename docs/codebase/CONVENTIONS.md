# Coding Conventions
审查：2026-10-10；基于本机文件、安装元数据、AST 和 SHA。此次未启动 Kit、训练或物理重测。完整机器清单见 [inventory.json](../resource_audit_20261010/inventory.json)。

## Core Sections (Required)

### 1) Naming Rules

五个既有包名保留：social_navigation、CrowdSim、protomotions、crowdsim_ppo、deployment。函数/模块多用snake_case、类用PascalCase；CrowdSim的大小写是兼容合同。scene CLI使用四个固定短名。manifest/profile/result 使用明确schema和完整SHA，实际运行与候选记录分开。[包注册](../../pyproject.toml)

### 2) Formatting and Linting

根 pyproject 提供 Ruff/per-file ignores；pre-commit hook固定0.8.1，包含license、typos、尾空白、YAML和1MB普通Git文件限制。SDK启动后的导入有E402豁免，示例配置重导出等另有F401/F821/F811豁免。配置表达既有习惯，不意味着当前所有历史文件均通过lint；此次未执行自动fix。

### 3) Import and Module Conventions

src布局与根兼容链接同时存在；源码命令使用src和根PYTHONPATH，project-root resolver 可由 SOCIAL_NAVIGATION_ROOT 覆盖，资产由 PROTOMOTIONS_ASSET_ROOT 定位。SDK模块必须在适当启动后加载，勿按普通包一次性导入所有backend。本机shell继承ROS PYTHONPATH；此次采集venv元数据时剔除它，防止将ROS包算成锁依赖。

静态import扫描记录绝对import根及例子；函数内可选import、tests、本地脚本模块须按scope判读。`scripts.*` 和 docs terminal helper 真实存在，不是 pip缺包。原始TRACE/PACER参考checkout无需整个安装进当前SDK，核心算法已有src适配。[runtime加载](../../src/CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py)

### 4) Error and Logging Conventions

资源空值/缺失/LFS pointer/大小SHA不符抛明确异常；scene gate严格检查来源，不自动用任意文件fallback。训练/原生父进程保留stdout日志、execution JSON、源输入快照和实际returncode；CLI传递子进程退出码。日志与验证文件用于复核，不把“status=running”解释为完成。

### 5) Testing Conventions

test_*.py/pytest，root + 两包tests；needs_lfs marker受严格检查。CPU夹具可隔离SDK依赖，实际接触/物理与录像需原生进程/evidence另验。新增模型的SHA只能证明身份，不能直接继承旧模型物理结论。此次文档变化以链接/清单/语法和资源gate验证，无新增模拟器测试。

### 6) Evidence

- [实际环境、文件身份、依赖和目录清单](../resource_audit_20261010/inventory.json)
- [本轮四场景预检与发布资源校验](../resource_audit_20261010/resource_checks.json)
- [环境/USD 检查与版本覆盖诊断](../resource_audit_20261010/environment_checks.json)

- [Ruff/pytest设置](../../pyproject.toml)、[hooks](../../.pre-commit-config.yaml)、[项目定位](../../src/protomotions/repository.py)
- [资源异常/gate](../../src/CrowdSim/humanoid/application/resources.py)、[SDK导入](../../src/protomotions/utils/simulator_imports.py)、[父进程 receipt](../../scripts/run_trace_pace_verified.py)

## Clone portability update — 2026-10-10

The current four-scene flow uses `configs/reproduction/four_scene`, exact in-memory resource bindings, the committed SDK bootstrap/native shutdown bundle, fixed-bank rebuild templates and `scripts/run_four_scene_benchmark.py`. See [clone reproduction](../../CLONE_REPRODUCTION.en.md) and [validation](../CLONE_VALIDATION_20261010.json). Historical experiment receipts remain provenance, not new-run configuration.
