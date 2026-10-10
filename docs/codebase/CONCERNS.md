# Codebase Concerns
审查：2026-10-10；基于本机文件、安装元数据、AST 和 SHA。此次未启动 Kit、训练或物理重测。完整机器清单见 [inventory.json](../resource_audit_20261010/inventory.json)。

## Core Sections (Required)

### 1) Top Risks (Prioritized)

| 级别 | 已核实事项 | 处理方向 |
| --- | --- | --- |
| P1迁移 | source-bound地图/SDK/terminal helper，SMPL链接指外部arena | 按引用迁移并重新生成/准入地图；独立验证生命周期 |
| P1新SDK | target Sim6.1/Lab17.0.2；7core缺包+2越界、scene队列接口差异已有复现 | 独立profile和适配；不能直接改.venv链接 |
| P2环境维护 | SDK HEAD5951d48与profile4ecd不同，apply实际拒绝 | 核对完整SDK source来源后协调记录/门禁；现有packagecheck不代替此检查 |
| P2依赖声明 | root缺少TRACE/SMPL部分明确功能依赖，crowd GUI包与SDKheadless不同 | 新环境从SDK完整manifest安装；将功能extra/环境约束作为后续代码维护项 |
| P2语料扩展 | shapely/rvo2、JAX/pyroki等可选生成依赖缺失 | 按目标工具建立独立环境；不混入现有SDK |
| P2远程恢复 | 自有LFS与上游双端点，第三方checkout未随根仓库clone | 用release helper；按固定revision恢复参考仓库 |
| P2外部发布 | HF旧记录待身份，读插件not found | [TODO] 获得可访问模型仓库的远程revision/文件核验后才能列为恢复来源 |
| 科研缺口 | 作者私有权重/精确2848 ID/八场景协议缺失 | 保留独立重建声明；本地成果按自己的协议验收 |

### 2) Technical Debt

运行中的terminal helper仍在docs封存目录；src/runtime通过动态编译加载，包分发和清理容易漏。scripts配置与兼容链接使完整checkout成为运行合同。data内多个AMASS manifest版本和不同模型族并存，应保留选定v7和profile绑定，候选不可覆盖历史成功证据。

### 3) Security Concerns

AMASS/SMPL及派生身体资产许可不同于源码；原始动作/人体参数不随私有release或wheel发布。manifest的完整路径记录用于本机可追溯性，可能透露目录结构，后续外发需明确范围。权重SHA只证明身份；pickle读入仍须用对应受限/显式loader，不应反序列化不可信资源。此次未读取凭据、未加载Torch pickle。[SMPL受限loader](../../src/CrowdSim/tools/smpl_mesh_visualizer.py)

### 4) Performance and Scaling Concerns

output逻辑170.11GB；外部AMASS树81.64GB；data17.11GB；release2.69GB。统计不跟随符号链接、排除.git/.venv/cache等，不是物理块占用也不能简单相加推算去重磁盘量。历史rollout/checkpoint/源码快照占较多空间；当前map/物理源/恢复状态不能盲删。多原生实例争用16GB显存，按场景串行运行。扩散采样、SMPL skin和录像也是开销项，此次未profile吞吐。

### 5) Fragile/High-Churn Areas

根依赖、README、sdk simulator/isaaclab、场景复制、原生清理和训练入口是Git历史高改动或SDK相关边界；需优先回归。本机4/4/1/1人60秒有限成功不足以保证高密度、复杂地面、多种子、全场景或EVT闭环导航。足部原点/FK/皮肤诊断不等于测得接触斑滑移/摩擦。

### 6) `[ASK USER]` Questions

此次资源与依赖梳理无需用户额外决策。环境升级、清理数据、再次上传均未实施；不存在为完成本次只读审查必须回答的意图问题。

### 7) Evidence

- [实际环境、文件身份、依赖和目录清单](../resource_audit_20261010/inventory.json)
- [本轮四场景预检与发布资源校验](../resource_audit_20261010/resource_checks.json)
- [环境/USD 检查与版本覆盖诊断](../resource_audit_20261010/environment_checks.json)

- [绑定生命周期](../../src/CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py)、[场景源合同](../../src/CrowdSim/humanoid/runtime/dynbench_scene.py)、[SDK apply门禁](../../scripts/isaaclab3_environment.py)
- [发布catalog](../../resources/releases/pedestrians_20261010/catalog.json)、[发布排除范围](../../resources/releases/pedestrians_20261010/manifest.json)、[原生实测限制](../FOUR_SCENE_SPEED_20261010_ZH.md)
- [兼容性审查](</home/lfm/Documents/ChatGPT/Social Navigation/isaaclab_compatibility_audit_20261010/ISAACLAB_COMPATIBILITY_REVIEW_ZH.md>)、[Git扫描与高改动文件](.codebase-scan.txt)

## Clone portability update — 2026-10-10

The current four-scene flow uses `configs/reproduction/four_scene`, exact in-memory resource bindings, the committed SDK bootstrap/native shutdown bundle, fixed-bank rebuild templates and `scripts/run_four_scene_benchmark.py`. See [clone reproduction](../../CLONE_REPRODUCTION.en.md) and [validation](../CLONE_VALIDATION_20261010.json). Historical experiment receipts remain provenance, not new-run configuration.
