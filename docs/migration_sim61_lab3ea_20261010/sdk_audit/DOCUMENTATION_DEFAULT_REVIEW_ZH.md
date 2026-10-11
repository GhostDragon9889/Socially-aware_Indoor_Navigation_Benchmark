# 当前文档默认环境复核

本轮仅扫描文档并报告，不修改其他负责人的文档或运行源码。当前新默认应为 `env/isaaclab3_ea_sim61/profile.json` 和 `.runtime/isaaclab3_ea_sim61/.venv/bin/python`。复核快照与每份已阅读当前页的 SHA 见 [机器记录](documentation_default_review.json)。以下是当时内容，负责人随后更新应以其新 SHA 为准。

## 已明确的新默认

仓库 README、CLONE_REPRODUCTION、`docs/guides` 中英安装/依赖/配置/使用页均指向6.1；FULL_PIPELINE 第33行也已明确新默认，并把下面旧6.0命令归入历史记录。本轮未发现这些入口误把旧SDK当作默认。

## 仍建议由文档负责人补充的指引

| 仓库相对路径 | 需要核对的表述 |
|---|---|
| `docs/codebase/STACK.md` | 第8/42/69/98行仍把package12/Sim6.0/旧extra/目标缺包作为当前栈；适合保留为迁移前审查，并增加当前入口。 |
| `docs/codebase/TESTING.md` | 第12/15/16行使用根 `.venv/bin/python`；需明确历史解释器或改为当前 `NAV_PY` 示例。 |
| `docs/codebase/INTEGRATIONS.md` | 核心SDK集成行仍为旧checkout名称，需要区分历史与当前。 |
| `docs/PROJECT_STRUCTURE_ZH.md` | 第64行说默认解释器复用 `.venv`，与新runtime/VSCode目标不一致。 |
| `docs/RESOURCE_DEPENDENCY_MAP_20261010_ZH.md` | 第29行旧profile称当前，第45行迁移前缺模块/安装冲突需历史标记；不能拿该原始SDK检查否认新的独立overlay。 |
| `docs/GITHUB_SOURCE_ZH.md` | 克隆准备中旧profile说明需指向新默认；原发布commit/资源历史保持。 |
| `docs/SHARED_SOCIAL_EVALUATION_PROTOCOL_ZH.md` | 第3行仍以Sim6.0为目标；这是旧实验协议，应增加新迁移范围指针。 |
| `docs/SHARED_SOCIAL_AMP_CURRENT_ZH.md` / `docs/CURRENT_REPRODUCTION_ZH.md` | 顶部最新段为旧6.0实验；需链接本轮6.1验收，并保留失败记录与原指标。 |

`STRUCTURE_AND_EVALUATION_CURRENT_ZH.md` 已明确2026-10-03历史范围；只需要协调其“当前”指针。所有日期实验页的旧SDK命令/指标都是来源证据，不能批量替换或删除。本文没有修改这些页面。

这份复核不把所有候选字符串算作错误，也不声称每份历史文档完成了人工语义认证。完整候选/文档清单作为外部原始取证记录保存在 [来源索引](SOURCE_INDEX.md)。
