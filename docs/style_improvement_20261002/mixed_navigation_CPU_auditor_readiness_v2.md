# 混合导航审查器：v2 CPU冻结

22项CPU测试及Ruff通过，actualstep2000参数／模型／138来源重新155项准入通过。主工具：[audit_mixed_navigation_ground.py](audit_mixed_navigation_ground.py)；完整新source身份：[v2 JSON](mixed_navigation_CPU_auditor_readiness_v2.json)。

v1的所有准入／测量算法及实际plan/reset/failure测试保持，修正仅生成Markdown的复现命令：使用本138来源wrapper及注册参数，而不是不支持该来源的旧helper CLI。移除新增formatter并回退调用后全AST与v1相等；v1文件/归档不覆盖。没有新候选capture已通过的声明，没有GPU/SDK/34读取。

CLI及完整测量范围见保留的 [v1说明](mixed_navigation_CPU_auditor_readiness_v1.md)。运行完成后仍要求完整6000控制帧／24000actor帧／96000actor子步，不替换原MMD协议或效果验收。
