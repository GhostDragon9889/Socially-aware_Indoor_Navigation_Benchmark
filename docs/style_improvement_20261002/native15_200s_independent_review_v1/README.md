# Warehouse 15 人 200s 独立 CPU reader 登记

当前仅完成 **准备门禁**，真实 200s 终态产物尚未审计。`registration.json` 明确 `native_execution_audited=false`、actual counts/quality 为 null，未读取进行中的 result 或 mutable capture。

SDK-free 实际 deployment entry、父登记和 source174 `f9c007ec…` 冻结归档/174 成员均已绑定。7 项 CPU 回归与 Ruff 通过。旧四人 v1/v2/route reader 按原始 bytes 复制；执行时仅把 v2 的四个重置点数量、slice、role 常量由 4 改为 15、总射线由 132 改为 143，反向恢复后**整个 AST 与原版完全相同**。力、COM、4 子步、4 足、接触、跌倒和路线公式未改。

真实 source174 的 `dynbench_live.py` 查询所有重置起点，`amp_world.py` 额外请求 128 个自由点。因此本轮是 **15+128=143** 条实际 pre-reset 射线，不能照抄旧四人的 132。

终态完整数量门禁：6,000 控制步、90,000 actor controls、24,000 shared substeps、360,000 actor substeps、1,440,000 足观测；static 分母 90,000，unordered peer pair 分母 630,000。完整保留所有 fall/reset/contact 帧，并核真 COM、正质量、745 个 static collider、唯一声明地面、全部 source files 和实时 query counter 记录。

根通知实际进程结束后执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
docs/style_improvement_20261002/native15_200s_independent_review_v1/audit_native15.py \
--registration docs/style_improvement_20261002/native15_200s_independent_review_v1/registration.json \
--output docs/style_improvement_20261002/native15_200s_independent_review_v1/actual_audit.json
```

原 source174 归档是历史身份；合法后续源码变化会单列 current observation，不会用新树替换已运行代码的绑定。完整源快照不构成全部运行 import ledger。任务执行、测量完整性与导航效果验收保持分开。
