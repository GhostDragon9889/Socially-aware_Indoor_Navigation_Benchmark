# 物理步后 AMP：原始物理审计入口

本目录补齐未来post-step AMP四人200秒评价的可执行CPU审计。新AMP训练、真实终态、预检、运行凭据及捕获尚未生成；当前模板中的未知文件大小和SHA为null，模板状态明确为pending，不能接纳权重或物理结果。

## 复用的计算

入口读取固定的历史q936审计源码，仅在内存替换7处schema名称及1处936→937源码计数。逆向恢复整份原源码字节已检查。其terminal/source/capture/audit函数全部复用，原f851的`Audit`和`audit_capture`原始物理计算保持原函数；原始来源回调在读完后恢复。

实际运行前额外检查新来源：937份e57源码、a06配置、选定55cc终态审计入口、真实fresh2000闭合训练凭据、root封存记录、资源预检、原185生产源码及旧Diffusion63。四人／6000控制步、全部跌倒、接触、idle和reset仍进入原分母；原90D／45D风格评价另行运行。数据一致性通过与物理零失败是独立字段。

此处是固定q、状态判别器、物理步后时序的机制对照。Nav论文式2写状态历史与动作，当前状态判别器消融的结果不提供严格作者实现等价性。

## 待真实产物发布后

1. 先完成已登记的新Diffusion40k终态和固定AMP对照，再串行执行post-step AMP的fresh4完整审计、fresh2000训练与终态审计。
2. 实际训练和watcher退出后，封存完整终态及root记录。新的root seal schema为`state_only_q.poststep_actual_terminal_root_seal.v1`，保留原实际GPU／watch工具会话、启动／完成块和独立checkpoint事实。
3. 按[固定评价计划](../../style_improvement_20261003/state_only_q_poststep_evaluation_source_plan_v1/README.md)实际完成资源预检。receipt记录真实root工具块`actual_root_tool_chunk`、退出0、原CPU环境和命令；`preparation_plan_sha256`绑定静态6a09评价计划，避免与包含receipt的后续binding循环依赖。
4. 以[模板](binding_pending_template.json)发布新只读`actual/commands.json`，补充真实文件记录，将状态改为`actual_terminal_and_preflight_registered`。只有实际产物存在且已核对时才可填写。
5. 执行固定四人200秒命令，保存真实root会话／启动块／完成块、closed0、实际GPU环境、185源码前后记录以及整个只读捕获目录。owner schema为`state_only_q.poststep_actual_fixture_root_owner.v1`；保留`MMD_launch_filter=false`、`default_promoted=false`和`sealed34_outputs_read=false`。
6. 实际退出并封存后运行：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/state_only_q_poststep_fixture_audit_v1/audit_poststep_fixture.py \
  --binding /absolute/path/to/actual/commands.json --binding-sha256 ACTUAL_SHA \
  --owner /absolute/path/to/fixture200s_actual_root_owner.json --owner-sha256 ACTUAL_OWNER_SHA \
  --output /absolute/path/to/registered/actual_raw/audit.json
```

注册的输出路径须为新文件。失败报告保留，不覆盖重跑；当前不运行此命令。SHA占位符不是资源身份。

## 本轮验证范围

原源码逆字节与AST／compile检查通过；新入口Ruff通过。命名为`commands.json`的只读pending模板在接触未来文件前被拒绝。未执行派生的metadata模块、未加载模型、未读新捕获、未启动SDK或GPU。未来原始物理审计仍待真实产物，当前结果不证明动作质量或论文效果。
