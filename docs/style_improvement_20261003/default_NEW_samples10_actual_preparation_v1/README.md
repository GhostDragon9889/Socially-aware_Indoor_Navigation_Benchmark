# 十候选实际进程与原 raw CPU reader

仅登记 NEW b563＋默认 AMP62c4 的四人200秒 fixture，方法变量仍只有候选1→10。生产185、AMP、DP、场景、历史、物理、PD、reference adapter、goal20、15步重规划、seed7不变。源准备主记录：[原prepared packet](../default_NEW_samples10_preparation_v1/source_freeze_v1.json)，[独立source peer](../default_NEW_samples10_source_peer_v1/source_peer.json)。

[registration.json](registration.json)只保存父记录、已执行preflight和未来路径。helper运行时从已冻结的原注册构建虚拟plan：原NEW case替换samples10的command／preflight／output／owner／audit身份；其余preflight逐字段完全相等。未来GPU receipt、CPU receipt、audit均不存在，没有未来capture准入或质量结论。

## root 的确切命令

GPU需在当前q训练和评价完成后由root串行执行：

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/default_NEW_samples10_actual_preparation_v1/launch_registered.py --phase fixture_new
```

root观察GPU child真实closed0并封存输出后，再执行原raw数学的CPU审计：

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/default_NEW_samples10_actual_preparation_v1/launch_registered.py --phase fixture_new --cpu-audit
```

两条完整数组也在[commands.json](commands.json)。追加`--prepare-only`仅验证来源并打印待运行命令；不启动GPU/SDK/模型，不读取未来captured数组。

## 保留的逻辑

- Owner加载已审`bc617…`的原`main`，仅改phase choices、schema及删除历史OLD重跑排序段；本轮父NEW已真实完整闭合，父GPU／CPU记录严格绑定。原Popen、wait、进程组TERM→KILL、异常、cleanup_error、真实returncode及finally正文不变。owner失败、非零退出和日志保留。
- source binding复用父原185／原archive／数据／opaque模型SHA／已完成DP935训练来源检查，再增加外部entry、参数、本注册与冻结源码身份。执行前后必须完全相等。默认AMP原0664不变。新owner不是旧bc617实际执行身份。
- Admission复用父完整guard；只换新closure schema／owner路径与虚拟plan，历史DP与部署DP采用已修v2的真实producer ABI。整份历史AMP62c4训练provenance保持OLD63e；实际planner仍由arguments＋evaluation_planner_change＋opaque pre/post SHA绑定NEWb563。资源header/stat单独不能证明SHA。
- v2 runtime function仅将`diffusion_samples_per_actor == 1`改为10；100步DDPM／FP32／未compiled、actor-only、alpha0、seed7初始化边界不变。producer `base_checkpoint`仍只要求path/SHA，bytes由独立artifact record验证。
- CPU adapter只给原`0c08…` raw入口替换admit callback，继续调用同一`f851…` raw数学；COM、contact、falls、phase、denominator没有新公式。[metadata证明](metadata_exact_parent_CPU_proof.json)记录函数精确逆替换及原owner全部5个try/except/finally AST子树相同。

报告完整6000控制／200秒／4人及全部异常、失败、idle和reset。原零跌倒／零static／零peer gate保持；source/integrity passed不等于physical gate passed，MMD不滤除失败。actual child closure、模型效果、显存峰值与native准入都要等待实际运行；不自动推广默认。
