# 联合 CPU 回归 v2（只修改测试）

原失败日志保留：[487 PASS / 3 FAIL](integrated_training_measurement_gate.log)。两项 isolated AMP 奖励时序测试缺少实际默认 `ground_capture=None` 上下文；整入口 AST 比较被独立新增 expert opt-in setup/reporting 接线改变。

修复保留真实源码执行及历史 baseline：

- 两种 discriminator 的原始/末步 history、actor → physics → reward 顺序断言全部保留，只补关闭采集的真实上下文。
- disabled-foot 等价检查明确缩至完整物理控制循环（整个 `for step` AST），严格仅剥除精确足力开关分支；与原 before-ground 源码比较，没有把现源码复制成 expected baseline。
- 原 CLI/contact construction、prepare → world.step → publish → increment/reset、四子步 adapter、peer 单次采样及原四字段投影检查保留。

验证：

- [原三项 targeted v2](integrated_gate_targeted_v2.log)：**3 PASS**。
- [相同 17 文件联合 v2](integrated_training_measurement_gate_v2.log)：**490 PASS，0 FAIL / 0 SKIP，6.36s**。
- 两个修改测试文件 Ruff PASS。只运行 CPU；CUDA 隐藏，没有调用 Kit。

新旧测试 SHA、before 副本、原失败/新结果日志与 XML 完整指纹：[review JSON](integrated_training_measurement_gate_review_v2.json)。业务、evaluator、训练 checkpoint、原 GPU/采集结果均未由本修复更改。
