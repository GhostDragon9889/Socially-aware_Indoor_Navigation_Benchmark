# 隔离 state-action 训练 v3

[完整登记](../../output/isolated_training/mixed_state_action_v1/fullfresh2000_v3_registration.json) 已封存；**6 个 CPU 回归、Ruff、fresh2000 纯前端预检及 59 个真实 CPU 模块来源检查通过**。926 个复制代码／资产文件、原 146 冻结源和 6 个资源与 v2 不变。launcher SHA `2f2a8137599f83a259073e210bac4c283303bae03c765682aefb4885bc24d3a5`；manifest SHA `f2af12c16bb7f5a0808657e4f2589bc3c4db91370c4201086c5b4416bdafb8fa`；完整 archive SHA `ca2ee4fac7d71a4b7880a34283c23e2e15acd9b43f2376f3ad2c5bfdf3f0cb6c`。

实际 v2 pilot4／恢复至8已由 root 执行，SDK 快速退出绕过 `finally`，完整 GPU 导入报告缺失。v2 的 CPU59模块映射不能替代真实 GPU 导入映射，该历史限制未补造。v3 为每个真实 owned loader 成功执行后同步写事件日志并 `fsync`，原子更新累计来源报告；CPU 子进程 `os._exit(0)` 回归证明已观察的来源保留。运行入口单列 selected-before-runpy 事件，未伪称它是 import。若 SDK 未触发 finally，报告明确 `observed_not_finalized`／`runtime_finalization_observed:false`，真实训练 result／checkpoint 仍是执行状态依据。

## 入口及快照

```bash
bash output/isolated_training/mixed_state_action_v1/fullfresh2000_v3.sh
```

独立 CPU 进程：

```bash
bash output/isolated_training/mixed_state_action_v1/watch_snapshots_v3.sh
```

固定 fresh2000、256 actors、rollout32，共 16,384,000 transitions；两课程各 8,192,000。每 500 更新保存，watcher 封存 500／1000／2000；它仅核验基本结构、预算及双 SHA 相同字节运输，**不验证完整恢复语义、不挑选质量候选**，之后仍须独立 payload／Adam／RNG／课程状态审核。

v2 watcher 的 interpreter 被 `Path.resolve()` 解引用至 base uv Python，尚未执行；该版本及其 archive 保留。v3 watcher 使用显式 `.venv/bin/python` 路径，实际 CPU Torch 导入确认 `sys.prefix` 为 IsaacLab-Protomotions `.venv`、CUDA 未初始化。冻结业务代码不因此改变。输出／资源使用绝对路径，独立缓存保持 `code/output→runtime_scratch`。

root 已启动 fresh2000_v3；首个只读检查实际 sidecar 有 89 事件／86 owned 模块，逐项路径和复制 SHA 正确，manifest／launcher 与登记一致。该首观察不是最终训练完成或行人效果验收。训练期间不再修改本 bundle 的代码或 wrapper。
