# 隔离训练缓存修正 v2

**160 项 CPU guard 检查、59 个实际业务模块来源检查通过；v2 GPU 训练尚未由本报告验证。** [CPU readiness](../../output/isolated_training/mixed_state_action_v1/CPU_readiness_v2.json)、[独立 CPU 检查](isolated_state_action_source_CPU_verification_v2.json)。

v1 实际 GPU 初始化在创建相对 `output` 缓存时失败，PPO 尚未开始；[原失败 result](../../output/diffusion_amp/amp_style_mixed_state_action_isolated_pilot_v1/result.json) SHA `2853438ba2dd4cd3329f1e0b5c96ec3588b80cf7881ce7806c2590e4378a7ae4` 已保留。v1 的源、命令、日志及 CPU 准入记录均是修正前证据。

v2 仅新增 `code/output → runtime_scratch` 链接，目标是 bundle 内的独立可写缓存目录；926 个代码与资产文件、全部旧 146 源的字节不变。复制代码继续为文件 0444／目录 0555，执行 guard 仍拒绝当前仓库或缓存目录中的 Python 代码，CPU 检查已实际验证缓存可写与缓存代码执行被拒绝。它不提供对资源 symlink 的 OS 级只读隔离。

v2 manifest SHA `429fdd0c23569400e17584f80d84d993374dac2b57006d5035f7740dc628779c`；完整 archive SHA `a433396eb28da25781705d520a62252994cefccf5836482eadffc4e1642c215e`；launcher SHA `31fd40be8a59f23c19b6c07fc258ebbd8a350402f967bb49dcc46ddfe8ef648d`。[完整命令](../../output/isolated_training/mixed_state_action_v1/commands_v2.json) 全部使用绝对资源／配置／输出路径，输出独立于失败 v1。

## 实际执行入口

```bash
bash output/isolated_training/mixed_state_action_v1/pilot4_v2.sh
```

进程退出后封存真实 `step_4.pt`／result／metrics，再运行：

```bash
bash output/isolated_training/mixed_state_action_v1/resume4to8_v2.sh
```

fresh 4 更新为 32,768 transitions；恢复再 4 更新为累计 65,536。运行后仍须核验实际来源、模型／Adam／nested 三组 RNG 与课程状态。本报告不将 CPU 准入、可写缓存或训练执行完整性视为行人质量验收。
