# Shared SocialAMP：独立原生退出修复入口 v4

入口接入实际退出0的八人 SDK v18；每次检查封存的进程记录、入口身份和当前插件／扩展清单。App 启动前校验完整扩展副本，启动后确认实际 development path、扩展、Python／绑定和原生库映射。安装 SDK 不变。

与 v3 相比仅迁移退出 helper、加载验证和来源记录。共享 World、4+4资产释放、接触弱引用、PPO／判别器、奖励、固定train2848、uniform expert 与原显式profile保持。完整源码逆还原到 v3、v2，原函数字节／AST检查及Ruff通过。新的入口是否正常退出、训练是否成功，必须由实际进程单独确认。

运行固定数据探针：

```bash
PYTHONDONTWRITEBYTECODE=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261004/shared_social_native_overlay_entry_revision_v4/probe_shared_world.py --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json --control-steps 8 --fall-probe --output output/diffusion_amp/shared_social_world_probe_overlay_v4
```

探针使用零动作和合成next-value，验证原始帧归档、整组reset及终止时序，不表示策略质量。训练入口要求 `--config`，八人32-transition短检查显式使用旧封存 `shared_social_training_entry_staging_v1/smoke.json`；fresh pilot 使用同目录 `training.json`。无物理resume或旧策略迁移。
