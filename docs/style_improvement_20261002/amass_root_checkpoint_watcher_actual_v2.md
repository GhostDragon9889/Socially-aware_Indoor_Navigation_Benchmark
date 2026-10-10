# 实际 AMASS-root CPU 检查点监控 v2

实际4/8 payload门禁通过后，sole watcher session63841已就绪。

- 执行源码SHA：`db8d243aedc5f43cd0f920b3888c8e35e207929826be9f26e850c8dad7713eb8`。
- [同字节exclusive源码副本](watch_amass_root_checkpoints_actual_v2.py)。仅封存，没有启动第二实例。
- [实际4/8恢复状态门禁](amass_root_actual_pilot_checkpoint_schema.json)。
- [实际命令、源码和启动元数据](amass_root_checkpoint_watcher_actual_v2.json)。
- 独立目录：`output/diffusion_amp/amp_style_amass_root_snapshots_v1`，训练目录只读。

124份GPU预启动源码archive SHA `ed16c4cdbfd533a9fe2bfeb50c3f717cdbb0f033462dadd8b4acf62a0adaf56a` 保持不变，其中watcher是历史准备版本。实际CPU工具根据真实source_archive/151-knot linear-tail/endpoint velocity/origin API完善，单独封存为v2；GPU业务/资源/配置没有改动。

监控只复制200/500/1000的原完整权重与写own ledger，不launch/kill/resume训练，不写last/metrics/result。使用CPU weights_only/mmap和inode pin；logical payload检查不代表PhysX接触/solver缓存或GPU逐位恢复。
