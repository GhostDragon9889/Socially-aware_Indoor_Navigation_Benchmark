# AMASS-root 检查点守候计划

当前状态：**代码准备完成，尚未启动，等待实际冻结源码与真实 pilot 门禁**。仅 AST/Ruff 已通过；不能据此宣称实际 checkpoint schema 验证通过。

新脚本：[watch_amass_root_checkpoints.py](watch_amass_root_checkpoints.py)。旧 watcher/ledger/权重保持原样。

接口：`--training-output TRAIN --snapshot-dir FRESH --source-manifest SOURCE_JSON --contract-checkpoint ACTUAL_STEP4_OR8 --max-seconds 2700`。

- 在第200更新前启动，最多守候45分钟，精确保存200/500/1000；丢失时只记录 missing，绝不重标或重构。
- snapshot目录必须独立于训练目录且不存在；exclusive mkdir，权重发布使用exclusive hardlink。
- last.pt/result只读；weights_only/mmap CPU读取，inode/size/mtime pin；不launch/kill/resume训练，不写last/metrics/result。
- source archive全部SHA校验，CPU导入冻结实现；实际pilot推导actor数量、rollout、config/资源/runner契约，不假定未来键。
- 实际AMASS class严格恢复motion IDs、初始phase、world offset、clip remaining、episode、observed/plan ages、statistics与共享RNG；active计划与真实bank复核。
- 默认expert_sampling=None允许；若启用，完整私有采样器必须严格恢复。
- 保留完整model/optimizer/normalizer/replay/global RNG/physical payload；不据此声称PhysX solver/contact缓存或GPU物理逐位恢复。

收到真实pilot后先执行CPU schema gate，再开sole watcher。此计划不是训练效果或论文复现验收。
