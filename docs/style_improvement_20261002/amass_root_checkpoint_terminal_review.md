# AMASS root 课程检查点终态验收

CPU only；sole watcher 已退出 0；ledger `passed`，缺失检查点 0。训练目录全程只读，固定快照写在独立目录。

|更新|真实 transition|字节|SHA256|
|---:|---:|---:|---|
|200|1,638,400|652092533|`5a2f60ae79c9675b628ef8221f6fb5c9c384841c17e89e196363c4b95b4deb17`|
|500|4,096,000|652092533|`8db53ffad6b60782a7e73dd924de675f0caaa941d951efb6806fad22cd092314`|
|1000|8,192,000|652092533|`dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e`|

最终独立重新计算三文件 SHA256；200/500 与首次封存哈希一致。完整 ledger：[snapshots.json](../../output/diffusion_amp/amp_style_amass_root_snapshots_v1/snapshots.json)。详细验收：[JSON](amass_root_checkpoint_terminal_review.json)。

## 终态与计数

`result.json`：passed，1,000 更新 / 8,192,000 transition / 32,000 控制步；elapsed_s=1688.6771960980113。训练实际 fall157、time-limit79923、reset80080；curriculum reset80336 含初始256个 actor reset。training minimum pelvis=0.3564301133155823 m。这些训练统计不是 heldout 或论文达标证据。

## 范围

每份保存的是实际当时完整 payload：模型、优化器、normalizers、replay、runner、AMASS frame/offset/episode/tail、两个 CPU RNG；实际 bank 与151个轨迹 knot 的契约检查通过。expert_sampling=None，未额外产生 sampler 状态。

124文件 GPU 业务源码 archive 保持不变；archive 内 watcher 是旧准备版，实际执行的 CPU 工具另存 [actual_v2](amass_root_checkpoint_watcher_actual_v2.md)。未启动第二 watcher。

未声明 PhysX solver/contact cache 或逐位物理恢复，未逐浮点扫描全 replay，也未重算1GB bank SHA。author2848原始列表仍未知；当前资源是2,848个有来源的 AMASS 分段，尾部线性 XY 是声明的工程命令假设。新 full22/style 评估尚未完成，暂不作效果提升结论。
