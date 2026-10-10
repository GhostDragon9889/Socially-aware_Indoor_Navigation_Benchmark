# 封存的34段最终测试配对

先固定候选，再读取test；本审计仅CPU重算既有捕获，不调参、不改变检查点。

| 完整34段 /7739帧 | negative1000 | AMASS-root1000 |
| --- | --- | --- |
| tracking_successes | 4 | 11 |
| sequences_with_fall | 1 | 0 |
| frame_weighted_mean_body_position_error_m | 0.445018475 | 0.344130298 |
| frame_weighted_mean_body_rotation_error_rad | 1.52240062 | 1.53523374 |

全部34初始物理状态/AMP history以及参考目标/phase/time通过dtype+bitwise配对；原始位置错误阈值、跌倒连续时段、全文主指标已独立重算。
实际权重内1000updates/8.192M与完整parent-disjoint门禁通过；training源码tar逐member核验。baseline test的world ba10eeb4与训练70914a2不同，现有before/tar无对应bytes，因此该单文件完整来源绑定标未知；另5个baseline与6个candidate捕获source均匹配。

本结果支持这一固定候选在本地test的root跟踪位置和成功数改善，旋转均值略差；不证明style质量、200秒导航或作者64.6%协议等价。
单seed、固定本地样本，不以success-only子集替代完整分母。部署导航失败的历史结果继续保留，未提升默认模型。

预选SHA `429bbe40425d43736a1d62434af6cd7cdb0fa7ce4b0e5be3a396056e9a8861b9`；candidate SHA `dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e`。
