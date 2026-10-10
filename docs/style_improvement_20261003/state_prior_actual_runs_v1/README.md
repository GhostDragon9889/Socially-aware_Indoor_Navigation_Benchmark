# 公共 AMP 先验：真实训练运行记录

## 当前阶段

正式 fresh2000 已实际完成256环境×32步×2000轮，即16,384,000次物理转换；GPU与watcher均退出0，三个阶段snapshot封存齐全。第500与1000轮已封存并设为只读，各通过2,057项独立完整状态检查。独立终态2,062项通过，完整22跟踪17/22／零跌倒；四人200秒零跌倒／接触、40路线点、0.1333停滞人秒。90D最低门通过，45D姿態覆盖退化；原生Warehouse四人／15人200秒均已结束，零跌倒／接触，44／160次到达、8／32循环。同场景风格距离改善但路线进展下降，15人K50覆盖亦下降。30秒室内姿态回放与媒体审计完成，默认部署模型保持原step1000。[完整实际比较](../state_prior_complete_review_v1/README.md)。计数以实际[训练指标](../../../output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_v1/metrics.jsonl)为准。

## 实际通过的门禁

- [合并后44新增＋224回归与生产预检](../merged_action_state_prior_CPU_v1/README.md)；[183文件来源V11](../native_runtime_epoch_v11/native_pedestrian_source_v11.json)。
- [隔离934文件实际冷初始化与资源准入](../state_prior_training_preparation_v1/actual_build_CPU_readiness.json)。
- [实际pilot4：2068项](../state_prior_training_audit_v1/actual_pilot4_final_v2/audit.json)，32,768转换。
- [真实4→8恢复：2120项](../state_prior_training_audit_v1/actual_pilot8_v1/audit.json)，65,536累计转换。
- [第500轮：2057项](../state_prior_training_audit_v1/actual_milestone500_v2/audit.json) · [第1000轮：2057项](../state_prior_training_audit_v1/actual_milestone1000_v1/audit.json)。
- [发行包与解包资源验证](../state_prior_packaging_review_v1/packaging_review.json) · [评估协议独立审查](../state_prior_evaluation_peer_review_v1/peer_review.json)。

本轮将公共state-only判别器的1950历史列扩展到2019输入，并复制其AMP历史统计。69动作列初始化为零，57有效列随后正常学习，完整恢复保留已学习权重；12冻结动作坐标保持零。这是D与历史归一化统计的耦合初始化实验，默认新开关为false。公共预训练数据与本地AMASS的交集未知；不冒充作者定制权重或作者2,848动作IDs。

## 已实际完成的评估

[冻结协议](../state_prior_evaluation_registration_v1/registration.json)与[实际GPU命令](../state_prior_evaluation_registration_v1/commands.json)：完整22条／11,982计量帧、四人6,000控制帧／200秒，固定90D风格计算和原b25dffa…45D参考基。保留失败、停滞、原速度分桶和未支持桶；不重新拟合参考来提高分数。必须正式进程实际退出、2000轮权重封存并通过独立终态审计之后执行。

[上游资源更新核查](../public_resource_refresh_v1/report.md)列明当前公开入口及复现协议的具体缺项。

## 保留的失败证据

第一次生产预检误用CLI参数（exit2），后改用真实preflight模块（exit0）；独立审计脚本一次调用错误属性的框架失败保留，随后冻结修正reader实际通过。第500轮首次恢复门禁因快照OS可写而拒绝；root核对SHA后仅chmod0444，原字节不变，原拒绝报告保留。以上不被替代为效果通过。
