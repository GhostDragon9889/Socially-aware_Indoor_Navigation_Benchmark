# AMP 专家采样：实际运行

本轮沿用 `19f334…` 的训练方法与 outer AMP weight 2，只改变真实 AMASS 专家窗口的采样比例。完整训练仍使用固定 train 2,848／validation 22／test 34；新 Diffusion 权重不混入这轮 AMP 实验。具体配置、单变量源码证明和 ESS 在[冻结准备记录](../state_prior_stratified_training_preparation_v1/README.md)。该准备记录保留启动前的状态；本页记录实际运行。

## 已完成的试训

- GPU owner 的实际工具进程 `15812` 已退出 0：fresh 4 updates，32,768 physical transitions；训练过程 0 falls、125 time-limit resets。它们是训练计数，不是动作质量或导航成功率。
- [完整检查点](../../../output/diffusion_amp/amp_style_state_prior_stratified_isolated_pilot_v1/step_4.pt) 是独立只读副本，SHA `1e14e326…`；[GPU 封存记录](pilot4_seal.json) 绑定全部实际输出。
- CPU 审计的实际工具进程 `66636` 已退出 0，全部 **2,083 项检查通过**：[审计报告](../state_prior_stratified_training_audit_v1/actual_pilot4_v1/audit.json) SHA `162bc676…`，[实际退出记录](pilot4_CPU_actual_launch.json) SHA `697843be…`。
- 实际检查模型、两个 Adam、回放、归一化、随机数和恢复状态，以及重建真实动作库后的 sampler registry／probability hash、private CPU RNG、**16,384 expert windows／4 batches**。专家窗口数与 physical transitions 是不同计数。

## 完整训练已完成，质量评价进行中

独立 fresh 2,000 updates、16,384,000 transitions 已完成，GPU工具进程 `99866` 与 CPU watcher工具进程 `71836` 均真实退出0。[实际训练owner](fullfresh2000_actual_launch.json)、[快照监测owner](watch_actual_launch.json)。训练主体3,851.15秒，225次探索跌倒重置、107,156次time-limit重置，均不是部署失败率。完整训练没有从试训恢复，新Diffusion权重也未混入。

最终独立只读快照SHA `98a49058b4bbf7ffc83ba88c78dbd03e5be16c3bf7ece320356be9ec6f9f7bef`，653,271,383字节，与训练器`last.pt`完整字节一致。[终态封存](terminal2000_root_seal.json) SHA `b1592bc3…`。首次封存脚本误猜原始文件名，导致CPU审计的只读门槛失败；[失败记录](terminal2000_CPU_first_failure.json)保留，未改审计规则或重训。

修正封存后，CPU工具进程 `99226` 真实退出0，**2,072项完整检查通过**：[终态报告](../state_prior_stratified_training_audit_v1/actual_terminal2000_v1/audit.json) SHA `b9fa0221…`，[实际关闭记录](terminal2000_CPU_actual_launch.json) SHA `603ae1dc…`。包含真实模型、两个Adam、回放、RMS/RNG/terrain和恢复接口，重建真实专家库并检查私有采样状态；实际采样8,192,000 expert windows，与16,384,000 physical transitions分开记录。

全22动作GPU工具进程 `21634` 真实退出0，20／22成功、0跌倒、全身位置误差0.214709m／旋转误差0.782873rad。失败动作20、257全部841帧保留。固定90D MMD²为0.04156963，比19f的0.04209234略低；慢速与较快速度桶仍有退化：[完整tracking CPU报告](../state_prior_stratified_CPU_metrics_actual_runs_v1/tracking_CPU_report_v1.md)。

四人200秒导航GPU工具进程 `34910` 真实退出0，完整6,000步、0跌倒／0接触／0重置，路线顶点[8,8,8,8]合32（19f为40、默认为21）。三个原CPU指标均退出0，raw203项检查通过。90D支持4桶，共同三桶MMD²为0.06751219（19f0.06811695、默认0.04098256）；45D MMD²0.03460670，K50／K100覆盖0.58／0.54。所有停滞与unknown桶保留：[完整navigation CPU报告](../state_prior_stratified_CPU_metrics_actual_runs_v1/navigation_CPU_report_v1.md)。物理准入满足，native结果尚未产生；没有替换默认权重。

上述完整评价显示局部改善与取舍，尚不能认定整体风格优于默认。[冻结评价计划](../state_prior_stratified_evaluation_v1/README.md)与[五项CPU指标计划](../state_prior_stratified_CPU_metrics_preparation_v1/README.md)已通过独立源码审查，尚无未来评价结果。只有完整导航零跌倒、零静态接触、零行人接触，才继续 native4／15；风格分数不筛选物理评价。没有替换默认权重，也没有证明达到或超越论文效果。

审计恢复验证不保证 PhysX 的逐位确定性；实际运行仍保留 `runtime_finalization_observed=false`。封存34条测试输出未用于优化。
