# AMASS root sealed500→750：实际续训独立验收

## 结论

**178项CPU只读检查全部通过。** 真实GPU续训已退出成功，封存终点为 [step_750.pt](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/amp_style_amass_root_from500_to750_v1/step_750.pt)，652,093,749 bytes，SHA `1e87b82b643a1d94bf2117de5a57a27ebb1e0e9811a4cabf66ab1588fcaf9592`。独立完整`weights_only=True`加载、严格CPU trainer admission及全部133个payload Tensor有限值检查通过；输入文件在审计前后哈希一致。

本轮是从sealed500 SHA `8db53ffad6b60782a7e73dd924de675f0caaa941d951efb6806fad22cd092314` 开始的**新实际500→750续训**。新增250更新/2,048,000 transitions/8,000 controls；终点累计750更新/6,144,000 transitions/24,000 controls。它不是原fresh1000训练在750时的未中断轨迹或补找快照，PhysX接触和求解器缓存没有序列化，不证明物理逐位重放。

## 增量状态与预算

| 实际保存计数 | 起点sealed500 | 新续训step750 | 增量 |
|---|---:|---:|---:|
| self/task/value RMS count | 4,296,000 | 6,344,000 | 2,048,000 |
| AMP history-coordinate RMS count | 6,344,000 | 9,416,000 | 3,072,000 |
| actor/critic/task encoder Adam step | 4,708 | 7,236 | 2,528 |
| discriminator Adam step | 1,000 | 1,500 | 500 |
| replay accepted | 105,926 | 126,359 | 20,433 |
| curriculum committed actor steps | 4,096,000 | 6,144,000 | 2,048,000 |
| curriculum resets | 40,243 | 60,260 | 20,017 |
| falls | 84 | 130 | 46 |
| real-clip-end truncations | 39,879 | 59,839 | 19,960 |
| cap-only truncations | 24 | 35 | 11 |
| real-clip/cap overlap（已包括在clip end） | 2 | 2 | 0 |

真实250行日志的update501…750及累计transition逐行连续。Adam增量与实际logged minibatches之和精确相等；学习率仍actor2e−5、D1e−5，全部Adam参数状态有效。AMP normalizer是1950维history-coordinate统计，每更新加入8192个live和4096个expert样本，没有误乘history10。

本次result的46 falls及19,971 time-limit resets是续训段计数，正好等于课程累计差值：19,960 real-clip ends + 11 cap-only。共20,017 resets没有再次加入初始化256。elapsed418.334751s；训练跌倒发生率为2.2461/100,000 transitions，只是此训练分母的统计，不是tracking成功率或风格阈值。

## 配置、来源与恢复契约

- `TrainingConfig`、全部trainer provenance（data/physics）、runner contract和课程config与sealed500相同。CLI与原fresh训练仅`resume/updates/output/checkpoint_every`四字段不同；真实终点arguments与冻结续训preflight完全相等。
- 原source124 archive安全路径、精确path集合、bytes/SHA逐项通过；实际CPU loader及训练business源码匹配prelaunch源码。新增审计文档不伪装为旧GPU源码。
- 固定AMASS v7 manifest/bank、child_joint声明与来源身份匹配。全部256个已保存actor的native151 phase、固定世界translation、合法source frame、episode、计划重放及age0必须expired的reset边界检查通过。
- COMMON物理状态6字段及69D DOF形状有效，root XYZW unit quaternion有效。AMP10帧history与每actor同episode，课程共享corpus CPU RNG逐位相同。
- replay保持65,536条，cursor从40,390至60,823，与累计accepted模capacity相等；全部raw history/actions finite，12个冻结action坐标恰好为零。
- Python/NumPy/Torch RNG可由私有CPU validator接受，CUDA保存16字节uint8状态；CPU审计没有恢复CUDA RNG或启动SDK。23个model state Tensor中22个随训练改变，architecture及契约保持。

末状态task height1024 std中位数0.275616，value std12.9262；此处只记录finite/恢复状态，没有把归一化分布当成已收敛证据。

## 文件

- [完整独立机器报告](amass_root_from500_to750_independent_terminal_audit.json)、[执行日志](amass_root_from500_to750_independent_terminal_audit.log)。
- [delivery身份独立绑定](amass_root_from500_to750_delivery_independent_binding.json)：官方sealed delivery的checkpoint/source SHA、bytes及全部增量/累计预算、segment计数与独立报告精确相等；审计没有加载mutable `last.pt`。
- [专用CPU脚本](audit_amass_root_continuation.py)、[准备检查](amass_root_from500_to750_independent_preparation.md)。原fresh1000 gate及报告未修改。

此项验收只确认实际训练交付与恢复契约。完整22条validation tracking由新的固定750执行单独评估，风格/MMD仍使用资源代理唯一冻结协议；本报告没有重复计算。封存34条test输出尚未读取。
