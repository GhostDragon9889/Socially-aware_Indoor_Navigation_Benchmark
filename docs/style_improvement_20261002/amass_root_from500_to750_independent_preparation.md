# AMASS root sealed500→750：独立续训审计准备

## 已验证起点

CPU只读152项门禁通过。起点是同一次fresh训练封存step500，SHA `8db53ffad6b60782a7e73dd924de675f0caaa941d951efb6806fad22cd092314`，累计500更新、4,096,000 transitions、16,000 controls。完整`weights_only=True` payload及严格CPU trainer loader接受，133个已序列化Tensor全部finite；原始124文件prelaunch archive逐文件hash及安全路径检查通过，审计实际导入的business源码保持该冻结身份。

| 起点序列化计数 | sealed step500 |
|---|---:|
| self/task/value RMS count | 4,296,000 |
| AMP history-coordinate RMS count | 6,344,000 |
| actor/critic/task encoder Adam step | 4,708 |
| discriminator Adam step | 1,000 |
| policy replay retained/accepted/cursor | 65,536 / 105,926 / 40,390 |
| cumulative curriculum actor steps | 4,096,000 |
| cumulative curriculum resets | 40,243 |
| cumulative fall terminations | 84 |
| cumulative real-clip end / cap-only / overlap | 39,879 / 24 / 2 |

这些起点Adam/RMS计数与原fresh训练前500行日志独立对应。COMMON物理状态、69D native child-joint速度、same-episode AMP history、native151计划、固定translation、年龄/phase及新reset必须重新规划的契约有效；课程与corpus保存的私有CPU RNG逐位相同。CPU使用私有validator接受Python/NumPy/Torch RNG字节，不恢复CUDA或启动PhysX。

## 待封存终点的独立验收

preflight明确恢复上述sealed500，新增250更新/2,048,000 transitions，预计累计750更新/6,144,000 transitions/24,000 controls。与原fresh命令相异的CLI字段仅`resume`、`updates`、`output`和`checkpoint_every`。损失、超参数、数据、物理契约及source124没有改变。

终点检查须基于起点计数加本次250行日志；本次result的falls/resets/time limits为续训段计数，课程state保存的是累计计数。独立脚本将核对二者差值，不额外加入初始化256个reset。默认归一化的预期终点self/task/value count为6,344,000、AMP count为9,416,000，Adam增量以真实logged minibatches为准。新report必须显式标注500→750 continuation，不能当成原未中断fresh训练在750处的轨迹。

- [准备机器报告](amass_root_from500_to750_independent_preparation.json)、[日志](amass_root_from500_to750_independent_preparation.log)。
- [专用CPU审计脚本](audit_amass_root_continuation.py)，Ruff通过；保留原fresh1000审计脚本及其报告不变。

此准备PASS不表示终点已验收。终点需另存固定`step_750.pt`并取得hash通知后执行final模式。PyTorch物理/课程状态可恢复不证明PhysX接触/求解器缓存逐位重放；新回合与原fresh750不能假设物理轨迹相同。封存34条test输出未读取，没有重复官方风格/MMD或误差分解计算。
