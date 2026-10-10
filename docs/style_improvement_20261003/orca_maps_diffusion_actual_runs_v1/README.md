# 1,000场景 Diffusion：真实训练已完成

沿用原模型结构、AMASS初始化、归一化和学习规则，fresh训练5,000步。真实GPU子进程退出0，训练主体269.69秒。全部5,000行训练指标和初始／最终诊断保留。训练阶段固定验证损失0.999245→0.047883，不能替代新旧部署权重的配对评价。

- [事前参数登记](../orca_maps_diffusion_experiment_v1/registration.json)、[真实训练关闭记录](training_actual_launch.json)、[终态封存](terminal5000_root_seal.json)。
- 最终权重SHA256：`b56346fbb0c23dc9656a28fa054b45178ef5922a0e52c66b18531b94e296da90`，279,052,920字节。
- 原trainer的`step_5000.ckpt`与`last.ckpt`是同一inode的两个硬链接，均封存0444；另存`sealed_terminal/step_5000.ckpt`，不同inode、nlink1、相同完整SHA。未修改原初始化或旧部署权重。
- [真实CPU元数据检查](terminal_CPU_metadata.json)：初始1,000步／旧5,000步／新5,000步，模型结构和state_mean/std完全相同；新模型及Adam矩有限，每个参数的Adam计数5,000；真实数据manifest和首步map／neighbor梯度匹配。该检查不证明物理质量或完整恢复。

## 同一批新验证场景的实际配对评价

事前选择100个新validation场景，每场景一个合法窗口。新旧模型使用同样的输入、批次、损失timestep/noise和采样seed，保留全部碰撞与失败预测；两个模型均在CPU执行原100步FP32 sampler。当前指导条件为无waypoints的原始离线采样。

| 原始离线指标 | 原63e7… | 新b563… |
| --- | ---: | ---: |
| 固定配对验证loss | 0.058683 | 0.053336 |
| ADE m，越低越好 | 1.132818 | 0.852463 |
| FDE m，越低越好 | 2.636335 | 1.879224 |
| 邻人碰撞point fraction | 0.0354 | 0.0236 |
| 障碍碰撞point fraction | 0.1548 | 0.1352 |
| 邻人碰撞trajectory fraction | 0.23 | 0.17 |
| 障碍碰撞trajectory fraction | 0.45 | 0.38 |
| 最小邻人表面间距 m | −0.784281 | −0.732448 |
| 最小障碍表面间距 m | −2.628902 | −2.720159 |

ADE均值下降约24.75%、FDE下降约28.72%。最严重的障碍穿透没有改善。碰撞是在ego开环预测和同帧记录的全部专家邻人／真实polygon之间计算，半径0.4m、容差1e−4m；不是多行人闭环，也不是IsaacSim真实接触率。不会据此直接替换默认Diffusion或宣称超过论文。

逐数组独立核对表明：100场景中ADE为65改善／35退化，FDE为73改善／27退化。记录的loss noise、timestep和窗口索引逐字节一致；全部共用真值和上下文由原corpus重新构建后相等。DDPM初始／反向噪声未单独保存，这项核对不包含其逐字节比较。

![全100场景：误差和障碍间距](paired100_scene_comparison.png)

[原始逐场景CSV](../orca_maps_diffusion_actual_array_peer_v1/all100_scene_metrics.csv)、[独立数组核对](../orca_maps_diffusion_actual_array_peer_v1/actual_array_peer.json)、[绘图脚本](plot_paired_scenes.py)。图中每点为事前选定的一场景一个窗口，未删除退化或碰撞预测。

[原始配对报告](../orca_maps_diffusion_paired_evaluation_v1/actual_comparison_v1/comparison.json)、[共用输入与每批完整预测目录](../orca_maps_diffusion_paired_evaluation_v1/actual_comparison_v1/)、[真实CPU关闭记录](../orca_maps_diffusion_paired_actual_runs_v1/comparison_actual_launch.json)。最终执行使用修正异常清理的owner v2；模型数学和原v1场景登记保持。

同185源码、固定原AMP和相同物理／路线／引导参数的旧、新Diffusion实测与CPU核对均已完成：[实际物理配对](../default_newDP_physical_actual_runs_v1/README.md)。新模型200秒出现1次跌倒、70个行人接触pair-controls、9次路线顶点到达；旧模型0跌倒、0接触、21次到达。离线改善未转化为本轮闭环改善，默认保持旧权重，新模型的native4／15分支未执行。此次扩展是本地map corpus：800／100／100场景，4–15行人、每场10秒、最多20障碍；不是作者原数据，也未补齐TRACE的ORCA-Interact数据。封存34条行人测试输出未用于训练或调参。
