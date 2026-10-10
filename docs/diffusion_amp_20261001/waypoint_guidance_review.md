# 固定权重、固定噪声的 waypoint guidance CPU 实验

日期：2026-10-01。只读冻结 social diffusion，未训练、未启动 Kit／CUDA、未编译模型，未修改当前 AMP 的规划参数。新增独立 [benchmark 脚本](../../scripts/benchmark_paper_waypoint_guidance.py) 和 [CPU 诊断测试](../../tests/test_diffusion_waypoint_guidance_benchmark.py)。

## 条件与可复核产物

- 权重：`social_diffusion_training/step_1000.ckpt`，SHA256 `69d4336d002c84fa848a93a65dd622e5222a663fac63f61b3e94c6872b8c9d15`。两轮实验前后内容一致；没有保存新模型。
- 实际 ORCA **test** 四个独立 scene：000020–000023，window IDs `[2139,3784,7190,10959]`，episode／actor／frame／neighbor IDs 均记录。
- 真实邻居数量 `[12,9,13,3]`；224×224 obstacle raster 真实占据像素 `[9815,1561,13245,643]`。history31、future52，目标为 expert future index49（+5s）。
- 完整100步DDPM、FP32、CPU4threads、无compile。每个有限结果的100次噪声抽样完整SHA均为 `ea14d73265d0fdb30c15049e09c12e6c8ac6d91962774f41703c88af14b23547`；输入／权重／条件／噪声相同，只有 guidance scalar 改变。
- expert 参考在这些50future points的40cm障碍disc、80cm邻居pair代理均无重叠；expert最大位置差分速度1.977m/s。

宽范围结果与每个预测 NPZ：[guidance_result.json](../../output/diffusion_amp/waypoint_guidance_cpu_verified/guidance_result.json)。低强度稳定区结果：[guidance_result.json](../../output/diffusion_amp/waypoint_guidance_cpu_stability/guidance_result.json)。初次未捕获 IndexError 的 benchmark 失败记录仍保留，未覆盖；最终脚本会明确记录某个超参失败并继续完成测量。

## 定量结果

每行仅四个真实heldout窗口、一个共同noise seed91。终点／ADE为50帧开环 XY误差；第一0.5秒的推进是计划位移在current→expert5s目标方向上的投影。

| weight | endpoint m | ADE m | 前0.5s向目标推进 m | obstacle point proxy | neighbor point proxy | 最大差分速度 m/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.31174 | 0.67187 | 0.75305 | 14% | 3.5% | 3.019 |
| 0.5 | 1.28954 | 0.66158 | 0.73867 | 13.5% | 4.5% | 2.957 |
| 1（当前默认） | 1.28361 | 0.65864 | 0.73469 | 13% | 4.5% | 2.935 |
| 1.5 | 1.27955 | 0.65705 | 0.73303 | 12.5% | 4.5% | 2.925 |
| 2 | 1.27609 | 0.65595 | 0.73216 | 12% | 4.5% | 2.919 |
| 3 | 1.27025 | 0.65440 | 0.73131 | 12% | 4.5% | 2.914 |
| 5 | 1.26257 | 0.65255 | 0.73050 | 12% | 4.5% | 2.911 |
| 10 | 2.12191 | 0.81391 | 0.78026 | 11.5% | 4.5% | 3.271 |
| 20 | 3.39872 | 1.59413 | 0.58893 | 3.5% | 5.5% | 2.844 |
| 100 | 数值失败 | — | — | — | — | — |

weight100在18次noise draws后出现nonfinite坐标，feature-grid整数索引变成−2^63，触发实际 `IndexError`；不是环境依赖失败。weight20的较低障碍点代理伴随更大的轨迹误差和更差的目标推进，不能独立当作避障改善。

weight1→2 的终点平均改善仅约0.6%；→5也仅约1.6%，前0.5秒向目标推进反而略降。方向cosine1→2为0.794795→0.794904，改变量很小，expert初始方向cosine0.519365→0.519414。一个慢行expert case的首0.5秒方向cosine约−0.918，即当前采样仍朝该expert初始位移相反方向。

采样存在非物理位移：weight1约11%的10Hz差分速度超过ORCA专家最大速度2m/s，最大差分加速度约24.22m/s²；weight2/5的最小neighbor surface gap反而从−0.324m变为−0.355/−0.389m。更低endpoint误差不能直接说明更适合低层SMPL跟踪。

## 十个时点 oracle 全路径补充实验

在同四个heldout输入和相同100次noise draws上，进一步使用真实future在`[0.5,1,...,5]s`的十个XY位置。**这比仅5s endpoint多给了九个真实未来点，不是公平的预测质量比较**；用途是检查timed path guidance能否改善计划早段与目标约束。实际A*路径只有几何路线，并不具备这里oracle未来的动态避让信息。

| oracle条件 | endpoint m | ADE m | 前0.5s向目标推进 m | 初期expert方向cosine | obstacle proxy | neighbor proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K1，weight1 | 1.28361 | 0.65864 | 0.73469 | 0.519365 | 13% | 4.5% |
| K10，weight1 | 1.26770 | 0.65938 | 0.73889 | 0.519065 | 13.5% | 3.5% |
| K10，weight10 | 1.22606 | 0.64075 | 0.71666 | 0.519556 | 11.5% | 4.5% |

K10 weight10中loss对K个点取平均，所以每点effective weight=1；它数值稳定，不等于前表不稳定的single weight10。终点较single1好4.48%、ADE好2.72%，但前0.5秒目标方向位移低2.45%，方向cosine几乎没有改善；慢行case仍与expert初期方向相反（−0.917）。最深neighbor surface gap从−0.324m恶化到−0.418m，超过2m/s差分速度的比例由11%升至13%。

原始记录、准确timedpoints、全部每case指标及预测NPZ见 [oracle10结果](../../output/diffusion_amp/waypoint_guidance_cpu_oracle10/guidance_result.json)。该实验4.1秒、exit0、CUDA未初始化、checkpoint SHA未变。甚至oracle轨迹也只小幅改善并有代理重叠，不能据此断言A*多点会修复真实导航；它可以作为明确新eval条件验证，当前corrected训练仍保持single endpoint/default1。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:src \
  ../IsaacLab-Protomotions/.venv/bin/python scripts/benchmark_paper_waypoint_guidance.py \
  --cases 4 --waypoint-count 10 --guidance-weights 1 10 --cpu-threads 4 \
  --output-dir output/diffusion_amp/waypoint_guidance_cpu_oracle10_new
```

## 为什么大权重不稳

对于本实验一个waypoint，闭式guide梯度是 `meanXY−targetXY`。原有更新不改变：

```text
guidedMean = mean − weight * posteriorVariance * (mean − target)
guidedMean − target = (1 − weight * posteriorVariance) * (mean − target)
```

固定模型最大posterior variance为0.99875766。要使**这一个显式梯度更新**不放大mean-goal残差，需 `weight≤2/maxVariance≈2.00249`。5／10／20／100会在早期子步反转并放大该残差；这一局部代数条件并不证明完整神经DDPM、轨迹连贯性或物理安全。对于K个**不同时间索引**的waypoint，梯度按K平均，相应界限为`2K/maxVariance`；K10时约20.0249，重复索引则需要按累计重复数量另算。

当前sampler在t>0使用variance×gradient，最后t0输出由denoiser重新预测而没有硬端点投影。goal仅作为采样soft guide，不进入已训练context encoder的独立goal feature；因此提高scalar不能保证整个5秒轨迹到达任意新目标。添加硬投影、改变指导的时步schedule、重训goal-conditioned网络或改用不同sampler都属于新方法／阶段，不能伪称此次仅换scalar已修复。

## 参数建议与方法边界

保持当前 **1**。若在训练完成后单独、显式新评估探索参数，可试 **1.5或2** 并记录runtime参数和checkpoint SHA；收益在此次小样本中很小，不建议将5以上作为导航修复，10/20退化，100实际发散。

这四个目标恰为模型训练分布中的真实expert未来endpoint，四个expert位移约6.735、7.531、0.436、9.316m。共享物理fixture的任意A*目标与受训练SMPL实际姿态／速度是不同条件；本实验不能推断真实200秒评估的4.68m终点误差会改善，也不能报告导航成功率。地图代理按40cmdisc与静态polygon计算，邻居使用固定expert future，不会回应预测行人；它不是真实多行人共同执行或PhysX body接触。

可复现命令（新output目录）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:src \
  ../IsaacLab-Protomotions/.venv/bin/python scripts/benchmark_paper_waypoint_guidance.py \
  --cases 4 --guidance-weights 0.5 1 1.5 2 3 5 --cpu-threads 4 \
  --output-dir output/diffusion_amp/waypoint_guidance_cpu_new
```

两轮单端点正式benchmark均exit0，分别约7.2/7.7s，`cuda_initialized=false`、`checkpoint_unchanged=true`。四个CPU诊断回归验证准确0.5/5s时序、终点正确却初始反向／200m/s末端跳跃、静止与frame0错误、K10时点的next-frame索引；最终 **4 passed（0.06s）**，新两Python文件 Ruff PASS。完整证据与源码hash见 [汇总 JSON](waypoint_guidance_review.json)。
