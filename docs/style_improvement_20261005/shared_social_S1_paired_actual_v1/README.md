# 新鲜同 AMP 的 Diffusion／SFM：实际完整配对

两臂均已实际完成 6,000 控制步／200 秒，GPU、完整原生 CPU 复算、
初态比较和闭环风格测量全部退出 0。**两臂质量均未通过**，默认模型保持。

## 固定协议

[运行前登记](registration.json)预选 S1 Pop-Art 终态、seed7、N4 fixture、
Diffusion EMA40000；先各自 smoke30，再串行运行两臂6000。不使用历史DP作为新鲜基线。
S1 训练使用 v7 的固定真实 AMASS2,848；作者私有动作ID未恢复。
同一 SMPL、AMP、场景、路线、初态、causal reference、30Hz 控制／120Hz 物理与测量。
完整初态含身体／关节、AMP历史、原生link／COM、episode、路线、资源SHA及RNG。

[完整初态准入](full200s_initial_admission_v1.json)确认全部32通道字节相同，
资源／协议／RNG相同，且为真实 `live_original_world`。
实际进程退出另见 [Diffusion退出](diffusion200s_execution.json)、[SFM退出](sfm200s_execution.json)；
初态reader自身不推测进程退出。

SFM 是公开 NavIsaacLab `d742f0c5…` 方法重建：联合预测150×1/30秒，
速度上限1m/s、完整metric EDT、peer exponential／TTC、4m邻域；预测期固定目标。
输出raw51后使用共同causal151适配。它不加载Diffusion权重，也未恢复作者私有配置。

## 独立结果

| 指标 | Diffusion + S1 AMP | SFM + 同一 S1 AMP |
| --- | ---: | ---: |
| 跌倒次数 | 5 | 20 |
| 静态接触 actor-controls | 0 | 21 |
| 行人接触 unordered pair-controls | 37 | 31 |
| 路线点 | 3：[1,1,1,0] | 6：[2,4,0,0] |
| 原生双脚支撑占比 | 90.4354% | 76.6417% |
| 同episode根部位移速度均值，m/s | .178477 | .404366 |
| 速度低于.1m/s的有效间隔 | 19303/23991，80.4593% | 9931/23976，41.4206% |
| 45D姿态MMD² | .04492030 | .08635795 |
| 冻结K50／K100覆盖率 | .70／.52 | .82／.67 |
| K50／K100归一化熵 | .381954／.399485 | .549118／.557523 |
| 共同五速度桶90D MMD²均值 | .11212462 | .13122441 |

Diffusion 的较低跌倒和风格距离伴随较低速度／路线进度／覆盖，不能称为整体优胜。
共同速度桶的结果也不是每桶一致改善：SFM在[0,.1)、[.5,1)、[1,1.5)三个桶更低，
Diffusion在[.1,.5)、[1.5,2)两个桶更低。超过2m/s的桶缺少验证参考，保持unknown。

[Diffusion全量原生审计](diffusion200s_raw_CPU_audit_v1.json)和
[SFM全量原生审计](sfm200s_raw_CPU_audit_v1.json)各覆盖24,000 actor-controls／
96,000 native actor-substeps；所有24身体位置匹配原生末子步，失败和reset均保留。
脚部速度来自身体原点，不是接触点滑移；根部／参考距离不是作者Path Error。

[闭环风格完整结果](navigation_style_pair_v2.json)与
[CPU实际退出及8项回归检查](navigation_style_pair_execution.json)绑定当前实现。
参考为原先冻结的validation选窗，15关键点、30Hz、根部中心／heading／固定chain尺度；
原90D数学及K50／K100中心均不改，不使用Test34数组，不重新拟合，不过滤跌倒。
共同桶两臂固定同样样本数和同一组参考索引。两臂5／20个跌倒终止帧的合法区间全部纳入。
首次风格CLI因SFM实际来源标记与短名称不同而退出1，失败源码／日志保留；修正来源guard后v2退出0。

## 重跑 CPU 风格测量

从仓库根目录运行，输出必须是新文件：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_navigation_style.py \
  --diffusion-audit-execution docs/style_improvement_20261005/shared_social_S1_paired_actual_v1/diffusion200s_raw_CPU_execution.json \
  --sfm-audit-execution docs/style_improvement_20261005/shared_social_S1_paired_actual_v1/sfm200s_raw_CPU_execution.json \
  --output /tmp/shared_social_navigation_style_new.json
```

GPU入口与来源检查见[共用初态准备](../shared_social_paired_initial_v1/README.md)。
两个完成的GPU输出和所有报告保留；没有自动模型晋升。

## 论文边界与后续

本次只是在单seed／本地N4障碍fixture完成同控制器消融。
[NavIsaacLab IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)的八场景、N3–15、200秒任务
及作者计数分母仍未匹配，局部MMD²也不能与作者Table III数值排名。
下一阶段需要训练可用的共享控制器，再进行原生多场景／多seed和风格／物理／路线独立评价。
