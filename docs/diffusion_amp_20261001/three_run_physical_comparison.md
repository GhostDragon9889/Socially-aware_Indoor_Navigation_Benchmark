# 三轮四人200秒真实物理评估对照

日期：2026-10-01。三轮均为root实际执行PhysX／Diffusion5D／action-conditionedAMP／SMPLoverlay／真实MP4；本代理仅在timed状态passed后运行4threads CPU evaluator并审查已有NPZ／JSON，未启动Kit或CUDA、未修改业务或模型。完整输入／模型SHA与参数、逐人事件和指标见 [比较JSON](three_run_physical_comparison.json)。

## 结论

**当前仍未达到稳定导航。** corrected timed路径减少A*失败并到达2个路线顶点，但仍16次跌倒、4/4人至少跌倒一次，行人几何重叠代理明显增加。姿态覆盖率改善不等于论文复现达标；所有导航成功率与真实接触率仍未知。

## 指标表

| 指标 | 旧错时PDproxy＋单端点 | 修正功率＋单端点 | 修正功率＋静态A*时点路径 |
| --- | ---: | ---: | ---: |
| 实际控制帧／人数 | 6000／4 | 6000／4 | 6000／4 |
| 物理时长／actor-seconds | 200s／800 | 200s／800 | 200s／800 |
| fall事件／逐人计数 | 16／[3, 5, 5, 3] | 16／[8, 2, 2, 4] | 16／[2, 3, 7, 4] |
| fall frame-agent比例 | 0.0667% (16/24000) | 0.0667% (16/24000) | 0.0667% (16/24000) |
| affected observed episodes比例 | 80% (16/20) | 80% (16/20) | 80% (16/20) |
| 至少跌倒一次的行人 | 100% (4/4) | 100% (4/4) | 100% (4/4) |
| fall事件/100actor-seconds | 2 | 2 | 2 |
| 时限reset | 0 | 0 | 0 |
| 路线顶点总数／逐人 | 1／[0, 0, 1, 0] | 0／[0, 0, 0, 0] | 2／[0, 2, 0, 0] |
| A*失败查询数 | 271 | 68 | 1 |
| 合法起点connector恢复查询 | 旧代码无此恢复 | 0 | 457 |
| 同刻plan tracking mean／RMS m | 0.205935／0.253318 | 0.215236／0.262486 | 0.204316／0.255367 |
| 计划末点→conditioninggoal mean m | 4.683032 | 5.307368 | 4.642314 |
| root-point地图代理frame-agent count | 0 | 0 | 0 |
| 30cmagent-disc代理frame-agent count | 1494/24000 (6.2250%) | 34/24000 (0.1417%) | 1604/24000 (6.6833%) |
| agent-disc代理事件／affected episodes | 22／30% | 2／10% | 34／40% |
| 实际PhysX地图／行人接触 | 未记录 | 未记录 | 未记录 |
| PoseK50 coverage／normalizedentropy | 0.72／0.642781 | 0.84／0.713463 | 0.90／0.724494 |
| PoseK100 coverage／normalizedentropy | 0.54／0.625749 | 0.67／0.654700 | 0.78／0.714662 |
| generated→heldout Gaussian MMD／MMD² | 0.363882／0.132410 | 0.330677／0.109347 | 0.337235／0.113728 |
| reset ID/mask边界不一致 | 0 | 0 | 0 |

## 实验合同与对照边界

- 三轮共享相同物理rig／masses／limits／gains／120Hz、控制30Hz、几何cuboids／origins／routes；JSON保存逐字段比较。所有socialdiffusion权重SHA均为`69d4336d002c84fa848a93a65dd622e5222a663fac63f61b3e94c6872b8c9d15`。
- corrected两轮完整AMP文件SHA均为`1d5051cfc36df71b706e4a136458d32539a8084f15c3d1eb9e4dd2b11568213b`，各1000updates／2.048m保存的训练transition；旧轮使用不同trainedAMP与旧错时PDproxy。
- corrected训练／评估功率来自每120Hzsubstepprojectedjoint effort×同步postvelocity，四平均W；它是joint传递功率而非孤立motor通道。旧capture缺历史power不能事后精确修正。
- 训练config仅minibatch上限不同：旧2048、corrected16384；64env×32steps实际每rollout2048样本，源码每epoch均形成一个2048尾批。corrected两轮完整trainingconfig相同；配置差异仍如实保存在JSON，不隐藏成完全相同的实验。
- correctedendpoint使用一个5sA*lookahead，guidanceweight1；timed使用同一静态A*路径的10个0.5..5s点、totalweight10／每点effective1，并额外允许最多0.5m合法connector找回dilated-free起点。**不是futurepedestrianoracle；connector恢复没有修改physicalroot／DOF。** 457是查询恢复次数，不是457次actor移动或独立事件。
- timed减少A*failure68→1同时增加457connectorrecoveries，不能把改善单独归因于10waypoints。root保留标准真实fallresets；从NPZ排除这些episode边界后，非reset最大XYcontrolstep位移旧0.116／endpoint0.127／timed0.138m，无新增可见跳转。
- 三轮24,000frame-agent样本、16跌倒事件、20observedepisodes（16以fall终止、4末段right-censored）。立即reset让fall逐帧比例很小；不能把0.0667%当作者未公布分母的fallrate。6001episodebounds下仅运行6000steps，末段censoring不等于导航成功。
- 全局目标字段实际是A*conditioninglookahead，可随重规划移动；lowtrackingerror不证明完整routearrival。timed只有actor1到达2个顶点，其余0；全局success合同没有公开，successrate仍None。
- agentdisc代理radius0.3m，统计任一相交pair涉及的frame-agent，排除selfpair；没有PhysXpaircontact，不能说0root-pointmap代理就是全身无碰撞。
- 三轮K50／K100trainclusterJSON SHA完全相同，train/test参考与2048generatedpose采样方法一致；姿态包括reset-first、terminal/fall帧。局部pose-dependentheight归一化、keypoint选择、单poseGaussianMMD的参数不是作者完整协议。
- 单scene、单seed、每条件一条200s记录；物理轨迹／fall／planningcalls不同后RNG消费不同，不能作为paired-noisecausal消融或多seed统计。

## 实际产物

- 旧错时PDproxy + 单端点：[runner](../../output/diffusion_amp/shared_obstacle_evaluation_1000/result.json)、[CPU指标](../../output/diffusion_amp/shared_obstacle_evaluation_1000/pedestrian_metrics/evaluation.json)、[MP4](../../output/diffusion_amp/shared_obstacle_evaluation_1000/pedestrian_motion.mp4)。
- 修正功率 + 单端点：[runner](../../output/diffusion_amp/shared_obstacle_corrected_endpoint_1000/result.json)、[CPU指标](../../output/diffusion_amp/shared_obstacle_corrected_endpoint_1000/pedestrian_metrics/evaluation.json)、[MP4](../../output/diffusion_amp/shared_obstacle_corrected_endpoint_1000/pedestrian_motion.mp4)。
- 修正功率 + 静态A*时点路径：[runner](../../output/diffusion_amp/shared_obstacle_corrected_timed_1000/result.json)、[CPU指标](../../output/diffusion_amp/shared_obstacle_corrected_timed_1000/pedestrian_metrics/evaluation.json)、[MP4](../../output/diffusion_amp/shared_obstacle_corrected_timed_1000/pedestrian_motion.mp4)。

本代理timedCPU评估exit0，约1.12s；endpoint由resources代理已完成，旧轮已有CPU指标，均未重复执行物理仿真。
