# N4 原生控制／全身姿态审计：历史实测通过，新鲜两臂待执行

[`audit_shared_social_high_level.py`](../../../scripts/audit_shared_social_high_level.py) 接纳明确登记的 **30 或 6000 控制步**。解决了既有完整 200 秒 reader 无法验收新鲜 SFM／Diffusion smoke 的接口缺口。原接触、episode、参考相位和运动／支撑公式保持；保留每一帧，包括跌倒、接触、静止和 reset。

新鲜两臂入口及固定 S1／seed7／formal DP40000 资源登记见[初态准备](../shared_social_paired_initial_v1/README.md)。**没有启动新的 GPU 任务或捕获新鲜配对初态。** S2 owner 仍独占 GPU；先等它完整退出与独立评价，再执行登记的两臂。

## 实际验收

| 最终检查 | 实际结果 | tool chunk／exit |
| --- | --- | --- |
| CPU 契约 | 7 tests passed；预算逆还原、退出闭合、全场计划、初态链接、非根身体／四元数／速率／dtype 和依赖漂移拒绝 | `148ec7`／0 |
| Ruff | 两文件通过 | `09dd15`／0 |
| 历史短 Pop-Art smoke | 全 30 控制步／120 actor-controls／480 actor-substep samples；3 次跌倒完整保留 | `d705af`／0 |
| S1 终态完整 200 秒 | 全 6000 控制步／24000 actor-controls／96000 actor-substep samples；5 次跌倒、路线点 [1,1,1,0] | `4fe438`→`bf6065`／0 |
| 全身原生来源 | S1 全 576000 个身体位置与同控制步末子步精确相等；全部 2304000 个原生身体四元数通过 norm 容差 1e-3，实际最大误差 2.3841858e-7 | 两个最终实际 reader 的 report |

短测使用早期短检查模型，不是 S1 的终态 checkpoint。上述两份历史捕获均没有新增的初态记录，**不是新鲜 DP／SFM 配对**。7 项测试包含 mock／合成数组，只验证拒绝条件；真实姿态来源证明来自独立、已经正常退出的原生运行文件。

S1 的全部运动／支撑测量与既有冻结诊断完全相等，base audit 全字节相同；6000 步时两个预算派生函数所有 AST 节点不变。全版本与实际 exit、文件 SHA 见 `execution.json`。这里没有重新计算动作风格分数，位置一致核对仅补上风格输入的原生身体来源证明；69 DOF 通道仅核验完整、有限 FP32，没有独立第二份 native DOF 捕获。

## 来源和退出合同

原 base reader 与 locomotion reader 按 SHA 加载。base 只有两项可逆时长／schema 替换；两个函数只改登记的预算 literal 节点，以特定节点逆还原 AST。30Hz 常量和指标公式不改变。全部五个 transitive 来源另按 SHA 核验，报告同时绑定 own source 和原冻结 helpers；前后文件身份必须相同。没有 Torch／SDK／权重加载，也没有修改 S2 的 22 项来源。

退出必须是外部工具观察到的真实整数 0，直接回执必须含真实 `terminal_tool_chunk`。完整历史运行可以显式提供原 post-owner 回执，复用**全部原 parent／五个 child／measurement map／CPU audit／只读捕获**接纳；不会伪造子进程的工具标记。reader 不自证自己进程已退出，需调用它的外部工具另观察实际 exit。

第一版完整 S1 检查 `c33e3e` 退出 1：内部 child 回执没有外部工具标记。失败日志和原源码保留；随后新增显式 parent 接纳，真实 parent `c57118` 已退出 0。没有改写旧回执。最初仅四项测试、parent 版五项、全身版六项、最终依赖版七项，全部相应源码和报告保留；它们是审计演进，不是新增训练或挑选 rollout。

## 使用方式

历史 S1 复算命令已实际执行，重跑时必须换新的输出文件名：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/audit_shared_social_high_level.py \
  --execution-receipt output/diffusion_amp/shared_social_popart_terminal2000_evaluation_v1/navigation200s_execution.json \
  --post-owner-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_execution.json \
  --control-steps 6000 --historical-no-initial --output /tmp/S1_native_audit_new.json
```

新鲜运行使用该次真实外部退出回执，并强制绑定该次初态。以下是**未执行的调用模板**，需替换尖括号内容：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/audit_shared_social_high_level.py \
  --execution-receipt <该次外部工具观察的导航退出回执.json> \
  --control-steps 30 --paired-initial <该次目录>/paired_initial_condition.json \
  --output <新的审计输出.json>
```

该次初态文件／archive 必须与 SDK result 声明一致、来自 live original world；第一个 native control 的 counter／episode、四人 raw t0／reference t0 位置和参考 t0 速度须与初态完全一致。单臂链接仍需另运行初态两臂比较，并观察两个真实退出；新鲜结果不能使用 `--historical-no-initial` 绕过。完整运行将预算改为 6000，不能从 30 步推断 200 秒效果。

## 论文边界与后续

本轮重读[NavIsaacLab IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)：高层消融使用同 AMP、8 场景、N3–15、200 秒；动作风格按统一 15 点／30Hz／root-center／heading／scale 处理。因此本地 N4 的 root→reference 距离不命名为论文 Path Error，接触阈值 onsets 不命名为脚步数。作者任务清单和 rate 分母尚缺，当前没有达到／超越论文的证据。

[AMP 原论文](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)讨论 style/task 分离、mode collapse 及不同任务的训练预算；其 100–300M 样本不等于 Nav 作者未公开的预算，也不是直接延长本地训练即可改善的证据。S2 固定真实 AMASS2848 训练完成后仍须分别核验完整跟踪、固定风格、路线和物理表现，再决定后续单变量实验。默认模型不晋升。
