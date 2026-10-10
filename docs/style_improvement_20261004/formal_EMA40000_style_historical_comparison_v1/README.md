# EMA40000 与 OLD63e：固定 AMP62 的历史风格对照

本记录只整理已经封存的 JSON、标量和真实 CPU owner 记录。没有重新执行指标、读取数组或权重、构造 NN、启动 SDK/GPU，也没有读取 Test34 bank、动作或输出。完整来源、当前文件权限、逐 bin 数值和限制见 [comparison.json](comparison.json)。正式 40000 步主结果和默认配置保持不变。

## 结果与取舍

两组使用同一个 AMP62c4 控制器、4 人 shared-obstacle fixture、200s/6000 controls、seed7、guidance20、alpha0、S1、原 causal reference speed1.5m/s/acceleration2m/s² 和 replan15。OLD 使用 DP63e7；新组使用正式 EMA40000。数据和模型训练 recipe 随 DP 改变，不能把此对照归因于某一个训练超参数。

| 已保存指标 | OLD63e | EMA40000 |
| --- | ---: | ---: |
| 90D speed-bin 等权 macro MMD²，共同 5 bins | 0.07238060315284697 | 0.055925381836245736 |
| 45D pose MMD²，sigma1/cap2048 each | 0.020156771501670168 | 0.02297688065796266 |
| K50 已占用 / 覆盖 | 43/50 = .86 | 43/50 = .86 |
| K100 已占用 / 覆盖 | 72/100 = .72 | 71/100 = .71 |
| K50 normalized entropy | .6662098900698707 | .6749666071767377 |
| K100 normalized entropy | .6360175656624188 | .6706253715455985 |
| root 物理速度均值，m/s | .5606234859819201 | .7424614886088999 |
| 与固定参考的物理速度分布 W1，m/s | .3533679939881452 | .17137778542414567 |
| 近地 toe 速度代理均值，m/s | .39733391538420626 | .5807479654816293 |
| 近地 toe 代理速度 > .1m/s 比例 | .20717832838487613 | .2790559972931822 |
| 受支撑 foot-body-origin 切向速度代理均值，m/s | .07313596881453263 | .08532028401560977 |
| fall / static actor-controls / unordered peer-pair controls | 0 / 0 / 0 | 1 / 0 / 45 |
| 每 actor route vertices | [2,8,6,5] | [6,6,6,6] |

90D macro 更小，但 45D MMD²、两个足部速度代理均值更大；K50 覆盖不变，K100 少一个 cluster。entropy 和路线变化不证明动作质量或物理安全更好。本记录不作全面超越、整体风格增益或因果声明。

### 90D 的共同支持 bins

| 物理速度，m/s | OLD MMD² | EMA40000 MMD² | OLD / NEW generated count | balanced samples each，OLD / NEW |
| --- | ---: | ---: | ---: | ---: |
| [0,.1) | .16203664593416445 | .06506601250632893 | 5843 / 1163 | 148 / 148 |
| [.1,.5) | .045007834283063985 | .08033514267095665 | 5435 / 5649 | 256 / 256 |
| [.5,1) | .02661006877925609 | .022268538448030117 | 7459 / 10172 | 256 / 256 |
| [1,1.5) | .05132978002607369 | .05230914002966891 | 5138 / 6811 | 256 / 256 |
| [1.5,2) | .07691868674167668 | .05964807552624407 | 100 / 180 | 100 / 180 |

两个中低/中高速度 bin 的 MMD² 更大。两组均沿原 minimum30/cap256 的规则；最高支持 bin 的 balanced 样本数不同，不能声称每个 bin 使用完全相同样本数量。[2,3) 只有 4 个参考样本，[3,+inf) 无参考，两组均为 unknown/null，未补零、未丢弃未知 bins。物理速度 mixture 已改变，90D macro 不能单独决定控制器优劣。

## 分母、失败和真实 CPU 闭合

两组完整捕获均为 24000 actor frames。OLD 的合法同 episode pose 端点为 23996；EMA40000 为 23995。每组排除最初 4 个 actor 端点，新组另外排除 1 个跨 reset 端点。没有 fall/upright/idle/outlier 质量过滤；coverage 对全部合法端点计算。MMD 使用原 deterministic/balanced cap，因此不能保证每一个失败样本进入 capped MMD，完整失败样本仍保存在来源中。

新 raw audit 为全 6000 controls、24000 physics substeps、96000 actor-substeps、384000 foot-substep observations；真实 F1、peer-pair45、static0，物理 zero gate 未通过。peer actor-controls 为 90。contact flag 是每个 30Hz control 内四个 120Hz substep 的 union，root/reference 为 control 末端快照；45 不等于独立 encounter 数、精确接触子步持续时间或 impulse。

真实 style90 owner：session96705，launch `e8cf33` → completion `657bb7`，closed0/child_exit_observed=True。pose45 owner：session33799，launch `17f4f4` → completion `d886a7`，同样 closed0。owner 的报告、sidecar、reference JSON/import ledger 的 bytes/SHA/readonly 均已用小文件元数据独立核对；没有为此打开 basis NPZ。

## 固定参考、数学和历史来源限制

OLD90 的 `backward_ground200` 与 OLD45 的 `guidance20` 指向同一份历史 states SHA `7597743b18e76260aecea84fe732aaacad3914570683d671b5a3e076e9f2ab8e`，不是全22 tracking 或另一个 native 场景。两份历史报告当前 **mode0664**；它们的完整当前 SHA 与固定来源记录吻合，但不能称 OS readonly。本轮没有更改这些历史文件权限或内容。

已封的 `OLD_current185_vs_historical.json` 记录当前185 OLD 与历史选定 fixture 的全部 physical/reference 数组 dtype、shape、C-order bytes 相等；physical NPZ 本身 SHA 也相同。这里仅核对该已有 JSON 证据，没有重读或重算数组。由此可引用历史固定指标作为已完成 current185 OLD 捕获的对照，不能说本轮重新运行了 OLD style metric。

四份报告完整 reference 字典相等：validation22、256×61 固定选择、选定 body/root-rotation 数组 SHA 相同。90D 使用原82ee入口数学与 b09fa58c 特征；45D 使用原208ff数学和 b25 已保存 K50/K100 centers，无 refit。45D 的 fit2048/available15360、fit indices SHA 和 fit samples SHA 均一致。新 report 有元数据适配和验证子集 resource admission，不声称执行了原全 manifest、全 split tensor validator。

| 固定身份 | 完整 SHA256 |
| --- | --- |
| 共同 AMP62 控制器 | `62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b` |
| OLD DP63e checkpoint | `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac` |
| EMA40000 checkpoint | `4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff` |
| v7 train manifest metadata | `68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca` |
| validation selection c274 | `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564` |
| saved pose45 basis b25 | `b25dffa6612d6202b5faf534381c24a4558f347639b366214ec61f5a2478dcdb` |
| OLD90 report | `fa2e158e6888fe464f9a916cdbf0789f03858718ade01544c877320ea0db1306` |
| OLD45 report | `9478afe6b16540f7624867e1d736f30f78399c84177c12d6c9b616bc496c3780` |
| NEW90 report | `3cdd18a2771922e6913f9b364ee7b0352806a089ac104e266a1daac993e9cdf7` |
| NEW45 report | `6764cb597cfabb0cb47f11efbc39f2526480aed7eea8040bc2663ed05e02458c` |
| current185 OLD↔历史数组 identity proof | `faac9271671c0d2f0bef905c5c357ce8e36e7a74b4d2fe219a7c00bbf58dde56` |
| NEW raw audit | `f792375fc55f3c6fd8d50349a98760611bda204d4c6ae93da3d2ef8a37c483d7` |
| NEW90 CPU owner | `83155e5b6d622018dedab869830c535b323295978f8cc10ef684f1402851f283` |
| NEW45 CPU owner | `faa98c1d07450f89d830d76b8e526c1a32b3aa577864256e52495a210f5ab178` |

模型/bank/basis 身份在此由既有 JSON 记录引用，未重新打开大文件。每份实际读取 JSON 的完整路径、bytes、SHA、mode/nlink 都在 [comparison.json](comparison.json) 中。

## 接触与足速度的可解释边界

OLD 和 NEW style90 中 `measured_foot_contact_body_origin_speed_m_s`、实际 foot contact 与 true COM 通道均为 null。近地 toe 速度是由位置/近地条件得到的几何代理，不是实际接触点滑移。

独立 raw audit 的 foot-body-origin 切向速度采用真实 link-origin velocity 和受支撑条件：`norm(v_link_origin - dot(v_link_origin,n)*n)`，foot resultant >1N。OLD/NEW supported 样本数分别 209618/198307，分母随 support 状态改变。foot origin 在滚动/旋转时也会运动；未观察 contact patch slip、friction force 或 impulse。较大的代理均值只是需要排查的诊断信号，不能直接归因于“脚在滑”。没有专门 heel 接触点测量，不把 Ankle 命名成 heel。

## 下一轮 AMP 动作训练的有证据目标

1. **Root command following。** 独立登记 AMP 分支，固定 planner、reference adaptation 和 fixture；同控制相位检查 root position/heading/velocity 对 reference 的响应，保留失败与 reset。这里速度分布改变、仍有 F1/peer45，说明需要查控制跟随和安全；本对照没有重算新的跟踪误差，也未识别其原因。不要仅靠90D选控制器。
2. **Heel/toe 支撑诊断。** 现有 L/R Ankle、Toe 的 support 与 body-origin 切向速度，以及近地 toe 速度，可作次级诊断。实际 heel/contact-point 新测量必须独立实现并实测，不能从本报告代理补出，也不应以代理下降代替物理合格。
3. **Fall/peer/static 硬门。** 保留全部6000 controls，完整200s要求 fall0、static actor-controls0、unordered peer-pair controls0，才满足已经登记的本地 native 扩展物理条件。EMA40000 当前未通过；90D 改善不取消硬门。无 MMD 启动质量过滤、自动默认晋升或历史选优。

这些是下一轮可测目标，不是已实现的新训练效果。这里的局部链长尺度、bandwidth、clustering、身体集合与作者未公开的设置不能声称论文 Table III 同口径，也不作严格作者实验复现声明。
