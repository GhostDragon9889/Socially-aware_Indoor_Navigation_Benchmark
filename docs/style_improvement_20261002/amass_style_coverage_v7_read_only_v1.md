# AMASS v7：2,848 段动作的只读覆盖审计

**结论：数量达到2,848，语义与运动覆盖仍不均衡；本次没有重筛数据、改保留集、训练或产生新权重。**

只读 CPU `weights_only=True,mmap=True` 检查原训练库11,303条和 v7 2,848段。v7 manifest、1.15GB bank、9.56GB 原 train bank 的完整 SHA256 本次重新计算并匹配；未读取保留集模型评估输出。

## 1. 类别与真实运动量

| v7 互斥代理类 | 段数 / 比例 | 总时长 | 每段根速度中位数的中位数 |
|---|---:|---:|---:|
| walking_like | 720 / 25.28% | 3692.2s | 0.483m/s |
| running_like | 17 / 0.60% | 58.4s | 2.219m/s |
| turning_like | 1568 / 55.06% | 10440.1s | 0.662m/s |
| sidestepping_like | 116 / 4.07% | 871.9s | 0.380m/s |
| standing_looking_like | 427 / 14.99% | 2284.9s | 0.040m/s |

- `running_like` 只按整段根速度≥1.8m/s分类，不是跑步语义真值。候选池也是17段，当前平衡选择没有漏选其他已合格 fast 类；这17段仅58.4秒、来自CMU3/ACCAD14，全部不足250帧，因此没有 diffusion-eligible 整段 fast 类。
- 当前159段包含官方或文件名 run/jog/sprint/scamper 线索，分布为turning93、side25、walking16、fast16、standing9；其中34有官方描述、125仅文件名。fast 类还包含1条名称为side_step_left，速度阈值不保证语义一致。
- 已有冻结10帧窗口证明含 **6,063/494,792 =1.23%** fast窗口，涉及168父动作；low_speed窗口为22.31%。17/2,848不能替代窗口覆盖。此处复用原证明，不伪称新重算其采样结果。
- `turning_like` 的55.06%由整段 yaw span≥0.8rad分类；慢速/fast/侧向有优先分支。它不表示55%时间在原地转身。本次根运动量来自真实相邻姿势，未进行最佳时间对齐。

## 2. standing427并非427条真静止

- 427段占14.99%，恰到15%数量上限；候选450段，最终截去23段。
- 固定诊断阈值下：**103**段为运动学安静代理、**3**段为“根基本静止但足部明显活动”代理、**321**段为混合或未定。安静不等于经标注的true idle；其余也不等于坏动作。
- 180/427 =**42.15%** standing文件名含push：KIT121/EKUT54/Eyes5。KIT存在明确`push_recovery_stand_*`，EKUT存在`PushStand*`；原正则含stand、负正则没有push/recovery，来源类别未与普通站立分开。这是文件名层面的混类风险，没有声称测到外力，也没有自动删除。
- 3个强代理实例：官方CMU `132_55` 描述 **Walk With Wild Legs**；EKUT `PushStandStraightPedar_07`；TCD `Run02`。TCD样本根速度中位数0.086m/s、足部heading-relative活动间隔96.7%，应人工核验原地跑/实验动作，而不能根据root低速写成idle。

诊断预先固定：root速度<0.12m/s的间隔至少80%；quiet需足部相对heading速度>0.3m/s的活动间隔≤10%且foot世界高度P90−P10≤0.05m；active需活动≥25%且至少一足高度span≥0.08m。使用Ankle/Toe位置，不是接触点、力、足滑或真实支撑。阈值与全部逐段读数保存在JSON。

## 3. 来源、标签和实际可复核候选

- 699段/24.54%来自CMU官方动作说明；2149段/75.46%是文件名关键词，最终未知numeric proxy为0。五个style_class全部仍是运动学代理，不是AMASS/BABEL语义标签。CMU数字ID只用于连接已保存官方页面。
- KIT占43.29%（1233段），CMU24.54%（699段）。2,848段仅1,853父动作，644父动作贡献多段、最多4段；uniform-clip prior的有效父来源数约1408，不是2,848独立人物/动作。subject namespace不保证真人身份。
- 原库267条run/in-place描述线索已逐条登记；119条不足当前3秒门槛。20个未选父动作没有本次已检查的标签/时长拒绝，其中CMU5、ACCAD3、Bio12；**尚未通过完整新curation门禁，不是20条可直接加入的数据**。
- 可复核的官方候选：CMU `127_05` **Run to Quick Stop**（3.2s）、`127_04` **Walk to Run**（3.7s）。原bank root速度中位数分别0.636/0.559m/s，含混合阶段，不应整段直接标fast。`Run Duck Underneath`、`Run Dive Over Roll Run`另列保留审查，不推荐作为纯导航跑步。ACCAD original_id6679已在原curation重复清单中，不能当新独立来源。

## 4. 建议下一轮的具体步骤

1. 先将普通idle/look、外力恢复、stationary-root足部活动分开登记；复核180条standing Push来源，保留无法确认的语义为unknown。
2. 用官方/可追溯动作描述与短窗口速度、转向、双足时序联合设计采样先验。已有159条fast词线索与6063个真实快窗口，改类名比例不能增加实际快步信息；作者的类别比例仍未知。
3. 审核短的CMU run/jog源；将AMP-only窗口和5秒轨迹完整性分别准入。任何降低时长门槛需新的显式bank/profile及边界/FK/速度/重复/父源隔离检查；不循环、padding或加假root位移补齐数量。
4. 若获准新版本，按父源/数据集限制权重，并用保持不变的validation评估；最终test不参与选数据。当前v7和运行中模型保持原身份。

## 可复现证据

[机器结果](amass_style_coverage_v7_findings_v1.json)、[全部逐段/候选读数](amass_style_coverage_v7_read_only_v1.json)、[速度边界复核](amass_style_coverage_velocity_boundary_check_v1.json)。脚本归档`amass_style_coverage_auditor_source_v1.tar.gz`；Ruff通过。

v7 manifest SHA：`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`；bank SHA：`46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`。

速度一致性：本次T−1真实相邻间隔速度与v7 stored gvs最大分量差3.58e−7m/s；按原T帧末速度carry规则重算的median与metadata最大差2.55e−7m/s。原诊断T−1 median最大差0.06775m/s仅来自末帧carry的样本数约定，不能据此宣称bank损坏。

论文IV-C列walking/running/turning/sidestepping/transition并按标签筛选；IV-B示例包含looking around，但没有公布作者2848 IDs、比例和阈值。[NavIsaacLab原论文](https://arxiv.org/html/2606.26265v1)。本报告不声称复原作者清单或已改善模型。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/audit_amass_style_coverage.py --output /tmp/amass_v7_coverage_NEW.json
```
