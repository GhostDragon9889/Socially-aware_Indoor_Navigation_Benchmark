# 地形统计保护：完整 22 段追踪与四人 200 秒评估准备

**CPU 注册通过，未来 step2000 payload 和全部评估效果尚待实测。** 当前训练由 root 在独立环境实际启动；本准备不执行 GPU/Kit、不改931隔离源、181生产源、模型、参考或历史报告。

## 固定合同

新注册脚本实际校验931 copied files和共享资源完整SHA，加载新绝对profile并调用隔离源码的两个真实CLI parser；没有Torch、NumPy或SDK导入。相对旧state-action同预算试验，training字段仅 `task_terrain_statistics_mode: legacy -> preserve_declared_suffix`。保护public PACER参考task[20:1044] mean/variance；prefix20、全局count及其既有更新公式保留。不是对旧checkpoint的回写或推理时换RMS，也没有据34 test调参。

- 完整validation22，所有motion IDs/frames保留，无max-clips/max-steps/择好帧。最低1秒、每帧平均body error>.5m的既有失败规则、seed7、deterministic、120Hz/30Hz、原始ref/初始化/phase一致性另查。11982 metric frames是固定fixture预期，不预造结果。
- shared cuboid四人6000controls/200秒，同guide20/ETA0/10 timed A* points、peer5/r.6、SDF10/r.8、causal1.5m/s及2m/s²、15步同步、true-substep power和ground1N。old57字段只改profile/checkpoint/output及隔离scene路径；scene字节完全相同。
- 两个entrypoints与旧926逐字节一致：driver6c7f9a…、tracking a89f05…。新旧均旧完整 PaperAMPNetwork adapter（task/actor/critic/D等完整构建验证，policy接口）；**不使用此次当前生产FrozenAMPActor缩减**，不另创建推理/arm/native世界。
- 新完整2000 fresh训练16.384M，两固定槽组各8.192M，与旧state-action2000同预算；与更早state-only/mixed1000/negative1000的比较还包含D/初始化/RMS/时序/课程/预算/RNG等多项，不能归为单变量RMS效果。

`registration.json` SHA498e353b…在新executor中硬绑定。执行还要求真实sealed2k完整SHA、deps新schema终点admission及actual executed-owned-import ledger。旧schema、partial milestone、未验证suffix/reference/hook、错来源不能放行；source931 manifest4227b3d8…/launcher6d56a3ef…固定。29项CPU admission/CLI负例通过，Ruff通过。未来output目录尚未创建，SDK-free解析通过不是missing checkpoint资源预检通过。

## 评分

主指标保留c274固定validation选择、90D pose+velocity、sigma1、原speed-bin/min30/max256方法，失败与闲置帧保留，只有reset跨界导数间隔被既有方法排除；另报完整22跟踪错误/成功/跌倒、200秒route/contact/idle/support，不以较小loss/MMD掩盖跌倒退步。

45D仅独立pose-only诊断。保存的 `pose_diversity_candidate_v1/actual_evaluation_v1/reference_basis.npz`（SHA b25dffa…）含centers_50/centers_100，来自c274 AMASS validation15360合法端点中的固定2048参考样本。**不是作者官方centers，也不是2848训练clip聚类。** 新wrapper直接加载保存矩阵，替换仅fit入口，冻结评分/pose extraction/kernel不变，生成样本绝不参与拟合。无refit、无新调参；不替换90D，不与论文TableIII数值直接比较。

所有新读者现仅prepared，实际终态数据缺失不会发布PASS：source/ledger bridge、完整22 90D+相同帧误差分解、200秒native raw force/COM/counter+idle、90D、saved-center45D。原f851/ee0/d19c9数学gate保持，epoch-specific准入分开。更多指标只是诊断，执行通过不等于质量改善。

## root 单 GPU 执行顺序

见commands.json。SDK解释器保留**词法 `.venv/bin/python`**，不得Path.resolve成共享uv裸binary。GPU命令的 `ACTUAL_SEALED_STEP2000_SHA256` 目前是显式无效占位符，executor会拒绝；实际终点后由root填真实SHA及deps终点admission文件，不默认mutablelast。

1. 独立sealed2k完整终点准入。
2. root完整22（protected_tracking2000_v1），CPU90D/phase/错误分解。
3. root四人200秒（protected_nav2000_v1），CPU raw/contact/idle、90D及saved-center45D。
4. 保存同协议旧2k与新实验全部真实结果；没有34 test选择、没有自动推广default。
