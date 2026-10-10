# 四场景 Diffusion + AMP 行人模拟：2026-10-10

> 已完成：四个场景分别连续60秒实测与60秒原生渲染视频，全部完整解码通过。以下为本地有限场景结果。

## 本轮方法与训练

采用公开 TRACE 轨迹扩散和 PACER/AMP 物理动作控制，继续使用本地真实 AMASS 专家动作。TRACE 网络权重没有重新训练；调整了扩散指导、速度和转弯约束、源地图净空路径代价与安全制动。AMP 本轮训练 512 + 256 + 128 更新、160 环境，总计 4,587,520 次真实物理交互。前两个候选未通过场景质量验证，保留记录并拒绝交付。最终较宽场景选用 128 更新的小学习率候选，窄场景保留已训练的 AMASS v9 父权重。

最终新权重训练 655,360 次交互，探索累计 1 次跌倒；部署录像源中的所有行人零跌倒、零重置。训练和 CPU 导出核验均正常退出，全部 2,848 个专家 ID 参与，四组归一化统计冻结，权重与实际初始父模型发生真实参数变化。实际初始父权重与公开锚定教师分开记录。未新增外部数据；拟合了训练集 3,770 个连续速度窗口、2,036 个较长初始化窗口。

[Trace and Pace 原始项目](https://research.nvidia.com/labs/toronto-ai/trace-pace/)与[AMP 原始论文](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)是方法来源。本实现为本地适配，作者 NavIsaacLab 私有的五状态扩散和定制 AMP 权重、原始动作清单及八场景协议仍未获得，因此结果均作为本地运行指标。

## 速度与可靠性规则

0.5–1.5 m/s 为正常巡航目标，起步、换向、避障、让行可减速或停车。所有原始30Hz控制帧及每人完整7,200个120Hz物理子步都参加 1.5 m/s 瞬时上限检查，原始低速帧全部保留。巡航由规划意图与方向一致性选择，不能按实际速度排除慢帧；每人至少 5 秒巡航，0.5 秒实际速度窗口的巡航均值在目标区间，低于 0.5 的窗口比例不超过 10%。报告同时列全帧均值及低速比例。

每个录像源为连续 60 秒、1,800 个30Hz控制帧、7,200 个120Hz物理子步。完整检查身体姿态、跌倒、重置、源地图与同伴包络、实际支持力以及超过 1 N 的非地面/同伴接触。补充以实际根位置的完整5秒窗口检测空间停留，交付限制为不超过10秒。足部中性 SMPL 皮肤相对声明支持面低于−1cm的帧比例必须为0；未测量实际接触斑滑移或切向摩擦力。

## 场景与覆盖

原始四场景位于 `/project/EXP/Sence/Sences/`，没有修改源 USD。使用已核验的静态副本、源绑定 PhysX 查询地图及有限解析支持面。保留障碍和未知区域；支持面采用 reference-flat 模式，不代表复刻所有非平坦地面。

YOHO 四人在三个较宽区域，NewMetro 四人覆盖左右两间，二、三层各一人行走在经过验证的主通道。未访问的狭窄通道、隔间与独立区域不算覆盖。覆盖是实际根轨迹0.75m邻域与已知空闲格中心的交集，分别报告整个已知空闲区域、声明净空安全区及出生可达区；它不是身体接触足迹。

## 已发现和保留的问题

二层 v16 最后19.2秒空间停留，虽通过旧20秒门槛，拒绝交付；v21 仍长期停留。二层 v22 及独立种子11 v23 完整通过新10秒、接触、姿态、速度检查，最长空间停留均为0。新增可选站稳停止目标功能默认关闭；二层验证开启，但两个通过样本实际目标更新事件均为0，不能把成功归因于该更新，也没有实测证明它消除此前的停留机制。CPU 四项合约测试通过。

YOHO 四区域候选短测通过、长测第83帧在竖通道偏出安全可规划/连接区域，保留失败后调整为三区域闭合路线。三层 v16 有5帧左手接触，拒绝交付；v17 更保守展望、居中路径和预警余量的完整候选通过。

NewMetro 原右镜头隔断遮挡，已调整位置；原左镜头8帧脚部被裁切，仅调整俯角后完整皮肤投影通过，并核对最近距离原生画面。全帧姿态与中性皮肤投影不是主观真实感认证。

## 录制与资源

视频按原始物理状态以30FPS等时回放，1,800源帧连续对应1..60秒物理区间，只有相机切换，没有插值、拼接不同物理运行、时间拉伸或修改姿态。全帧 FK 必须与真实24身体位置匹配至1mm内；全片 H.264 解码、帧数和时长另行验证。

本轮未加入 EVT 机器人，不将行人回放称为耦合机器人社交导航。原机器人资产已只读检查；其实际动力学与闭环社交导航尚未在本轮验证。

风格参考22条验证动作此前已使用，不作为新未见测试；34条测试动作未在本轮再次用于选型。公开 PACER 预训练的 AMASS 重叠未知。NewMetro 同路线单种子20秒对照，身体根部三维速度均值0.569→0.719 m/s（与下表XY速度口径不同）；不能用单次对照宣称统计显著或优于论文，抬脚幅度也未全面改善。

## 四份新视频与实测指标

- [YOHO：60秒](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/speed_coverage_optimization/videos_final_20261010_v1/YOHO_60s.mp4>)
- [NewMetro：60秒](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/speed_coverage_optimization/videos_final_20261010_v1/NewMetro_60s.mp4>)
- [SecondFloor：60秒](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/speed_coverage_optimization/videos_final_20261010_v1/SecondFloor_60s.mp4>)
- [ThirdFloor：60秒](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/speed_coverage_optimization/videos_final_20261010_v1/ThirdFloor_60s.mp4>)

各片H.264、960×720、30fps、1,800帧、60.000秒。每个原生子进程、拼接和完整解码进程实际退出0。视频镜头分别展示同一物理运行中的局部区域；镜头中的镜面人体是场景反射，不是新增行人。二层明确采用名义+2EV并逐帧核对曝光设置，腿部可见性保持。


## 本轮完整实测结果

| 场景 | 人数 | 全帧平均m/s | 巡航平均m/s | 瞬时峰值m/s | 出生可达安全区覆盖 | 整个已知空闲区覆盖 |
|---|---:|---:|---:|---:|---:|---:|
|YOHO|4|0.650–0.784|0.750–0.817|1.231|56.6%|12.7%|
|NewMetro|4|0.676–0.708|0.830–0.864|1.409|68.8%|29.0%|
|SecondFloor|1|0.498|0.721|1.097|42.3%|19.7%|
|ThirdFloor|1|0.471|0.779|1.326|74.8%|37.1%|

上表速度均为实际根部XY速度；巡航均值按规划意图选择，所有慢帧仍参与全帧统计。覆盖的第一分母是出生可达安全格，第二分母是全部已知空闲格，不能解读为覆盖整个USD。每人原始低于0.5m/s的比例见 [逐人CSV](</home/lfm/Documents/ChatGPT/Social Navigation/trace_pace_reproduction/speed_coverage_optimization/four_scene_verified_metrics.csv>)。

| 场景 | 完整120Hz物理速度峰值 m/s | 超过1.5m/s子步 | 动作姿态筛查 | >1N非地面/同伴接触 |
|---|---:|---:|---|---:|
| YOHO | 1.249 | 0 | 通过 | 0 |
| NewMetro | 1.416 | 0 | 通过 | 0 |
| SecondFloor | 1.118 | 0 | 通过 | 0 |
| ThirdFloor | 1.330 | 0 | 通过 | 0 |

二层独立seed11再实测60秒也正常退出，姿态、速度、接触和10秒空间停留检查全部通过。这是两次局部证据，不是任意种子和拥挤程度的保证。

| 场景 | 冻结22验证参考姿态MMD² | K50姿态覆盖 | K100姿态覆盖 | 双侧步态周期脚部原点抬升中位范围 cm |
|---|---:|---:|---:|---:|
| YOHO | 0.03340 | 0.28 | 0.21 | 3.34–4.87 |
| NewMetro | 0.02995 | 0.32 | 0.28 | 4.01–5.09 |
| SecondFloor | 0.03690 | 0.30 | 0.19 | 3.71–3.88 |
| ThirdFloor | 0.03502 | 0.32 | 0.23 | 4.15–4.35 |

这些是本地冻结参考与几何诊断，速度、掉头和停顿不同会影响结果。检测到双侧行走周期和屈膝不等于完全消除拖腿；脚部身体原点不是脚底接触斑。没有根据这些数值声称超过论文，也未新增人类感知实验。

## 代码和权重发布

- 配置：`/project/EXP/SangforAndSaut/ProtoMotions/configs/diffusion_amp/four_scene_speed_20261010/`，四份profile、scene、camera及新训练配方。
- YOHO/NewMetro AMP：`data/pretrained_models/local_amass_finetune/speed_conservative_20261010/Humanoid_AMASS.pth`，SHA256 `cb2783690a683cbfbbf14e0788d99a2a49d0da0513d11399682db776deb3553c`。
- 二/三层 AMP：`data/pretrained_models/local_amass_finetune/walking_turn_20261009/Humanoid_AMASS.pth`，SHA256 `0cd961343d9828f95b0fcd6ed38ca8fb391140115c1a8fe808ff4ec1884434a8`。
- 已补充净空居中A*代价、真实速度分桶步态先验、可选停止意图站稳功能；完整身体/同伴/未知地图保护仍执行。停止意图更新未在本轮通过样本触发，不作已解决停滞的依据。
- 新增独立速度/覆盖、空间停留、120Hz速度、足部皮肤、相机皮肤投影检查，以及带全套门槛的录像入口。录制工具检查源SHA与实际退出状态；保留原生清理，未修改SDK制造成功。

发布profile仅修改权重所在路径和准入说明，物理参数与已测试profile逐项相等，权重字节SHA相等；发布scene/camera字节与实测原件相同。地图查询、静态USD及probe路径保留原始证据绑定，因此需保留本机原始资源和任务归档；本发布不是可脱离数据直接跨主机运行的资源包。重跑物理并非逐位确定，仍应重新执行门槛。

### 重新微调选定候选

使用已配置的IsaacSim6.0 / IsaacLab3.0环境，在仓库运行；各输出目录必须是新的。以下是选定v3的128更新配方，之前两个被拒候选不是必需的初始化链。

```bash
cd /project/EXP/SangforAndSaut/ProtoMotions
NAV_PY=/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python
NAV_CFG=/project/EXP/SangforAndSaut/ProtoMotions/configs/diffusion_amp/four_scene_speed_20261010
NAV_PARENT=/project/EXP/SangforAndSaut/ProtoMotions/data/pretrained_models/local_amass_finetune/walking_turn_20261009/Humanoid_AMASS.pth
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/finetune_pacer_amass_verified.py \
  --output /tmp/fresh_speed_amp_training --updates 128 --num-envs 160 \
  --parent-checkpoint "$NAV_PARENT" \
  --optimization-config "$NAV_CFG/locomotion_conservative128_v3.json" \
  --gait-prior "$NAV_CFG/gait_conservative_prior_v3.json" \
  --temporal-gait-prior "$NAV_CFG/temporal_speed_prior_v1.json" \
  --walking-turn-windows "$NAV_CFG/walking_conservative_windows_v3.json" \
  --task-horizons-s 0.5 1.0 2.0 4.5
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$NAV_PY" scripts/audit_pacer_amass_finetune.py \
  --run /tmp/fresh_speed_amp_training --output /tmp/fresh_speed_amp_training_cpu_audit.json
```

### 新物理测试和录像

以下以YOHO为例，依序执行，任一步非零退出应停止并保留失败；不能直接运行后续录制。GPU/SDK任务按场景串行。新训练权重若需部署，更新profile路径及真实SHA后仍需完整场景准入，不能自动认定等于本轮选定权重。

```bash
NAV_SCENE=YOHO
NAV_ACTORS='[[0,2],[1],[3]]'
NAV_RUN=/tmp/fresh_YOHO_speed_physics
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/run_trace_pace_verified.py \
  --output "$NAV_RUN" -- --config "$NAV_CFG/${NAV_SCENE}_profile.json" \
  --scene "$NAV_CFG/${NAV_SCENE}_scene.yaml" --steps 1800 --measure-contacts --seed 7
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/evaluate_trace_pace_reliability.py \
  --run "$NAV_RUN" --execution "${NAV_RUN}_execution.json" --output "${NAV_RUN}_quality.json"
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/evaluate_trace_pace_posture.py \
  --run "$NAV_RUN" --output "${NAV_RUN}_posture.json" --require-walking-screen
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/evaluate_trace_pace_speed_coverage.py \
  --run "$NAV_RUN" --scene "$NAV_CFG/${NAV_SCENE}_scene.yaml" --output "${NAV_RUN}_speed.json"
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/evaluate_trace_pace_spatial_stasis.py \
  --run "$NAV_RUN" --output "${NAV_RUN}_spatial_stasis.json"
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/audit_trace_pace_substep_speed.py \
  --run "$NAV_RUN" --output "${NAV_RUN}_substep_speed.json"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$NAV_PY" scripts/audit_trace_pace_foot_surface.py \
  --run "$NAV_RUN" --steps 1800 --output "${NAV_RUN}_foot_surface.json"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$NAV_PY" scripts/audit_trace_pace_camera_skin.py \
  --run "$NAV_RUN" --camera-plan "$NAV_CFG/${NAV_SCENE}_cameras.json" \
  --actors "$NAV_ACTORS" --output "${NAV_RUN}_camera_skin.json"
PYTHONDONTWRITEBYTECODE=1 "$NAV_PY" scripts/record_pedestrian_coverage.py \
  --run "$NAV_RUN" --quality "${NAV_RUN}_quality.json" --posture "${NAV_RUN}_posture.json" \
  --speed "${NAV_RUN}_speed.json" --substep-speed "${NAV_RUN}_substep_speed.json" \
  --spatial-stasis "${NAV_RUN}_spatial_stasis.json" --spatial-stasis10 "${NAV_RUN}_spatial_stasis.json" \
  --foot-surface "${NAV_RUN}_foot_surface.json" --camera-skin "${NAV_RUN}_camera_skin.json" \
  --camera-plan "$NAV_CFG/${NAV_SCENE}_cameras.json" \
  --output-dir /tmp/fresh_YOHO_speed_video --tag "$NAV_SCENE"
```

另三场使用新NAV_RUN和视频目录，NewMetro `NAV_ACTORS='[[0,1,2],[3]]'`，SecondFloor与ThirdFloor `NAV_ACTORS='[[0],[0]]'`。皮肤相机检查只针对每镜头指定显示区域的行人，全部实际行人仍由同一物理源推进。完整运行后仍需查看原生画面是否被源几何遮挡。

## 核验范围与后续缺口

实际查看每段成片1、6、11、16、21、26、31、36、41、46、51、58秒的代表帧，并补充局部连续步态序列；抽样不能替代连续人工播放或人类感知评价。所有物理子步、姿态/FK、足部皮肤与已指定相机区域仍执行全量自动检查。YOHO主区域灰色地面保持源渲染结果，没有补贴图改变成片。

仍未验证：任意路线/高密度人群、多种子总体泛化、全部狭小隔间和非平坦地面、准确接触斑滑移与摩擦、EVT机器人驱动及闭环社交导航。完成的是本机四个场景的局部稳定行人模拟，正常巡航目标0.5–1.5m/s，允许必要减速；没有声称作者完整协议或超越论文。

完整散列与实际退出证据：[发布索引](/project/EXP/SangforAndSaut/ProtoMotions/docs/four_scene_speed_release_index_20261010.json)。
