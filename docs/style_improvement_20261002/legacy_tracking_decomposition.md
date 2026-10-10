# Legacy full tracking：同帧误差拆解（2026-10-02）

## 结论

完整22个 validation 动作、11,982个 post-step metric frames 已在 CPU 复核。**原 benchmark 指标完全不变：位置0.37729345m、旋转0.85947860rad、成功6/22（27.27%）**。作者精确的 pose-error／成功率协议仍未确认；以下诊断不得替代 primary 指标，也不得当作论文结果。

误差同时包含根运动与肢体姿态差异：根XY漂移均值0.311570m、wrapped root yaw mismatch 均值0.618995rad；分别去掉自己的根XYZ与heading后，局部24body位置误差仍0.135250m，heading-aligned body旋转误差仍0.562577rad。移除heading只把平均全身旋转误差降低34.54%，所以不能把0.859rad主要归结成一个纯heading偏差。此下降比例是描述性比较，不是可加的因果／方差占比。

## 1. 数据闭包与初始化

- 输入 result：`output/diffusion_amp/motion_tracking_legacy_full_validation_v1/result.json`；脚本实际核对全部22个 capture SHA，与其 result 声明一致。
- quaternion XYZW／24body canonical named SMPL，未改body顺序；初始frame9、随后连续10..end，始终同episode；initial reset与metric mask的合同实际验证通过。
- 重新用原 `body_tracking_errors` 计算每帧／每body primary errors，与保存数组、各clip summary和aggregate在1e-12容差内一致。
- 每个clip初始化24body平均误差：位置最大1.4467981e-6m、旋转最大1.1201053e-6rad，均小于1.5e-6。这支持命名、资产rig、初始化pose映射正确。
- **逐body最大值**为3.4390382e-6m／2.3253452e-6rad；不能把“每clip的全body均值<1.5e-6”写成“每个body都<1.5e-6”。这是浮点级reset数值一致性证据，不是后续actor tracking质量。

## 2. 定义与总量

所有诊断使用同一原始captured frame；没有phase shift、时间拉伸、best-match alignment、后验选最好yaw或修改success threshold。真实fall之后的原metric frames全部保留。

heading为root局部+X轴在世界XY平面投影的atan2。Local位置=`Rz(-own_yaw) @ (body_pos-own_root_pos)`，actual/reference各用自己的根；不做骨尺度归一化、不拟合高度或tilt。Heading orientation=`Rz(-own_yaw) @ Rbody`的同帧SO(3)geodesic。额外的full-root-relative orientation和parent-relative joint orientation分别移除整根rotation／自身parent rotation，单独标注为肢体诊断。它们均非paper primary。

| 诊断 | 帧加权均值 |
|---|---:|
| 原 world24body位置误差 |0.377293m|
| 根XY漂移 |0.311570m|
| 根XYZ漂移 |0.315300m|
| 根height绝对差 |0.026945m|
| 仅root中心化body位置误差 |0.157531m|
| root中心化+heading local body位置误差 |0.135250m|
| 根yaw wrapped差 |0.618995rad|
| 根full SO(3)差 |0.648278rad|
| 原world24body SO(3)差 |0.859479rad|
| heading-aligned24body SO(3)差 |0.562577rad|
| full-root-relative24body SO(3)差 |0.567145rad|
| parent-relative24body joint SO(3)差（Pelvis=0） |0.314163rad|

原 world body位置 MSE 的精确恒等分解（含交叉项）为：

```text
0.1890841120 m² = rootXYZ drift0.1318609047
                  + centered pose0.0512790901
                  + translation×pose cross0.0059441171
closure numerical error =1.91e-17 m²
```

根XYZ项数值约占总MSE的69.7%，但cross term存在，不能称为独立根漂移贡献率；local heading诊断也不能直接从世界误差相减得到独立肢体损失。

16/11,982帧的根+X投影norm<.1，接近竖直时heading会不适定；本报告保留这些帧（包含原fall期），不靠删除坏姿势降低均值。

## 3. 按body拆解

下表均为全部原metric frames均值。parent-relative表示实际named MJCF父子rotation差；小hand local误差不代表hand世界pose跟踪好，因为它继承wrist／elbow偏差。

| Body | world旋转rad | 去heading旋转rad | parent-relative旋转rad | local位置m |
|---|---:|---:|---:|---:|
|Pelvis|0.648278|0.146189|0.000000|0.000000|
|L_Hip|0.798494|0.386738|0.406800|0.014297|
|L_Knee|0.805370|0.488753|0.474557|0.135318|
|L_Ankle|0.780653|0.524636|0.249667|0.243851|
|L_Toe|0.773549|0.517868|0.030002|0.248967|
|R_Hip|0.749312|0.369135|0.391317|0.013767|
|R_Knee|0.769098|0.457673|0.398698|0.130679|
|R_Ankle|0.797234|0.512791|0.285168|0.238181|
|R_Toe|0.797240|0.518636|0.066182|0.251216|
|Torso|0.708262|0.252133|0.196764|0.015633|
|Spine|0.694076|0.239853|0.141010|0.039136|
|Chest|0.678891|0.249048|0.118763|0.046252|
|Neck|0.706245|0.374476|0.285253|0.070240|
|Head|0.819380|0.618452|0.354287|0.074629|
|L_Thorax|0.656114|0.341450|0.229469|0.062811|
|L_Shoulder|0.872271|0.619960|0.436772|0.072238|
|L_Elbow|1.207442|1.096246|0.833708|0.141562|
|L_Wrist|1.196866|1.049550|0.711427|0.316911|
|L_Hand|1.196853|1.049773|0.002858|0.359079|
|R_Thorax|0.772765|0.369763|0.203860|0.058402|
|R_Shoulder|0.906508|0.568288|0.415912|0.067209|
|R_Elbow|1.030414|0.781913|0.656724|0.131493|
|R_Wrist|1.131145|0.984295|0.649847|0.241671|
|R_Hand|1.131026|0.984233|0.000863|0.272445|

最明显的局部剩余误差在上肢：L/R elbow parent-relative0.833708／0.656724rad，L/R wrist0.711427／0.649847rad；hands本关节0.002858／0.000863rad却仍有世界旋转约1.20／1.13rad，主要继承上游chain的差异。膝关节parent-relative0.474557／0.398698rad；heading-local ankles/toes位置约0.238–0.251m，也仍需检查步态相位／足部落点，但本报告没有尝试寻找最佳时间对齐。

## 4. 按clip拆解

成功列仍采用原 position-only全序列门槛；小根漂移不能保证上肢或旋转正确。Motion ID沿用原AMASS validation bank。

| ID／文件 | 帧数 | 原成功 | 根XY m | 根yaw rad | local位置m | world旋转rad | 去heading旋转rad |
|---|---:|---|---:|---:|---:|---:|---:|
|0 Walking_3|580|是|0.327425|0.422690|0.116961|0.605715|0.479698|
|1 Jog_3|484|否|0.483496|0.650108|0.141926|0.898714|0.640293|
|4 Jog_1|664|否|0.488175|0.631954|0.144060|0.873261|0.619504|
|10 Walking_3|800|是|0.251876|0.314952|0.132960|0.515254|0.459889|
|11 Jog_3|784|否|0.391796|0.469960|0.131743|0.730053|0.611758|
|14 Jog_1|674|否|0.398124|0.553195|0.141683|0.761773|0.592954|
|19 Walking_3|371|是|0.139711|0.263823|0.094426|0.492984|0.397583|
|20 Jog_3|384|否|0.550220|0.844482|0.150038|1.117323|0.645337|
|23 Jog_1|420|是|0.345475|0.306742|0.120230|0.645997|0.546824|
|26 Walking_1|465|是|0.129343|0.317341|0.094206|0.526094|0.401302|
|234 0008_Walking002|727|否|0.331799|0.360084|0.098133|0.592232|0.423847|
|235 0008_Walking001|797|否|0.264678|0.614019|0.123151|0.851944|0.536868|
|242 0005_Jogging001|335|否|0.292775|0.095141|0.145120|0.634263|0.628089|
|244 0005_Walking001|1163|否|0.193697|0.152340|0.117434|0.466223|0.437142|
|245 0005_BackwardsWalk001|300|否|0.415868|1.956698|0.114920|1.976517|0.535942|
|247 0017_RunningOnBench002|110|是|0.128740|0.244967|0.159914|0.788925|0.674278|
|253 0007_Walking001|974|否|0.253224|0.229474|0.109676|0.555929|0.501639|
|254 0018_Walking001|523|否|0.349433|1.022717|0.115285|1.159238|0.543217|
|257 0018_DanceTurns002|457|否|0.150959|1.498855|0.278888|1.754686|1.092367|
|259 0018_DanceTurns001|172|否；fall|0.614765|1.604223|0.463834|1.720350|1.472506|
|260 0018_Catwalk001|343|否|0.376069|0.946112|0.129854|1.182134|0.672199|
|261 0018_Moonwalk001|455|否|0.202754|2.478228|0.155243|2.442262|0.520949|

BackwardsWalk245：world旋转1.976517rad→去heading0.535942rad，rootyaw1.956698rad；Moonwalk261：2.442262→0.520949rad，rootyaw2.478228rad。**这两段**确实以朝向差为突出特征。DanceTurns257／259去heading后仍1.092367／1.472506rad，259唯一发生fall；不能由全体平均推断每一段的误差机制相同。

Jogging242根yaw仅0.095141rad，而去heading误差仍0.628089rad；主要不是yaw。完整reference包括后退、moonwalk、dance／turns、running-on-bench等标签，不能简单把它当作纯前向walking同质测试。

## 5. 按reference速度分组

速度为capture中原 `reference_root_vel` 的XY norm，保留原legacy估计；固定bins `[0,.5,1,1.5,2,3,∞)`，无依据结果调bins。分组有动作类型／fall期混合，不能归因速度造成某种姿态误差。

| reference m/s（半开） | 帧数 | 根XY m | 根yaw rad | local位置m | 去heading旋转rad | actual速度m/s |
|---|---:|---:|---:|---:|---:|---:|
|[0,0.5)|1162|0.323297|1.330030|0.187766|0.760162|0.323264|
|[0.5,1)|5939|0.262243|0.521466|0.120236|0.494945|0.624892|
|[1,1.5)|4491|0.370479|0.586863|0.140663|0.595531|0.760977|
|[1.5,2)|386|0.349737|0.358605|0.144756|0.623552|1.090294|
|[2,3)|4|0.321149|0.074278|0.174634|0.698127|0.711757|
|[3,∞)|0|—|—|—|—|—|

整体reference速度均值0.930120m/s，actual0.661669m/s。多段Jog序列actual明显较慢，既可能影响root drift，也可能和不同步的肢体周期相关；这里是同帧记录对照，没有提出未经验证的原因，也没有用lag搜索改分。2–3m/s仅4帧、≥3m/s没有样本，不足以建立这些速度区间的性能结论。

## 6. 下一阶段使用方式与边界

1. 新controller完成后原样运行同一诊断脚本，先核对同22个reference、全序列step数量、framephase与success合同；新child_joint velocity profile需明确披露，不能隐去协议变化。
2. 优先对比root XY／yaw与上肢、膝／足的local残余，检查变化究竟改善导航还是姿态。对极端转向／后退与普通Jog分别检查，不从一个综合均值推出统一原因。
3. 当前actor只接收future rootXY任务与自身state，没有每帧reference3Dpose输入。其风格模型目标与确定性逐clip模仿不是同一个监督合同；本诊断可以显示偏差，不能假称作者的精确mocap-tracker协议已经复现。
4. 不改现有benchmark、阈值、历史capture／result／checkpoint。原6/22与0.377293／0.859479继续作为primary；本地diagnostic分数不与论文Table直接换算。

## 复现与文件

CPU-only脚本：[decompose_motion_tracking.py](decompose_motion_tracking.py)。完整22clip／24body／speed-bin统计与所有capture SHA：[legacy_tracking_decomposition.json](legacy_tracking_decomposition.json)。脚本含独立刚体全局yaw+translation不变性fixture、primary数组复算、SHA闭包、frame/reset合同、MSE closure gate；Ruff通过，未启动Torch／CUDA／Kit。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/decompose_motion_tracking.py \
  --input output/diffusion_amp/motion_tracking_legacy_full_validation_v1 \
  --output /tmp/legacy_tracking_decomposition.json
```

输出必须是新文件；未来capture可使用同CLI，不覆盖历史证据。
