# AMASS root 课程：八模型 full22 / 固定风格对照 v2

v1七模型全部保留；本次只增加真实step500完整验证。旧七份 evaluator 与 comparison run dictionary 逐值一致，原c274 reference、90D风格公式与速度bins未改变。原始失败/摔倒帧全部保留；step500有26个fall frame，未为风格评分剔除。

|模型|success/22|fall|位置 m|姿态 rad|inclusive MMD²|速度W m/s|
|---|---:|---:|---:|---:|---:|---:|
|legacy1000|6|1|0.377293|0.859479|0.093668|0.257637|
|amass2848_child500|1|0|0.441347|1.027874|0.050227|0.213351|
|amass2848_child1000|5|2|0.430993|1.254071|0.206480|0.092944|
|backward_penalty1000|11|0|0.300708|0.861454|0.046821|0.143767|
|stratified_experts1000|8|2|0.352941|0.791256|0.089115|0.200738|
|stratified_experts500|1|0|0.460473|0.869972|0.055555|0.305356|
|amass_root1000|17|0|0.229298|0.831882|0.056075|0.065651|
|amass_root500|12|1|0.357862|0.875744|0.038601|0.047519|

## step500与1000的真实取舍

step500：12/22、1 fall（SFU Catwalk001）；step1000：17/22、0 fall。500 inclusive MMD² 0.038601 / upright0.039434、速度W0.047519，比1000的0.056075 / 0.065651更接近冻结reference分布；但500位置0.357862m、rootXY漂移0.291173m明显差于1000的0.229298m / 0.142768m。低MMD不能抵消实际跟踪失败和摔倒。

500→1000，同帧yaw误差0.625359→0.551314rad；L/R wrist parent-relative rotation0.643825/0.623558→0.553035/0.513716rad改善；去heading局部pose位置0.125944→0.125401m接近，仍比negative baseline0.119189m略差。1000的跟踪提升主要来自根轨迹，未证明所有关节风格变好。

toe近地body-origin代理：500均速0.624416m/s、超阈比例0.316621；1000为0.696977/0.320159。代理有效区间13044/14321不同，这不是原生contact/COM，也不足以选更自然或防滑的模型。

当前1000是此完整低层跟踪验证较好候选；500保留为半预算checkpoint探索，不以单项MMD追选替代。真正多人Diffusion部署与原生脚部支持仍需独立验证，未作论文8场景成功声明。

## 固定速度bin MMD²

|实际速度 m/s|negative1000|AMASS root500|AMASS root1000|
|---|---:|---:|---:|
|0.0–0.1|0.051339|0.043297|0.088217|
|0.1–0.5|0.087036|0.089308|0.092030|
|0.5–1.0|0.016421|0.009179|0.014914|
|1.0–1.5|0.023122|0.019880|0.034496|
|1.5–2.0|0.056186|0.031341|0.050717|
|2.0–3.0|unsupported|unsupported|unsupported|
|3.0–∞|unsupported|unsupported|unsupported|

≥2m/s生成/参考样本不足，仍unsupported。MMD是速度条件下分布距离，不是逐帧mocap跟踪误差，也不是作者公开阈值；验证集已经反复用于迭代，不能作为未知test显著性或风格收敛证明。

每个模型22段/11982 metric frame、11960同段导数interval。新500与negative的初始physical24body/AMP history与完整reference/clock/target数组22段逐元素相等。500/1000分别4,096,000/8,192,000训练transition，不把不同预算称作同预算比较。negative1000和AMASSroot1000的PPO/D/RMS/reward/warmstart/physics与预算相同；命令、reset合法帧、clip-end分布一起改变。

证据：[evaluation](evaluation.json)、[comparison](comparison.json)、[verification](verification.json)、[step500同帧分解](../amass_root_step500_tracking_decomposition.json)。[七模型历史v1](../amass_root_style_v1/report.md)保持原样。CPU only，无Kit/GPU或业务/权重修改。
