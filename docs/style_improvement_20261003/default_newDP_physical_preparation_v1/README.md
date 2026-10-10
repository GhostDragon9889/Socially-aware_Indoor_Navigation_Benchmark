# 固定 default62c4 + OLD/NEW Diffusion 的物理配对登记

状态：**prepared only**。这里未启动模型、GPU 或 SDK，未消费未来 capture。生产185、旧训练/源/默认及2848/val22/test34均不改；不读sealed34输出。

两条当前185源的fixture顺序为OLD63e → NEWb563，各4人、6000控制帧、200秒。固定旧state-only default62c4与原5433B配置，唯一模型变量是Diffusion checkpoint；独立输出目录只是实验身份。原 `simulate_paper_pedestrians.py` 拥有物理执行，guide20、alpha0、seed7、timed shared scene、15步同步重规划、原globalplan/routes、causal_accel 1.5m/s及2m/s²、四子步/四feet、PD与physics/contact参数完整保留。实际SDK-free两个preflight解析参数仅 `diffusion_checkpoint` / `output` 不同，原数学未复制或编辑。

62c4是历史1000updates/8192000transitions controller，其完整训练provenance必须逐项等于历史native4实际result。它不是09e，也不声称在935训练；原checkpoint文件模式保持（实际观测0664），每次实际owner对opaque全bytes/SHA前后验证。新b563来自935源的fresh5000训练，严格绑定真实closed training、5000 root seal及0444/nlink1独立copy。当前物理执行则始终是完整185/9ddb来源；两者来源分开记录。

root使用 `commands.json` 的OLD、NEW GPU owner数组；实际退出后使用对应CPU raw owner数组。OLD即使有跌倒/接触，也完整保留且不跳过NEW。每条真实owner记录Popen退出、原错误及cleanup错误、源/模型前后、真实log和capture；CPU原始数学只有 `verify_sources` 被独立真实准入替代，替代发生在读任何NPZ之前。保留所有跌倒、reset、接触和idle样本。

只有NEW完整fixture6000/200s的 falls=0、static actor-controls=0、peer unordered-pair controls=0，才继续native4。native4也同185源OLD→NEW；NEWnative4完整物理零故障才继续同源native15的OLD→NEW。native阶段目前只预登记V5场景、未来路径/模型及顺序，不生成额外future命令或假结果。物理准入完全不消费MMD或风格分数。默认不promotion。

历史selected fixture及native168/174结果另作为参考，不能当成当前185的严格配对。绑定的 `native185_physical_source_derivation_v1/source_derivation.json` 逐527归档成员证实：physics/PD/AMPworld、contact、Diffusion planner、EvaluationActor、deployment等源逐文件原bytes相同；runner除两个仅非evaluation reward-resume guard外原bytes相同（本6000evaluation分支不执行guard）。168→174的scene map/height/render admission确有变化，trainer/checkpoint/runtime admission也变；没有whole-source/RNG/physics bitwise或actor输出相等声明。本轮严格比较由同当前185的OLD/NEW实际运行保证。

CPU准备仅实际源/配置/SDK-free入口及必要6项负例，不重跑旧suite，不反序列化模型。准备报告的 `future_effect=false` 不表示未来运行通过。第一次实际物理故障和失败报告均保留。
