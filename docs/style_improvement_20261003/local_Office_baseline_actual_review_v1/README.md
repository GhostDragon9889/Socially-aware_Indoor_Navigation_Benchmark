# 本地 Office：默认模型的实际 10 秒诊断

Root 的 GPU capture 与独立 CPU 审计都实际闭合 exit 0。审计覆盖默认 `62c4edc1…` controller、当前生产 185 文件、3 actors、300 controls（10 秒）、1200 个 120 Hz shared substeps，以及完整 14,400 个 foot observations。未发生跌倒、time-limit reset、静态接触或同行人接触；三名行人均没有到达首个路线顶点。

足部对已登记的工程参考 floor 的真实 force channel 在左右两侧都观察到非零支撑。三个 actor 的无支撑子步比例分别为 2.67%、0.33%、1.08%；COM 的 mass-weighted 重组最大误差为 `3.81e-6 m`。这些统计保留所有控制帧和失败，足部切向速度仍是 body-origin proxy，不是接触点滑移测量。

保存的重力配置为 `[0,0,-9.81] m/s²`，active USD magnitude 为 `9.81000042`，只读 observer 没有推进物理（counter 0→0）。这证明同一 context 的配置与 active USD 观察一致，未测量 native engine gravity。

[实际审计](Office3_audit.json) 与 [Root CPU 闭合记录](CPU_actual_launch.json) 均已只读封存。原 Warehouse4 reader 的说明字段仍写了 ray 行数 132；[独立元数据更正](metadata_erratum_v1.json) 绑定原报告，说明实际完整保留的行数为 **131 =3 reset origins +128 fixed samples**。原报告和 raw 数学没有改写。

这是本地新场景的短资源/支撑诊断，支持后续扩展评估。它没有验证 200 秒导航稳定性、场景覆盖或论文风格效果，也没有匹配作者八场景评估。
