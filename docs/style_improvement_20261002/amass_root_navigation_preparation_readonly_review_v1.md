# 四人200秒候选导航预检：独立只读复核

Root preparation 的实际 argv 经当前 pure parser/profile 验证通过；仅config/output/resume变化，以及旧记录缺少的eta键显式为0。checkpoint实际字节SHA `dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e` 与封存1000一致；baseline result SHA一致。

固定协议：4人/6000步/200秒，episode6001、seed7、30Hz控制/120Hz物理、64m floor。Diffusion trace_clean guide20、eta0，timed A* waypoints、causal reference1.5m/s与2m/s²、15步同步replan，one coherent sample。physics-only、不编译、不渲染、deterministic Torch。floor1N、peer/static1N原生normal-force记录与四substep signed power保持。

`--resume`加载控制器；evaluation始终使用validation corpus与独立Diffusion部署命令。AMASS-root profile的training declaration不会在这次trajectory_source=diffusion的evaluation中生成mocap课程。未把training状态/未来3D动作注入导航。

CPU frontend实际未导入Torch/Kit/Warp，未启动SDK。750候选应另绑定真实sealed750 SHA与独立output，其余parsed协议保持；不能把500续训后的750称为原连续1000实验的遗漏快照。

[详细JSON](amass_root_navigation_preparation_readonly_review_v1.json)包含历史baseline各physics/measurement契约，用于真实候选完成后的对照。尚无候选导航结果，此报告不作fall/contact/风格提升声明；未读34测试输出，也未重复计算ground/COM指标。
