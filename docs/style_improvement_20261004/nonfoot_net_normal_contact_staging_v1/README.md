# 排除足部的净法向接触判据：CPU接口已检查

`nonfoot_contact.py`只实现一个当前帧张量判据，还没有SDK采样器或实际终止接线。输入必须是实际postphysics、包含所有partner（含floor）的body净**法向**力[N,24,3]，不能用历史peer body-pair control最大力替代。

按实际body名称排除R/L_Ankle和R/L_Toe，对其余body先做世界坐标向量和，再求范数，严格>50N且progress>1。这保留公开PACER的判据数学；本机IsaacLab原生getter是法向力通道，Gym原tensor的摩擦包含语义尚未确认，因此是明确本地normal-only适配，不能声称力通道完全等价。

返回resultant、magnitude、threshold／progress条件和contact_flag的新张量。不修改输入、RNG、clock或physics，不处理enableEarlyTermination、disableCollision、轨迹偏差、fall、timeout或全组reset；未来runner还要组合实际条件并保留真正终止原因。

四项有界CPU测试实际61ce0b直接退出0，unittest耗时.025秒／工具wall.902411409秒：名称重排／足部排除、向量抵消与50N严格边界、progress／输出所有权／RNG，以及错误输入。CUDA前后未初始化；没有NN、权重或SDK测试。Ruff5d57bc、AST993fb9及独立源码审阅Required=[]通过。固定源名字只以AST literal读取，避免原observation模块的额外导入。root核对精确字节后封存source、tests及[实际检查记录](actual_CPU_checks.json)；真实SDK采样与终止仍未接入。

出处：[固定PACER终止代码](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/tasks/humanoid_pedestrain_terrain.py#L1388)、[同提交排除body配置](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/data/cfg/pacer.yaml#L48)。实际SDK数据语义以本机4ecd0b源码为准，较新[官方contact sensor文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/concepts/sensors/contact_sensor.html)已用显式normal命名；不把新属性假定为本机已有接口。
