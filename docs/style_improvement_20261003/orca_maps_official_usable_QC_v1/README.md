# 官方Maps：完整可用数据QC已实际完成

root实际转换进程45405已观察exit0（completion e60c2a）；本QC进程36128实际exit0（completion b0bd5b），耗时8.5955秒。全1000场景／2001文件SHA、原始scene与分割、状态和地图检查完成；见[完整报告](actual_QC.json)、[真实CPU闭合](actual_CPU_closure.json)和[实际转换owner](../orca_maps_official_usable_actual_runs_v1/conversion_actual_launch.json)。未重跑转换、旧raw QC或NN。

| split | scene | 合法neighbor窗口 | zero-window scene（保留） | 空地图（presence=True） |
| --- | ---: | ---: | ---: | ---: |
| train | 800 | 78149 | 91 | 41 |
| validation | 100 | 10211 | 10 | 3 |
| test | 100 | 9129 | 13 | 1 |

5640名行人、564000个10Hz agent-frames。全部真实t0初值、半径0.4m、原始trackTime副本、0…9.9s均匀时间点、有限五维状态及速度heading代理符合接口；history31／future52窗口完整，最大真实邻人数9，19slot为与Interact统一的padding。原时轴插值数学复用原Interact函数对象，本轮没有重复其旧证明或声称重新独立验证60Hz状态。

全部1000个真实map文件为2×1080×1080 uint8互补二值层，空地图的free全1／obstacle全0。每个nonempty地图均在固定全局grid及原polygon顶点邻近的实际1291608个像素探针上核对geometry；不是完整全像素重新栅格化，也不声明与官方cv2边界像素一致。原始114次圆盘／25次障碍重叠诊断保留，未成为删scene／window的过滤条件。

独立[三split实际NumPy接口记录](../orca_official_mixed_training_staging_v2/actual_numpy_interface.json)已由paper agent执行：每split各取empty／nonempty Maps，共6个窗口，map mask均True；另复用同一个已加载union验证每split两域B2，共6个窗口，Maps／Interact mask为True／False。没有构造NN或重复加载全部corpus。本QC报告冻结在其原数据检查范围内，未向里面追加后续接口结果。

首attempt因raw polygon点tuple在JSON转为list而在reader的结构比较处失败；[原源码／日志／真实exit1](first_attempt_error.json)保留。仅将该比较改为canonical JSON primitive序列，数值、转换代码及数据均未改变。随后正式全量检查真实exit0。

这确认资源可读取、全部样本得到保留和张量接口匹配；训练尚未开始，不证明作者数据身份、物理安全或达到／超过论文。源与结果均不涉及GPU、SDK、权重或封存34条测试输出。
