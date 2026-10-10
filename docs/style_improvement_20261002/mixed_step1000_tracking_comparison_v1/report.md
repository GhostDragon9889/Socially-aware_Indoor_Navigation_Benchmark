# Mixed step1000 完整22段验证

CPU重算既有真实捕获；完整11982帧和失败帧保留，初始状态与参考/相位/时钟dtype及bitwise配对。

| 主指标 | negative1000 | mixed |
| --- | --- | --- |
| tracking_successes | 11 | 19 |
| sequences_with_fall | 0 | 0 |
| frame_weighted_mean_body_position_error_m | 0.300708008 | 0.203485352 |
| frame_weighted_mean_body_rotation_error_rad | 0.861454446 | 0.87566073 |

mixed累计1000updates，总8192000transitions，每组4096000。baseline1000总8.192M；budget关系已单独记录。
全22为主分母；success-only/common-success子集仅描述性，不代替主指标或未知的论文error分母。
源码135tar逐member核验；预登记绑定实际评估6份runtime源，当前CPU admission的8份源另列，二者计数/用途不同。
检查点字节seal绑定训练前登记source/argv，独立完整payload门禁与本reader的nestedstate/三RNG、real151/proc51、episode/两组实际budget核查通过。
此前34test已在上一阶段开封；本次不读取它，也不宣称新的blind test。
本报告不读取34test、不改变部署默认、权重或业务源码，不把reference tracking称200秒导航成功。
