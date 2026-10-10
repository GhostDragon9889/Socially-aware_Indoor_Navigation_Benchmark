# Social simulator 显式资源释放修订

**源码准备，mutable，待 root／独立审阅和封存。** 未 import 候选、NN、SDK、数据或权重，未运行物理、训练或测试suite。配置v1已消除必填字段断言，实际8actor物理检查得到64提交子步、同组正contact1355.844N、跨组raw绝对max0与分离全0，但进程仍退出139（root `20529/cc9ddc → dbd6e1`）。这些原来源和输出保持原字节；report的passed不能替代实际closed0。

## 最小释放变化

三入口完整复制封存的config v1，只在原finally的 `sim.stop()` try/except之后、`app.close()`之前加入独立的clear try/except：

1. 确认 `type(sim).instance() is sim`，拒绝释放别人的singleton。
2. 调用 `sim.clear_instance()`。
3. 确认 `type(sim).instance() is None`，然后才将 `simulation_clear_instance_returned=True`。
4. clear异常按各入口原有Exception／BaseException边界加入cleanup errors，仍继续尝试app.close。

三个report的新增标记初态均为False；smoke主入口通过条件及probe／training的cleanup success均要求此标记。物理检查、样本分母、PPO／AMP、metrics、配置JSON、weights、全部PINS／旧helper导入路径与其余时序不改。`fast_shutdown=False`保持，不使用os._exit、不掩盖139。

installed主要依据为 `/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/sim/simulation_context.py`：772–778的stop停止manager／visualizers，不detach资源；813–845的clear先关闭physics manager，再visualizers／services／stage，最后singleton=None和gc。services错误可在singleton已None后仍抛出，故调用与after检查全部成功才记True。root原源码阅读`7f6474`；本次只读取源码，完整文件及方法SHA记录在 [source_derivation.json](source_derivation.json)。这解释需要补齐的释放步骤，**尚不能证明139消失**。

## 真实来源与未来执行

| 新入口 | config v1父SHA前缀 | 新 bytes / SHA前缀 |
|---|---|---|
| [smoke_group_physics.py](smoke_group_physics.py) | `2b8bc1c3` | 21939 / `fd44f44d` |
| [probe_shared_world.py](probe_shared_world.py) | `bb0f862b` | 32803 / `0cc582c7` |
| [train_shared_social.py](train_shared_social.py) | `cc48fdd8` | 28007 / `f61c6c3a` |

正常文件执行保留真实新 `__file__`／execution_source／execution_entry／ENTRY_IMPORT_BYTES记录；原父文件不能冒充唯一执行入口。新probe仍只校核旧d000 smoke pin；training仍真实复用旧876 output/source/helpers，不调用旧probe的_run。新8gate须另有实际rootreceipt与closed0。

training必须显式 `--config docs/style_improvement_20261004/shared_social_training_entry_staging_v1/smoke.json`，pilot改为同旧目录的 `training.json`；新目录不复制配置，原parse_args默认HERE/smoke.json不存在。原c31／be9仍绑定原f12 entry，后续必须明确修订评价metadata以接纳真实新training SHA，不能绕过旧gate或伪称原f12唯一执行。

## 源码验证

完整父／新SHA、每份3个精确字面fragment、全文件逆还原／AST与未改函数名单均记录于derivation JSON。仅_run与main发生上述cleanup／flag变化；全部PINS及其余函数保持字节和AST。

stdlib compile／逆还原／AST工具`0d8bdd` exit0；Ruff `c1b4c7` exit0、All checks passed。源码通过不等于资源释放或真实退出通过；下一实际进程须保留全部原物理记录、cleanup错误及真实exit code。

## Root审阅和封存

Root完整README与新增finally／flags实读c00360、整文件逆还原／AST与每个delta实读7d9f0c通过；独立全源逆恢复／AST／SHA5a4591、installed释放方法421216完成，Required=[]／Optional=[]。五文件设0444、普通独立nlink1。未运行新SDK、NN或训练，下一实际8gate结果另记；139消失仍未证明。
