# Social simulator 必填配置修订

**源码准备，仍 mutable，待 root 完整审阅、独立审阅和封存。** 未 import 候选、NN、SDK、数据或权重，未执行物理 gate。旧8actor执行的 AssertionError 和实际退出139保持为失败；这次修订不能证明退出139已经解决。

## 唯一源码变化

三份完整父源码各仅一个构造 fragment 增加 `experiment_name="diffusion_amp_reconstruction"`，与已成功的 `CrowdSim/humanoid/amp_world.py:138–139` 相同。其余完整字节经字面逆替换恢复；仅含构造调用的函数 AST 增加此 keyword，删除后等于父函数 AST。所有其他函数及全部 PINS 保持原字节。

| 新入口 | 父 SHA 前缀 | 新 bytes / SHA 前缀 |
|---|---|---|
| [smoke_group_physics.py](smoke_group_physics.py) | `d000200a` | 21284 / `2b8bc1c3` |
| [probe_shared_world.py](probe_shared_world.py) | `87600495` | 32114 / `bb0f862b` |
| [train_shared_social.py](train_shared_social.py) | `f12a5004` | 27326 / `cc48fdd8` |

完整父/新路径、bytes、SHA、精确替换 fragment、逆还原与各函数证明见 [source_derivation.json](source_derivation.json)。原 sealed 文件和生产185不改；num_envs仍为2或groups，120Hz/4子步、solver4/4、offsets、floor、collision/contact、D6/PD、reset、PPO/AMP、cleanup、通过条件、分母和所有配置JSON/权重均不改。

## 真实执行来源与旧 helper

新入口按正常文件路径执行，保留原 `__file__` 记录逻辑，不能把父文件声称为唯一执行源。smoke的 `execution_source`、probe的 `execution_entry`、training的 `ENTRY_IMPORT_BYTES`/`entry_sources`自然记录此目录的真实新入口。

- 新probe仍校核 `PINS["eight_actor_smoke"]` 的原d000源码；该pin没有执行或import旧smoke，不充当旧gate通过证据。修订版8actor gate须有单独的实际root执行receipt、closed0与原物理检查结果。
- 新training的 `PINS["probe_setup"]` 保留原876004。`main` 与 `_run` 确实从旧路径加载未改的output/source/asset/helper函数，报告应如实保留该旧helper来源。它不调用旧probe的 `_run`；本新training自己的 `_runner_setup`执行修订后构造。
- `ROOT = Path(__file__).resolve().parents[3]`层级保持正确；`HERE`为本目录。未复制配置JSON，因此training的默认 `HERE/smoke.json`不存在。未来命令必须显式加 `--config docs/style_improvement_20261004/shared_social_training_entry_staging_v1/smoke.json`，pilot则显式使用同旧目录的 `training.json`。配置的原封存身份及数值不变。

## 尚未完成的接线

原c31四人评价与be9 native评价目前pin原f12 training entry。真实新training入口为`cc48fdd8…`，旧gate不能接纳此来源；后续必须另做明确的评价metadata source revision并实际审阅。不得把新training报告伪装成原f12唯一执行，不绕过旧gate。

最终 stdlib compile/完整字面逆还原/AST检查工具`65b477` exit0；Ruff工具`b4a081` exit0、All checks passed。仅源码检查，不是物理或训练通过。root安排实际执行前应先封存，实际原8actor失败的记录与输出保持不覆盖。

## Root审阅与源码封存

Root README实读c30cd7、完整字面逆还原／AST及三个构造调用实读3e0f6c、完整derivation实读85030d；独立完整diff／AST67ad20、全逆恢复与SHA5a4c2d、derivation实读c6e665，Required=[]／Optional=[]。五文件设0444、普通独立nlink1。此为源码封存，尚无新版SDK实际结果；原139原因仍未证明。
