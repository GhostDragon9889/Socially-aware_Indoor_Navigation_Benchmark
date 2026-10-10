# 配置、资源与接触接口的结构调整

## 已完成

- `application/configuration.py` 提取原 CLI，保留默认值、路径及覆盖保护。profile 通过唯一 `settings.TrainingConfig` 校验，不复制另一份 trainer 配置。
- pure settings拒绝numeric参数中的boolean误用。旧 `learning_rate:true`、`gamma:false`、`log_std:false` 确实被接受；只增加boolean guard，去掉该guard后完整AST与before一致，合法数值/算法不变。[修前](numeric_bool_before.json)、[修后](numeric_bool_after.json)。
- `application/resources.py` 在 Kit 前检查实际选择的文件、AMASS split 身份/尺寸、Git LFS pointer、selected PACER SHA、MJCF XML及scene元数据。CLI seed和visual model override也在此校验。
- `python -m CrowdSim.humanoid.application.preflight` 提供同一 gate 的可独立诊断入口；当前 runner 已由 root 接入，不创建第二份训练入口逻辑。
- `AMASSCorpus` 相对路径改按 manifest 所在目录解析。真实 Torch mmap 回归证明相同尺寸、相同文件名的 cwd decoy 不再被误加载；缺失文件不静默回退。
- contact helper 按真实 enabled CollisionAPI 子prim解析static filter，修复visual Xform返回零接触的问题；修复Sim6删除的stage导入命名空间。
- 每actor一个exact-source native view，严格24sensor×Ffilter×3，保留body-pair法向力的substep最大值。源身体批量归约与逐身体采样等值；缺sensor、错误filter shape、非有限force显式失败，没有自动回退。
- [当前结构与评估索引](../STRUCTURE_AND_EVALUATION_CURRENT_ZH.md)和[完整pipeline入口](../../FULL_PIPELINE.md)增加资源预检、package边界、实际baseline及历史失败说明。

## 验证

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 新frontend/manifest路径/contact CPU回归 | 110 passed，含19个numeric参数×2boolean误用，0skip/0failure；4CPUthreads、无CUDA/Kit | [日志](focused_tests.log)、[JUnit](focused_tests.xml) |
| 当前两个training profile + 一个shared/capture evaluation profile | 三case通过；319/22/34 clips，PACER SHA重新计算；AMASS仅stat/declaredSHA | [当前文件预检](actual_resource_preflight.json) |
| 真正fresh Python导入隔离 | Torch/IsaacLab/IsaacSim/Omni/Warp/pxr均未导入，包括legacyentry和完整preflight | 测试subprocess及上项actual JSON |
| Ruff Python3.12 | 本轮源码/tests通过 | [日志](ruff.log) |
| root实际PhysX ContactV4/V5 | floor zero、wall10492.528N、peer1376.634N双向；cases逐字段完全相同 | [V4](../../output/diffusion_amp/contact_sensor_probe_v4/result.json)、[V5](../../output/diffusion_amp/contact_sensor_probe_v5_batched/result.json) |

V5通过后native actor batching默认开启，四人由96个body view变为4个actor view。最大body-pair法向力定义、阈值及物理采样频率保留；contact buffer容量显式记录，未做contact-rich saturation证明。

## 可核实的限制

预检不反序列化模型。签名、尺寸和selected SHA不能证明所有state_dict、tensor ID或动态控制兼容；真实加载、FK/蒙皮、PhysX、视觉各自仍有gate。根wheel仅发布`protomotions`，CrowdSim源码应用、外部数据、许可证和SDK环境保持独立。

历史三轮四人200秒均16falls、episode affected16/20；当时未记录真实contacts。新的native contact probe没有给这些历史run补造数据。32人50秒bounded直线路径0falls的低层诊断不能作为论文八场景导航结论。当前CPU四地图需真实PhysX复核，作者2,848 clip ID、定制weights、八场景及精确评价协议仍缺。

## 文件清单

业务改动范围：四个新`application`模块、`settings.py`的numeric boolean guard、`amass_corpus.py`路径解析、root初始`contact_monitor.py`的窄修复/批处理、三个新CPU测试文件、`FULL_PIPELINE.md`、新结构索引。`settings.py`/trainer原始迁移由依赖agent、runner/world集成由root完成；当前parser的新增trajectory/terrain flags由root持有。[准确before/current hash](resource_structure_changes.json)。

历史reports、checkpoint、原USD、FLUX/NavDP及SDK环境保留。未运行GPU/Kit或重复hash大型AMASS包。
