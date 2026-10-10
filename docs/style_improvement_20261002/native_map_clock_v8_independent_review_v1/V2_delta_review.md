# v8 proposed patch v2：只读差异复核

**未发现具体阻塞；真实 v8 SDK 执行仍 pending。** 本次没有修改 candidate/生产、执行 merge 或尝试绕过此前审批拒绝。原 v1 的 68 项证据与归档保留。

独立核对 [v2 合并清单](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_map_clock_v8_proposal_v2/merge_manifest.json)、[具体 patch](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_map_clock_v8_proposal_v2/proposed.patch) 和全部前后字节：**10 条固定目标、296,656 B**。相较 v1 仅增加 `tests/test_paper_scene_maps_probe.py`，且只允许既有测试的已登记 before SHA。单文件 250,000 B、总计 500,000 B 的硬上限、禁止链接/路径穿越、保护原 v7 文件、默认不写入等措施保持不变；merge helper 的 AST 除固定 allowlist 增加该路径外完全相同。

## 差异及实际作用

1. `tests/test_paper_scene_maps_probe.py` 仅修改 `test_actual_runtime_sets_zero_timeline_again_after_SDK_reset_to_authored_negative_start`。其余文件 AST 完全相同。现在用 standalone 真实 USD 匿名 stage/session 代替缺少原生 stage 接口的 `SimpleNamespace`，在执行从真实 `run_native_scene` 提取的 setup 片段时注入**实际 helper**，解决 NameError，并校验根层 RAM 与 session 目标。Timeline 和 Sim 仍为明示的替身；`Sim.reset` 故意回退到 −1/60 后，实际调用片段必须再次请求 0。没有将其写成 Kit 实跑。
2. `tests/test_native_map_v8.py` 仅将两处测试工作区解析改为稳定的 `CrowdSim.__file__` 工作区，避免 source-loaded candidate 的 `file_record` 函数位置改变路径归属。独立 AST 归一化确认全部几何、crop、CSR、0 秒时钟、v5/v7 精确输出与损坏证据拒绝断言保留，未删除检查。
3. producer、provider、grid、`native_map_v8.py`、历史 v7 reader 与其余候选测试的 payload SHA 均与 v1 一致。producer 仍是 `20173101ada3e5137d962abcdea574497e9624be7659984caa9869e7932d537a`，因此 v1 的真实四 USD 与 session/所有权/新旧 dispatch 审查适用于相同业务字节。

[独立结果](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261002/native_map_clock_v8_independent_review_v1/V2_delta_review.json) 记录全部检查及 exact patch/manifest/helper SHA。作者 staged focused 日志为 **216 passed /10 skipped**；本次只核对其日志及具体差异，没有重跑广泛测试，也没有把这些 CPU/合成结果当作四场景原生 PASS。

V2 不改变原 USD 或训练配置，不升级 SDK、GPU 或环境。实际查询必须继续保留 source/session/range/root-RAM/真实零秒时钟门禁。Warehouse 正在运行的 source168/V4 与原 v7 失败历史仍按各自 prelaunch 归档归属；后续应用应由 root 按具体 patch 审批并登记新的 epoch。
