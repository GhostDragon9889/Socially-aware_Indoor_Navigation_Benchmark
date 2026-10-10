# 最终集成独立复核（2026-10-01）

## 结论

在限定的最终集成范围内，未发现仍未处理的业务源码阻断。复核发现并交由root修复了两个具体问题：普通stop/start同帧录制丢失，以及无LFS CI误加载配置指针。当前代码已包含两项修复；本agent冻结业务源码，仅写此审查记录。

本次是只读源码/实际归档复核，加两个最小CPU复现及一项针对性回归。未启动Kit、安装、修改业务源码、提交或暂存；未重新跑宽回归。全suite、最终产物重建与freshness gate由root/dependencies agent完成，最终结论以 `docs/PROJECT_COMPLETION_20261001_ZH.md` 为准。

## 发现并完成的集成修复

### IR01 / P2：普通停止后立即重启录制丢弃前一会话

- 触发：第一次start→render得到1帧；在两次render间执行toggle stop、toggle start。
- 修复前真实 `RecordingMixin` CPU生命周期复现：新session启动，但旧 `.motion` 不存在，旧 `0000.png` 留在旧目录，内存motion被新session覆盖。此前cancel/start测试无法覆盖普通stop/start。
- Root已在 `protomotions/simulator/base_simulator/record.py:371` 新会话替换buffer前finalize旧会话并cleanup；encoder失败直接抛出，sidecar导出未完成则明确RuntimeError，保留旧folder/buffers，阻止静默覆盖。
- 新回归 `protomotions/tests/test_recording_lifecycle.py::test_stop_then_immediate_start_publishes_previous_session` 独立运行 **1 passed / 0.84s**：旧motion有1帧、旧mp4存在、旧临时目录清理、新会话继续记录1帧。视频编码器在此生命周期fixture中替代，motion构建与保存为实际实现；真实25Hz编码另有root的clock测试。

### IR02 / P1（CI gate）：lfs:false下读取存在的LFS指针

- `CrowdSim/tests/test_pedestrian_runtime.py` 真实本地checkpoint配置测试原先仅判断 `last.ckpt.is_file()`。
- `git ls-files` / `git check-attr` 确认checkpoint与 `resolved_configs_inference.pt` 都是tracked LFS文件。CI指针checkout中两者都“存在”，不能按存在性识别真实数据。
- 使用git HEAD中的实际pointer、只在临时目录构造checkout形态，调用真实 `configure_from_checkpoint`：两文件is_file均True，得到 `UnpicklingError: invalid load key, 'v'.`。本地已有payload的全suite不能捕获此问题。
- Root已在当前测试添加 `@pytest.mark.needs_lfs`，同时对sidecar不存在或未解析pointer显式skip。无LFS三目录gate按marker排除；full job获取真实 `.pt`，即使排除权重 `.ckpt` 仍能运行配置检查，因为该测试不加载权重本体。

## 已核查的最终边界

| 范围 | 结果与依据 |
|---|---|
| Record时钟与标记 | 当前record的video、motion、marker/object均来自control dt；native marker转换采用dataclass replace，不修改控制器原state；异常时先保存physical motion，clip.close在finally。Root现有 `root_lifecycle_mapping_tests.log` 记录22项通过，含真实25Hz 25帧→1秒MP4及motion fps一致。后补普通stop/start独立回归见IR01。 |
| Spawn地图坐标 | `CrowdSim/scene_setup.py` 的ROS左下角origin、cell center、yaw、negate/alpha与nav地图合同一致；平方清距不把地图外区域当自由空间，资源不足明确抛错；builder传入origin_yaw/negate/free_threshold。`test_scene_spawn_contract.py` 包含旋转中心、negate、边缘余量、间距、透明和不足空间用例。本次读取代码及root已有结果，未重复这些测试。 |
| Nav日志析构 | `_close_trajectory_logs` 对两句柄使用getattr；部分构造对象无timestamp属性不再AttributeError；每个真实句柄flush后finally close，重复关闭跳过closed句柄。Root修复了上一轮fixture暴露的构造失败析构warning。 |
| Python3.12 CI | lint/CPU/full使用Python3.12；Ruff target py312；CPU安装 `[mujoco,dev,crowd]`；`PYTHONPATH=src:. pytest --import-mode=importlib protomotions/tests/ CrowdSim/tests/ tests/` 覆盖三目录，普通gate `-m "not needs_lfs"`；IR02已补资源marker。此复核没有执行GitHub Actions远端runner。 |
| 主包wheel/sdist | 读取实际 `/tmp/protomotions-completion-packaging` 预构建归档：main wheel477成员包含checkpoint helper，SMPL mesh/mjcf/usd资源均未分发；main sdist包含CrowdSim入口/配置、src独立包、LICENSE、PointGoal配置、环境lock，未发现必需清单缺项，无src/build副本。CrowdSim为源码工作流，不宣称主wheel安装即含场景/权重。 |
| 独立crowdsim-ppo | 实际预构建wheel包含8个Python模块、PointGoal JSON、两CLI入口；依赖只有numpy/torch与可选pytest，无protomotions/IsaacLab硬依赖；打包默认JSON与源码默认等值。pipeline基于 `__file__` 定位源码或wheel配置，与CWD无关。 |

## 产物冻结要求

独立复核预构建归档时，发现独立wheel中的 `crowdsim_ppo/ppo_policy.py` 早于最后buffer参数guard；主wheel也早于record/checkpoint及backend后续改动，共10个Python成员与当时最新工作树不一致。这是预构建产物的新鲜度问题，不能把其早先测试结果当最终发布字节的验证。

已通知root；root已确认在源码冻结后重建全部wheel/sdist，由dependencies agent的新验证器逐成员核对最终业务源码。此报告不替最终freshness gate背书，也不重新构建或安装产物。主报告应记录重建后验证结果。

## 参考仓库源码完整性

- 参考 `/project/EXP/NavIsaaclab2.0/CrowdSim` 的70个 `.py`/`.yaml` 文件当前均存在。
- 参考内嵌protomotions的164个源码文件中，当前缺少旧 `agents/masked_mimic/{__init__,agent,config,model,utils}.py`。当前已安装SMPL checkpoint文本配置实际target为 `protomotions.agents.supervised.agent.SupervisedAgent` 与 `protomotions.agents.supervised.masked_mimic_model.MaskedMimicModel`，对应源码均存在；当前MaskedMimic control、motion manager、obs链也存在。
- 因此这5个旧路径差异不是当前已验证行人模型的必需缺失。此结论不保证任意旧NavIsaaclab2.0 pickle checkpoint可直接恢复；旧模型导入路径和权重schema仍需要与该模型逐项匹配。

## 验证边界

- 本agent仅独立执行IR01修复前CPU复现、IR02真实pointer最小复现与IR01修复后单项回归；其余项为源码/归档检查并读取root已有证据。
- 未验证远端CI执行、GPU训练收敛、任意旧checkpoint迁移、真实viewport异步捕获时序或关闭窗口时的active recording导出。没有据此新增“已验证”声明。
- 最终全suite、真实SMPL/physics/media、资源清单及最终打包新鲜度由主报告汇总。
