# 评估入口的可选足部支撑采集

## 已完成的接入

已有完整评估命令追加以下选项即可，不改变 controller 或 planner 的权重：

```bash
--measure-ground-support --ground-support-threshold-n 1
```

仅允许正的 `--evaluate-steps`、真实 `--resume` 和 `--power-source projected_joint_effort_substep_mean`。threshold 必须有限且 > 0；显式非默认 threshold 必须与采集开关一起使用。开关默认 False；没有新增默认模型、物理或奖励变更。

world 初始化前，contact reporting 显式取 `pairwise_contacts OR measure_ground_support`。初始化后从当前 USD 资产解析四足的实际 rigid-body 相对路径，构造每人的完整路径；当前 USD stage 的 `/World/ground` 解析为 enabled CollisionAPI 后代。`FootCaptureAdapter.from_world` 使用 Lab 已初始化的注册 physics view，不创建独立 SimulationView 或新增推进循环。

原 static/peer monitor 作为 composite 的成员，由现有四子步 hook 每次只 sample 一次。原 `physical_states.npz` 的字段和含义不变：若同时打开 pairwise contacts，只把原四个 contact 字段放入原 capture；足部、COM 和计数全部存入独立文件。

## 真实控制阶段

1. `prepare_control_step(control_step)` 在原 `world.step` 之前，使用 runner 的实际全局计数。
2. 原 world.step 的四个 120Hz `physics + scene.update` 后读取 floor force、live link state、COM、mass 和公开 Lab counter。
3. world.step 返回后立即 `measurements()`，在 reset 和渲染之前发布；`control_step` 为 pre-step 值，`control_step_after=control_step+1`，实际物理计数必须连续四步。
4. 原 reset 仍按原流程执行；只有 reset ID 数量 > 0 才显式重置采集 episode 边界，且旧控制已经发布。

捕获只读，不写姿势/动作、不补充任何物理步，不把 submitted-step counter 当成 PhysX engine 时间。足部传感器、正常接触与 COM 来源说明见 [foot_capture.md](foot_capture.md)。

## 独立工件与失败证据

`ground_support.npz` 包含：

- 四子步原始 floor-only normal force `[T,4,B,4,G,3]`、support/load/contact 系列和接触条件 body-origin 速度代理。
- 真正 live body COM、local COM pose、body masses、mass-weighted world COM，以及用于同瞬时验证的 common link 状态。
- 原始 `physics_step_before/physics_step/physics_step_end`、pre/post 控制计数、episode IDs、before-frame reset mask 和实际 terminated/truncated。
- 每个控制 `valid=True`，无 object 数组；共同 body/foot 名称单独保存。AMASS 参考 COM 不虚构，合同中为 null。

JSON 仅记录合同、来源、shape、完整控制数/子步数、bytes 和 SHA，不把完整力数组塞入结果。新工件存在时 preflight 拒绝；写入同目录临时文件并用原子 NOREPLACE 发布，已有文件不会被覆盖。

若控制内失败，已完成的合法前缀可存为 `ground_support.npz`，当前未完成数据另存 `ground_support_partial.npz`：`valid=False`、明确原因和实际可用 state/force 数量，缺少的子步不会补零。若失败发生在没有活动采集的阶段，结果明确声明无可用 partial；不会伪造数组或声称完整捕获。

## 当前验证范围

219 项 focused CPU 测试通过，Ruff 通过；源码已冻结，源 SHA / before 快照见 [验证 JSON](ground_support_integration_review.json)。主要证据：

- 关闭新开关后，去除仅属于关闭分支的新节点，完整 runner AST 与保存的 before 逐节点相同；旧 training / evaluation / pairwise CLI 的每个旧属性完全相等。
- 真实当前 runner AST 的调用段执行 CPU 假物理，验证 prepare → 唯一 world.step → 四次单次 peer sample → publish → 控制计数递增；发布先于 reset。
- 当前 pairwise 分支实际 AST 只投影原四字段，足部 COM 不泄漏到旧 `physical_states` schema。
- Archive 使用实际 FootCaptureAdapter 的四子步 CPU 数据，独立验证 native/common 顺序、非零旋转 COM、episode 边界、before/post 计数、部分数据无 padding、非有限/错 shape/错计数拒绝和无覆盖发布。
- 新 frontend 与 runner 的纯 import/parse 子进程没有导入 Torch、Kit、IsaacLab 或 USD。

这是源码/CPU 兼容性证据，不能宣称真实 GPU bit-equivalence。root 另行运行原生 pilot 和完整评估。本模块不把 neutral stance 真实倒地转换成通过：三 phase sensor/filter 与 COM QA 可单列有证据的结果，整体 `native_physical_probe_passed` / `native_capture_probe_passed` 仍保持 False，直到相应实际验证明确支持。

## 修改范围

本次既有文件仅修改 `CrowdSim/humanoid/application/configuration.py` 和 `scripts/train_paper_amp.py`；新增 `CrowdSim/humanoid/ground_support_archive.py`、`tests/test_amp_ground_support_cli.py`。没有修改 amp_world、trainer、奖励/参考模型、冻结风格公式或真实已有工件，也没有在本任务启动 CUDA/Kit。
