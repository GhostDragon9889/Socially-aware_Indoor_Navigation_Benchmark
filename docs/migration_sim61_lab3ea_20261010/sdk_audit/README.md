# Lab3.0EA / Sim6.1 SDK与可选后端审计

这是本轮精简取证包，项目源码未由本次打包修改，SDK源码始终未由审查/迁移任务修改。未提交或推送。默认环境见 [6.1 profile](../../../env/isaaclab3_ea_sim61/profile.json)、[安装指南](../../guides/installation.zh-CN.md) 和 [克隆复现](../../../CLONE_REPRODUCTION.md)。机器SDK绝对路径是审查对象身份，不是克隆用户必须使用的目录。

## 报告与矩阵

- [SDK版本、依赖/API迁移断点](SDK_MIGRATION_AUDIT_ZH.md) / [精简安装元数据](machine_inventory_compact.json)。
- [全Python静态范围与可选基线边界](ALL_CODE_AND_VENDOR_AUDIT_ZH.md) / [阶段收据](final_audit_receipt.json)。
- [Newton1.5.2迁移](NEWTON_MIGRATION_ZH.md)。
- [NavDP接口/XYZW/ProxyArray矩阵与资源缺口](NAVDP_API_MIGRATION_ZH.md) / [69条导入矩阵](navdp_namespace_matrix.json)。
- [native退出真实对象定位与修复验收](NATIVE_EXIT_DIAGNOSTIC_ZH.md) / [N1实际退出收据](native_n1_exit_acceptance.json)。
- [headless IOMMU模态提示设置](HEADLESS_MODAL_SETTING_ZH.md)。
- [仍需协调的当前文档指引](DOCUMENTATION_DEFAULT_REVIEW_ZH.md)。
- [来源与复制清单](SOURCE_INDEX.md)。

## 证据分别解释

| 证据 | 实际结果 | 能证明的范围 |
|---|---|---|
| 最终本子任务静态快照 |1514个Python文件/331368行，无AST失败；关键undefined规则清零|src/scripts/third_party/examples的当时源码；不等于所有分支运行或无bug保证|
| NavDP/vendor组件CPU |[36 passed](vendor_navdp_quaternion_cpu.xml)|选定真实方法/接口边界；未启动NavDP Dingo/RTX完整任务|
| Newton CPU |[35 passed,2 skipped](newton_all_cpu.xml)|新Newton实际模型/solver初始化与状态/接触核；CUDA跳过保持明确|
| 旧Newton CPU |[13 passed](newton_legacy_cpu.xml)|真实1.4dev；1.0回退未做真实安装运行|
| 实际目标SDK CPU |[通过](navdp_live_sdk_cpu.json)|真实ProxyArray、Tensor math、core WXYZ到Lab XYZW边界；未加载Kit|
| parent统一CUDA |[5 passed](gpu_five_actual.xml)|列出的Newton真实CUDA、FK、contact平滑和随机化索引用例；不证明所有模型/后端|
| 真实SecondFloor N1 |[parent exit0](native_n1_exit_acceptance.json)，60 controls/0falls，26borrow aliases释放，view弱引用在App.close前死亡|约2秒仿真/65.82秒墙钟；不是60秒四场景、N4、长训练或RTX通过证明|

29passed和combined64passed2skipped等先前CPU收据保留为阶段证据，不能相加当作独立最终测试总数。root/功能审查负责全仓最终collection与最终执行收据，本包不将4289 current/354 historical节点数误写为全部执行通过。RTX第二轮在本包快照时由root继续验证，其最终结果应引用独立录像收据。

## 明确未完成项

NavDP完整7入口仍缺casadi/pathfinding（teleop另需pynput与X/uinput）、匹配checkpoint、实际评估场景USD及sidecars、HTTP模型服务和真实交互/RTX回归；旧requirements不能覆盖主6.1runtime。可安装性与独立profile建议在NavDP报告中，未创建/安装该profile。FLUX完整旧SDK任务、PACER原IsaacGym训练以及Trace原始冻结环境仍单独隔离，选定CPU修复不构成整套基线迁移认证。Newton不支持SMPL exp_map、任意室内USD场景或非平坦terrain，现明确拒绝。

历史SIGSEGV与失败GDB记录保留：PhysxManager Timeline类缓存和Articulation/Data强借用SimulationView是分别定位的对象；新N1成功只覆盖其实际路径。所有外部大文件均保留原位置及SHA，未把视频、数组、大源码快照或巨日志复制进仓库。
