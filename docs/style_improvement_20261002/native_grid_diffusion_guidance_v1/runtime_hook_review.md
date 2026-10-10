# 原生部署接线只读审阅

CPU 源码审阅；未启动 Kit、GPU 或物理仿真，未修改 root/同事业务文件。完整读时 SHA 见 [runtime_hook_review.json](runtime_hook_review.json)。

单 App/单 PhysicsContext 的注入顺序正确。源 USD 保持世界米坐标，原 PhysicsScene 失活，生成 Terrain 从物理 SceneCfg 移除；真实 provider 地面、静态碰撞路径分别传入支撑/接触测量。

已反馈并修复地形特征边界：两个人初始距有限地面边缘仅1.25m，PACER 头部地形窗口为旋转±2m方格。当前 provider 对特征域外返回保守+1.8m；真实 root 支撑查询继续严格要求实际地面命中，不扩地面、不裁坐标。main 也已补原生 callback 的早期 clean/正权重门禁。

仍待 owner 更新的报告字段：原生 grid 的 mean-T 成本不能标为旧 AABB sum-T 成本；物理地面材质应归属实际 provider；state-only 权重不可无条件标为 action-conditioned；原生渲染应按场景 bounds 构图。薄入口当前 physics-only，镜头项不阻碍首轮物理探针。

这些检查不证明新的 USD 烹饪、稳定行走、碰撞回避或论文八场景结果。真实单实例验证由 root 执行。

新增首轮 GPU 风险：严格确定性开启时，安装的 Torch 文档明确把 CUDA `grid_sample` 反向列为抛错操作；原生 SDF 回调恰需坐标反向。已建议 owner 用只读标量表四角 gather/双线性权重，保留严格模式；CPU 要验证值/坐标 Jacobian，真实 GPU 错误/解决由 root 唯一实例确认。此处没有 GPU 执行证据。

最终跟进：owner 已修报告的 mean-T grid 单位、实际 provider 地面来源与 D 类型；四角只读插值已替换 CUDA grid_sample 反向，FP32/64 CPU 值/Jacobian 对照通过（同事门禁）。原生镜头建议仍只影响可选渲染。原始10文件读时快照保留；JSON 另列最终跟进 SHA。没有执行新 GPU 物理验证。
