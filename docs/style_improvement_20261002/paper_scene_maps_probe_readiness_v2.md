# 场景原生地图探针：v2 CPU 就绪说明

35 项 SDK-free 测试通过，Ruff通过；没有启动 Kit／GPU。完整身份：[v2 就绪记录](paper_scene_maps_probe_readiness_v2.json)。

本次仅修正元数据：`reference_flat_terrain` 模式创建有限静态地面时，`additional_Terrain_or_ground_created` 如实为 true；未选择该模式时为 false。回退这一处表达式后整个源码 AST 与 v1 相等。保留原8a940源码副本、v1就绪记录及v6四场景预检；实际执行会重新预检并绑定 v2 哈希，不将旧预检冒充新源码验证。

当前Lab3 `open_stage` 返回纯 USD stage，但 `SimulationContext.__init__` 明确选择 threadlocal stage、加入StageCache并attach到Kit，默认 `create_stage_in_memory=false`。`get_physics_step_count()` 与 `clear_instance()` 签名存在；前者只记录主动提交的步数，不是原生测量时钟。

安装omap接口与探针调用对应：四位置参数 settings、相对origin的volume bounds、返回world cell centers／world bounds。探针通过0.1m尺寸与0.2mm浮点容差、cell-center往返、occupied/free/raw计数核对重建 north-up，未盲reshape原生buffer。SDK允许实际地图范围小于requested bounds；本探针对此保留raw并保守拒绝资源准入。原生grid约定、actual stage attachment／PhysX cooking、全部freecell地板支持仍待root执行验证。

首次新增元数据测试因将AST `values` 列表误当函数失败，修正仅测试读取后35项通过；v8失败日志保留，v9为最终CPU证据。
