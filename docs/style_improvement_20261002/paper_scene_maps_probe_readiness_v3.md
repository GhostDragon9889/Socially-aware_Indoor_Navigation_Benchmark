# 场景地图探针：原生数据驱动的 v3 修正

38项CPU测试与Ruff通过；其中真实Warehouse失败pilot原始数据逐字节SHA绑定并回归。原生失败结果和原始89025个API free点仍完整保留；本报告只证明修正后的CPU布局合同，尚未重新执行Kit。

安装SDK `utils.compute_coordinates` 明确native top-left在(maxX−cell/2,minY+cell/2)，头文件规定row-major x-fastest。因此采用固定x倒序／y正序布局；不按数据最佳拟合选择轴序。实际3343 occupied点全部精确对应buffer4；56239个buffer5全部存在world点；API额外32786个点的buffer标签6保持unknown；未列出的502个cell全部为6。另7种轴序／翻转组合均不满足全部occupied与knownfree双向核验，诊断不作为回退逻辑。

实际world-center最大误差1.15e−6m；尺寸251×370，raw4/5/6计数3343/56239/33288。canonical free只取56239个经过raw5/world点核验的API行；完整API列表另保留，标签／API行／bufferindex及漏列unknown索引另存档。任何free-list4、occupied≠4、缺失knownfree、重复／离网格点均拒绝。

源身份、测试、失败pilot与完整机器证明：[v3就绪记录](paper_scene_maps_probe_readiness_v3.json)、[八种布局实际诊断](native_warehouse_omap_layout_actual_v2.json)。原生返回2D z中心.75及z范围.7..8不能独立证明.15..1.8全高度几何采样；完整高度box-overlap与floor/support/source准入仍保留，等待root新原生pilot。limited查询不能获得fullgrid准入，navigation_ready始终false。
