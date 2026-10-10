# 官方Maps：最小独立可用转换源

root已读取[转换代码](convert_official_maps.py)，按[commands.json](commands.json)的`convert_argv`实际完成全1000场景转换（进程45405，exit0）；[完整可用QC](../orca_maps_official_usable_QC_v1/README.md)随后真实通过。源、commands和准备闭合记录保留原prepared-only身份；此stage不修改原始文件、localMaps接口、旧Interact／mixed stages或生产185。

- 直接复用已实际通过的Interact `resample_raw`和`interaction_windows`函数对象：真实initPos/initVelocity作为t0，以实际trackTime插值到0…9.9s的100点；不外推、不将原首帧冒充t0。heading仍为速度方向代理。
- 1000个scene和Random0分割全部保留，episode历史31／未来52。采样窗口仅要求当前≤15m的真实邻人；zero-window场景仍写manifest，不要求看到障碍、不按几何唯一性或碰撞结果筛选。
- 各scene保存真实polygon、原map.pkl身份和2×1080×1080的global raster，`map_present=True`／`map_condition_mask=True`，包括45个合法空地图。无地图Interact的mask=False语义保持。
- 栅格复用冻结q935中的现有ORCACorpus Shapely occupancy函数，12px/m、90m；它与官方`cv2.fillPoly`的边界像素约定不能声明逐像素一致。crop由现有函数处理，外部unknown仍0.5；没有安装／改变CPU依赖环境。
- 新schema `navisaaclab.orca_official_maps_adapter.v1`、`condition_mode=official_maps`、max_neighbors19（真实source≤10行人，为mixed tensor边界统一padding）。scene／episode加`Maps_`前缀，防止与Interact同名scene混淆。

每scene `map={path:map.npz,bytes,sha256}`（数组key `raster`）、`source_map_pkl`、`source_raw`、`obstacles`；episode数组除原Interact四项外增加原`recorded_times`和真实`radii`。实际输出为`data/diffusion_amp/orca_maps_official_usable_1000_v1`。source-only `--prepare-only`原实际exit0见[准备闭合记录](actual_prepare_CPU_closure.json)，完整转换另见[实际owner](../orca_maps_official_usable_actual_runs_v1/conversion_actual_launch.json)。没有构造不存在的训练／质量结果。
