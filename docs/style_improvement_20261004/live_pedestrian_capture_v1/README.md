# 实时行人 RGB／深度采集

本轮补齐了对外采集入口，复用现有 Diffusion＋AMP 驱动的实时 SMPL 蒙皮、物理世界和控制循环。默认每两个控制步采集一次，视频为15 FPS；物理仍为120 Hz、控制为30 Hz。

## 已验证与待验证

- 真实 SDK-free 预检已退出0：[资源记录](final_actual_SDK_free_preflight.json)。确认默认 AMP62、Diffusion63、固定2848 AMASS清单、neutral SMPL、UV和纹理文件；预检没有创建模拟输出目录。
- 33项CPU接口测试通过，覆盖导航参数保持、资源环境一致性、单次委派、异常恢复、同render product深度接线、无效深度保留及采集文件审计。测试中的合成数据仅验证接口。
- 新源码通过 Ruff。当前训练绑定的185份源码逐一核对未变。
- **真实 RGB-D 渲染、帧同步、相机标定及新增入口的物理状态不变性尚未验证**。GPU正在执行正式40,000次Diffusion训练，采集验证安排在其退出后串行运行。新增接口不是动作风格或论文效果改善的证据。

此前判断需要另建实时 skin adapter 的报告已由[当前源码审阅](../../style_improvement_20261003/smpl_live_capture_source_review_v1/README.md)补正：原 `train_paper_amp.py --capture` 已创建带纹理SMPL，更新实际物理状态后、reset前捕获。原公开部署入口只提供纯物理评价；本轮新增独立采集入口，原驱动继续拥有唯一App。

## 运行

从仓库根目录，在当前训练退出后运行；每次使用新的输出目录。

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/capture_paper_pedestrians.py \
  --depth --inference-only \
  --amp-checkpoint output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt \
  --config configs/diffusion_amp/amp_style_amass_backward_penalty.yaml \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --duration-seconds 10 --capture-every-steps 2 \
  --output output/diffusion_amp/live_RGBD_NEW_RUN
```

加入 `--preflight-only` 即为资源预检。省略 `--depth` 时输出RGB。首次真实运行先用 `--duration-seconds 1` 验证渲染及状态诊断，再运行完整采集。

原生Warehouse需要同时指定场景和世界坐标相机：

```text
--native-dynbench
--scene-config configs/diffusion_amp/native_warehouse_pedestrians_clearance08_v1.yaml
--camera-eye -6.75 -8.95 3.2 --camera-target 0.35 6.35 1.0
```

这组相机位置已有独立录像观察依据；新实时RGB-D入口的取景仍需实际观察。新增相机覆盖只影响采集相机，行人的控制、路径和碰撞测量由原驱动执行。

## 输出与审计

| 文件 | 内容 |
| --- | --- |
| `deployment_entry.json` | 请求、资源预检、实际委派参数；执行验证状态保持false |
| `frames/frame_XXXXXX.png` | 原驱动的RGB图像，XXXXXX为零起始控制步索引 |
| `pedestrian_motion.mp4` | RGB视频，FPS为30／采集间隔 |
| `depth/frame_XXXXXX.npy` | 可选float32原始光轴深度，单位米；0和正无穷背景保留 |
| `depth/index.jsonl` | 深度文件与控制步对应、有效比例和范围；未知同步状态保持false |
| `result.json` | 原驱动执行结果、每帧前后物理状态／步数／RNG诊断 |
| `physical_states.npz` 等 | 原有物理与参考轨迹捕获 |

实际模拟进程退出后，用新的报告文件审计：

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/audit_paper_pedestrian_capture.py \
  output/diffusion_amp/live_RGBD_NEW_RUN \
  --output output/diffusion_amp/live_RGBD_NEW_RUN/capture_audit.json
```

审计检查完整文件集合、采集控制步、原始深度、视频分辨率／帧数／帧率，以及每帧物理缓冲、步数和RNG不变性。原生PhysX时钟API若不可用，保留null。文件审计不提供工具执行出处、不证明传感器帧同步、相机标定、取景质量或论文等价效果；真实启动／退出记录另行保存。SDK快速退出可能跳过外层Python清理，只有正常返回路径尝试detach和恢复临时helper。

[实际验证与源码身份](verification.json)记录工具退出、33项接口测试及185份源码核对；[独立审阅](../../style_improvement_20261003/smpl_live_capture_source_review_v1/CAPTURE_ENTRY_REVIEW.md)确认已报问题闭合。

## 方法依据

[NavIsaacLab论文](https://arxiv.org/html/2606.26265v1)的行人方法为轨迹Diffusion与物理AMP控制；RGB／深度服务于视觉反馈与后续导航观测。本次深度接口用于完整呈现和检查模拟输出。[Isaac Sim6相机文档](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_camera.html)列出RGB和`distance_to_image_plane`；深度按相机光轴定义，不等同于到相机中心的欧氏距离。新增helper复用已运行的Replicator接口，在同一render product增加深度annotator，沿用原有有限次数渲染更新。
