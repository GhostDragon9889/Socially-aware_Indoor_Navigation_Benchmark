# 显式 AMP＋Diffusion 行人入口

`scripts/simulate_paper_pedestrians.py` 复用现有 `train_paper_amp.py` 的评估分支。它没有自己的控制循环，不创建第二个 App/world，不调用旧 SFM/MaskedMimic，也没有后端回落。

当前仅支持已验证的四人 cuboid 共享物理 fixture。AMP、Diffusion 检查点、训练配置与 AMASS manifest 必须显式提供。入口不会自动选择仍在训练的混合模型，不默认加载 `last.pt`。下面使用当前保留的 negative1000 导航基线；公开 PACER 初始化与自训 Diffusion/AMP 属于声明的方法重建，不能称为作者未公开模型的精确复现。

## 预检

在仓库目录执行：

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/simulate_paper_pedestrians.py \
  --amp-checkpoint output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt \
  --config configs/diffusion_amp/amp_style_amass_backward_penalty.yaml \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --scene-config configs/diffusion_amp/shared_obstacle_timed_validation.yaml \
  --actors 4 --duration-seconds 30 --seed 7 \
  --output output/diffusion_amp/paper_pedestrians_baseline_30s_v1 \
  --preflight-only
```

此模式调用相同的 `parse_amp_args`、配置/训练来源校验和本地资源预检，输出完整转发 argv。实际本地资源 CPU 预检已通过；没有导入 Torch、NumPy、Kit/SDK，没有创建仿真输出目录。它不反序列化模型，不证明模型兼容性或导航质量。

## 物理运行

移除 `--preflight-only` 即调用既有评估器一次。输出目录必须全新；目录中会先保存 `deployment_entry.json`，记录入口、完整 argv、资源预检与计划预算，随后现有评估器保存实际结果。若运行失败，保留失败输出并另选新目录重跑。

入口的固定合同：

- 30Hz 控制 / 120Hz 物理；时长必须对应整数控制步。30 秒为 900 步、3600 actor transitions。`episode_steps=steps+1`，避免把录制末帧当 time-limit 成功。
- 物理运行、无渲染；严格 Torch deterministic evaluation，BLAS/CPU threads 为 4。不承诺 PhysX/Warp bitwise 一致性。已导入 SDK 的进程会被拒绝，必须使用独立 Python 进程。
- guide20、timed A* 路点、`trace_clean`、eta0；peer weight5/radius0.6，scene SDF weight10/radius0.8；每15步同步重规划。
- 因果 XY reference adapter：1.5m/s、2m/s²、初始超速显式 brake；不修改真实物理速度或瞬移身体。
- 原 filtered 静态/同伴接触与独立四个物理子步 feet-floor/live COM 通道均开启，阈值1N；原测量和控制逻辑保持在评估器内。
- terrain64/spawn6；固定共享场景提供实际 origins/routes/cuboids。其他 scene 文件只接受同一 fixture 的相同字节副本；目前不宣称支持 DynBench AMP 部署。

`--updates`、`--rollout-steps`、`--capture`、`--resume`、训练课程或指导权重参数不属于此入口，直接拒绝。57个当前解析字段与真实旧基线的56个字段逐项对照，只增加明确的 eta0；内部原有训练 bounds 1000/32/10 是评估器兼容字段，评估分支不进行训练。

CPU 回归覆盖完整参数对照、时长/输出门禁、拒绝无效参数/场景、SDK-free 导入，以及正常/异常时只调用一次现有 driver 并复原 `sys.argv`。注入 driver 的测试仅证明调用合同；真实新入口的30秒物理验证由独立 GPU 任务执行，未通过前不标为 SDK 验证完成。
