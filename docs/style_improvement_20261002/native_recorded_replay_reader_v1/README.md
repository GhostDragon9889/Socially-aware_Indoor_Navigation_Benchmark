# Native recorded replay：CPU 后处理 reader

[audit.py](audit.py) 是独立 docs-only 工具，不启动 Kit、CUDA、策略、优化器或物理。读取实际终态 result 与源 captures；成功和失败均记录。已完成外部与室内两次实际450帧成功路径审核；编码与来源合规、视觉可用性分别记录。

两个历史失败已单独封存：[首次回放失败](failed_replay30_v1.json)、[SDK Render 诊断失败](failed_placeholder_diag_v1.json)。均保留实际 error、0 rendered frames、physics0/policy0、源 result/NPZ 与原 USD53层逐 SHA；它们仍是失败。历史 renderer 源码与当前代码不同会记录，不能把 currenttree 当原运行源码。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python \
docs/style_improvement_20261002/native_recorded_replay_reader_v1/audit.py \
--run output/diffusion_amp/native_warehouse_recorded_replay_30s_v2 \
--output docs/style_improvement_20261002/native_recorded_replay_reader_v1/actual_replay30_v2.json
```

输出独占创建，不覆盖旧审核。成功路径必须实物满足450 PNG、15FPS/30秒完整MP4 decode、900控制帧stride2、源53层不变、Root弱sublayer1/session0、visualfloor无collision及timeline不推进。每PNG重新按生产 `stamp_recorded_frame` 核对**永久banner像素和post-control源时间**；episode/reset只存在frame metadata，工具不假装它们写进图像文字。所有450编码帧top60 crop实际decode，检查lossy banner像素meanabs<=15且原PNG白字>=90%仍亮，明确非OCR/非有损秒数digit逐位识别。源物理结果是历史 capture证据，不是回放的新零跌倒或接触验收。

8项CPU测试与Ruff通过：两个真实失败身份/源闭包、假bool零计数、缺失或错时banner、尺寸错误、post-step时间及变动文件拒绝。实际成功路径还执行完整450帧审核；CPU专项测试本身不代表视频交付。

## 当前实际交付状态

- [室内30秒回放摘要](actual_indoor_replay30_v1_summary.md)与[机器结果](actual_indoor_replay30_v1_summary.json)：450帧/15FPS/30秒，视频SHA `20965b4f74417f44c4a6a7596620712f7d3ed2c0dba4501dd5f6d50310241c71`，2253940字节。renderer绑定174文件V7来源（manifest `f9c007ec503236790766e768967a8b745b0dcde1a4a96dfd0b58986763ce403f`），历史物理姿态仍来自已保存的source158 capture。首/中/末帧0/225/449可见两名行人，属于有限抽帧的可用室内视图；未证明四人始终可见、动作自然程度或论文等效。
- [外部30秒回放摘要](actual_replay30_v2_summary.md)与[机器结果](actual_replay30_v2_summary.json)：450帧编码、来源、banner门禁通过，但首/中/末帧屋顶遮挡全部行人，**视觉交付失败**。renderer来源为171文件V6，不能替换成当前V7。
- [首次回放失败](failed_replay30_v1.json)与[SDK Render诊断失败](failed_placeholder_diag_v1.json)仍保留，未重标成功。

两次成功回放均physics0/policy0/timeline推进0，原53个USD层SHA不变。回放不是新的控制器物理试验，不继承新的零跌倒或导航质量结论；未提升默认控制器。编码banner检查不是OCR，episode/reset绑定来自实际metadata。
