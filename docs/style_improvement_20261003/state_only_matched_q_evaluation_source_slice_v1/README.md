# State-only matched q：生产185评价ABI最小确认

仅源码／metadata／原parser准备。真实CPU source-only `fb0044` exit0；没有导入Torch、加载模型、启动SDK／GPU或读取未来terminal。生产185、936 bundle、默认和旧冻结文件都未改变。未创建新执行器、owner、框架或测试套件。

## 现有loader可直接处理1950

- `TrainingConfig`支持state／`warm_start_discriminator=True`／action-extension=False及history-coordinate RMS；生产的`load_amp_profile`和`validate_reconstruction_config`实际解析该936绝对profile通过。
- `PaperAMPNetwork`按saved config构造：D width=`10×195+(state_action?69:0)`；state为1950，actor仍357＋task1044→69，57活动／12冻结。
- Full22入口`benchmark_paper_motion_tracking.load_evaluation_actor`调用原`PaperAMPTrainer.load(...restore_rng=False)`，saved config决定D及四RMS形状；完整model和moments严格加载，不重开public权重、不再次warm-start。随后只调用actor.policy，D不参与tracking动作。
- Fixture的`EvaluationActor → ReconstructionAMPInference`也采用saved config，先严格检查全部1950D／AMP RMS，再仅保留policy及self/task moments；AMP/value moments验证后释放。`prepare_legacy_evaluation_rng`使用同一saved state config构造临时CPU网络，不能用旧2019 config强行制造相同随机消耗。无需profile或loader数学适配。
- 真实pilot4完整CPU报告已PASS2087项，D1950、public D/history RMS转移true、4updates32768trans、AMP count249152；这只证明已有pilot payload，不代表未来2000终態准入或效果。

## 来源与原22动作数学

生产185 manifest9ddb及所读11来源逐记录匹配。Full22 benchmark、trainer、motion_tracking整文件与q935和state936相同；AMPWorld的step/reset/state/observations/planner_inputs等方法AST相同。185 AMPWorld整体文件不同：增加native-provider分支及`__init__/height_at/ground_height_at/physics_manifest`；当前full22不传native provider，选择原PaperPowerSimulator和terrain分支。没有声称全源、初始化RNG或PhysX输出bitwise等价。

[commands.json](commands.json)固定原full22 argv；仅checkpoint/output身份变化，实际封存SHA在运行前追加。split validation、完整22、min1s、失败阈值.5、seed7、4threads、无motion-id/max-clip/max-step上限保持。原完成11982metric frames的22-ID/逐clip分母记录在[CPU证明](actual_CPU_source_ABI.json)，所有失败完整保留。

Root在真实full2000退出／readonly snapshot封存及原background-v2完整terminal CPU审计通过之后，可直接运行：

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python scripts/benchmark_paper_motion_tracking.py \
  --checkpoint output/diffusion_amp/amp_style_state_only_matched_q_isolated_snapshots_v1/step_2000.pt \
  --checkpoint-sha256 ACTUAL_SEALED_TERMINAL_SHA \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --split validation --minimum-evaluation-seconds 1 --failure-threshold-m .5 \
  --maximum-terrain-extent-m 256 --deterministic-evaluation --seed 7 --cpu-threads 4 \
  --output output/diffusion_amp/state_only_matched_q_tracking2000_v1
```

完整绝对数组与CUDA0/CUBLAS4096:8线程环境见commands。这里的SHA未知，不可用pilot、mutable last或名称推断替代。原q frozen executor硬绑定2019／935／q terminal，不能拿来准入本state1950／936；此次直接复用原benchmark，root实际owner需如实记生产185来源与新terminal CPU报告。

Fixture原`simulate_paper_pedestrians.py --inference-only`无需state1950适配；如未来由root批准执行，选该checkpoint实际SHA与匹配state936绝对profile，保持旧63e/guide20/peer5/scene10/alpha0/S1/replan15/完整6000。此slice不登记或启动future fixture/native。

首次probe选错profile父目录、随后过宽whole-AMPWorld相等断言均失败关闭并保存在first_source_only_probe_failure.json；修正仅CPU metadata证明，未修改任何评价数学或重跑旧suite。未来物理及论文效果仍须真实训练/评价；未晋升默认。
