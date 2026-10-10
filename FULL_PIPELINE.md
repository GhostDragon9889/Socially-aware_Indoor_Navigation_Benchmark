> 2026-10-10：项目已重命名为 **Socially-aware_Indoor_Navigation_Benchmark**，核心源码迁至 `src/`。
> 入口与目录说明见 [项目结构](docs/PROJECT_STRUCTURE_ZH.md)；历史路径通过兼容链接保留。

# Full Isaac Lab PPO pipeline

## 2026-10-10：速度与路线优化、四段新60秒视频已完成

本轮AMP真实训练4,587,520次交互，选用655,360次交互的小学习率候选用于YOHO/NewMetro，窄场景保留已验证AMASS v9父权重。四场景4/4/1/1人各完整60秒，11项物理门槛、姿态、巡航速度及10秒空间停留检查通过；所有30Hz实际速度样本不超过1.5m/s，巡航均值0.721–0.864m/s。换向和避障减速保留。四段新1800帧/30fps视频完整解码与全帧FK核验通过。

新权重、配置、逐场覆盖范围、证据及复现命令见 [本轮完整报告](/project/EXP/SangforAndSaut/ProtoMotions/docs/FOUR_SCENE_SPEED_20261010_ZH.md) 和 [发布索引](/project/EXP/SangforAndSaut/ProtoMotions/docs/four_scene_speed_release_index_20261010.json)。本轮训练有1次探索跌倒，交付物理源零跌倒/零重置/零>1N接触。当前没有覆盖全部场景，也未加入机器人或证明达到作者整体效果。以下历史记录保留原时间与结论。


## 2026-10-09：四场景 60 秒步态优化及录像已完成

固定2848真实AMASS新增768更新/3,145,728物理转移，训练零跌倒。四场景各完整60秒，11项物理门槛和独立姿态筛查通过；四段1800帧/30fps/60秒原生录像完成。成对步态与独立测试结果、实际人数组4/4/1/1及剩余差异见 [本轮报告](/project/EXP/SangforAndSaut/ProtoMotions/docs/FOUR_SCENE_AMASS_GAIT_20261009_ZH.md)。这是本地工程复现，未证明作者八场景总体效果。以下为保留的历史结果。

## 2026-10-09：YOHO 姿态与速度优化已完成

固定2848真实AMASS新增128更新/524,288物理转移。四人YOHO完整200秒、零跌倒、无初始化后重置、实际退出0；11项本地物理门槛和行走姿态筛查全部通过，起步与全程无手臂过水平动作。30秒30fps真实物理状态录像已完成；全900帧FK误差小于1mm。

成对60秒AMP比较中，行走侧倾p95由6.795°降到4.862°，扭转由8.004°降到7.443°；摆臂幅度接近，风格MMD与支撑脚原点速度代理退步。Diffusion公共网络权重保持，推理引导与有界轨迹优化有调整。当前只是单场景/四人/seed7工程复现。

详见 [YOHO结果与复运行](/project/EXP/SangforAndSaut/ProtoMotions/docs/YOHO_DIFFUSION_AMP_ZH.md) 和 [来源证据](/project/EXP/SangforAndSaut/ProtoMotions/docs/yoho_diffusion_amp_evidence_index.json)。以下为保留的历史结果。


## 2026-10-08：真实 Warehouse 新一轮 Diffusion+AMP 优化

已完成固定2848 AMASS / 512次微调 / 2,097,152实际物理样本；四人原始Warehouse完整200秒，零跌倒、零>1N静态和行人接触，54个路线节点，实际退出0。30秒30fps室内+跟随视频已完成。姿态稳定性和部分姿态指标改善；按速度条件的动作MMD与覆盖仍退步。

详见 [Warehouse 当前结果](docs/WAREHOUSE_DIFFUSION_AMP_ZH.md) 与 `docs/warehouse_diffusion_amp_evidence_index.json`。本次一场景工程复现不等于作者完整八场景协议。此前AMASS anchor512和旧Social AMP结果保留为历史对照。


Use Isaac Sim 6.0.0.1, the pinned Isaac Lab 3.0 development checkout, and the supplied SMPL assets. Run from this source checkout. The installable ProtoMotions core and the CrowdSim pedestrian application have separate packaging boundaries.

## Local fixed-AMASS PACER fine tuning (2026-10-08)

The actual AMP fine tune is complete:128 isolated environments,512 PPO+AMP updates,
2,097,152 real physical transitions, all2,848 training IDs sampled. Actor, critic,
task encoder and discriminator weights changed; all public normalization moments
remain frozen and the complete exported policy passes an independent CPU audit.

Select `configs/diffusion_amp/trace_pace_amass_finetuned.json` explicitly. Four-person
reciprocal200s and circulation60s both exit0 normally and pass ten independent local
physical gates, with zero falls or measured pedestrian/static contacts and no post-initialization resets.
Videos render the admitted actual physical archive through IsaacSim6 RTX; no new
physics is claimed by the recorder. See [commands, exact artifacts and results](docs/AMASS_FINETUNE_ZH.md).

Held-out pose/speed style MMD and fixed-pose coverage regress versus the public AMP
reciprocal baseline. No style improvement, universal realism, realtime execution or
complete private NavIsaacLab eight-scene reproduction is claimed. The original
public profile and all failed native exits/recordings retain their original evidence.

## TRACE + PACER reference implementation (2026-10-08)

The current reference implementation for the requested Trace and Pace method is
[`scripts/run_trace_pace.py`](scripts/run_trace_pace.py), using the public TRACE
DDPM and public PACER AMP checkpoints with actual PhysX body feedback. See the
[Chinese method, commands and acceptance report](docs/TRACE_PACE_REPRODUCTION_ZH.md)
and the [current training terminal state](docs/SHARED_SOCIAL_AMP_CURRENT_ZH.md).

Public PACER observation samples now follow the author's0..4.5s clock.75 focused CPU
checks and Ruff pass. Native-clock single-person12s exits0 (route error.216m,
stopped speed.0355m/s); co-directional200s exits0 with24 arrivals and no falls/contacts.
The scoped keep-right200s v2 also exits0 normally and passes all10 independent local
checks:24 arrivals, no falls or measured contacts,99.96%+ supported substeps,1.76Hz
alternating support onset median; all control objects die before SDK shutdown.
Earlier negative native exits and failed/stagnated crossings remain preserved.

A later route audit found that global keep-right costs created40m detours for12m
reciprocal queries. The current preference is restricted to narrow cross-sections
(`right_hand_corridor_width_m:4.0`), reducing those routes to12.62–13.24m while
keeping opposite lanes apart. The reciprocal60s run exits0 and passes10 local gates with12 arrivals, no falls/contacts and8 passing proxies; the200s run also exits0 normally and passes all10 independent local gates:48 arrivals, no falls or measured contacts,25 passing proxies,99.875%+ supported substeps;
the older circulation evidence does not certify sustained opposing corridor traffic.
Latest planning takes1.594s per batch and is not verified realtime. The full run includes49/402 explicitly marked analytical-braking plans; maximum actual pelvis speed2.030m/s exceeds the1m/s planning cap. Fixed-seed GPU physical trajectories are not bitwise repeatable, and brief backward motion remains recorded. Public weights
have not been fine-tuned on the local fixed2,848 AMASS clips. No exact private Nav
weights/eight-scene reproduction, universal robustness or human realism certification
is claimed. Listed older training/benchmark rows retain their original record times.

The files are grouped under existing `diffusion`, `amp`, `runtime`, `analysis`,
`configs/diffusion_amp`, and `scripts` boundaries. Recorded source identities are
in [`docs/trace_pace_source_index.json`](docs/trace_pace_source_index.json).

## Current pedestrian results

| Component | Current result | Selection |
| --- | --- | --- |
| AMASS style data | 2,848 genuine, nonoverlapping segments from 1,853 trials | Fixed v7; the authors' private IDs are unavailable |
| Diffusion | Trained five-state social trajectory model, step 5000 | Fixed deployment model; ORCA-generated interactions are an engineering dataset |
| Navigation AMP | Backward-penalty step 1000; four people for 200 s, zero falls, 21 arrivals | Selected deployment baseline |
| Mixed AMASS/procedural AMP | Step 2000 tracking 19/22, zero falls, position 0.200 m; navigation 11 falls | Tracking improves; navigation candidate rejected |
| State-and-action AMP | Fresh 2000 updates / 16.384M transitions; full tracking 17/22, zero falls, position 0.208 m / rotation 0.726 rad | Navigation: 2 falls, 2 arrivals, 74% idle; rejected |
| Opt-in state-prior state/action AMP | Fresh2000 / 16.384M; tracking17/22; fixture200s0 falls/contacts,40 arrivals | Same-speed style and fixed pose coverage regress; default unchanged |
| New state-prior native Warehouse | 4/15-person200s:0 falls/contacts,44/160 arrivals,8/32 cycles | Common-bin pose distances improve; route progress and15-personK50 coverage decline;30s video verified;default unchanged |
| Native Warehouse | Actual 15-person 200 s: zero falls/contacts, 211 arrivals, 49 route cycles | 1,325 independent raw checks pass; single scene / seed |
| Native map resources | Actual v9 resource gates pass for Office / Warehouse / Full Warehouse / Hospital | 4,441 independent raw checks; 547,321 native cells; original USD disk/RAM unchanged |
| Shared Social CNN AMP S1 / Pop-Art | Fresh fixed-AMASS2848 / 2000 updates; tracking0/22, seven clips fall; N4/200s five falls, 37 peer pair-controls, three route vertices | Quality failed; 90.44% double support and 149–176s longest support runs; defaults unchanged |
| Shared Social CNN AMP S2 / full PPO epochs | Fresh256 / 2000 complete; every training/evaluation/audit process exits0, strict admission passes | [Quality failed](docs/SHARED_SOCIAL_FULL_EPOCH_RESULTS_ZH.md): tracking0/22; coverage improves but navigation falls5→9 and route vertices3→1; defaults unchanged |
| Complete S1/S2 tracking root/body diagnosis | Actual CPU exit0; each22 clips/11982 metric frames, all failures retained, original world metrics exactly reproduced;22 tests and Ruff pass | [Readonly diagnosis](docs/style_improvement_20261006/shared_social_tracking_root_body_CPU_v1/README.md): S2 rootXY .853973m/root rotation1.252676rad; root-relative .218184m and own-root-frame .132644m are diagnostics only. Original .918030m and0/22 success remain; no paper-score substitution or causal training claim |
| Full anatomical-axis and native-reset diagnosis | Actual CPU exit0; all2848/523272 source frames and each S1/S2 validation22/11982 retained;34 checks and Ruff pass | [CPU diagnosis](docs/style_improvement_20261006/shared_social_heading_geometry_CPU_v1/README.md): hip lateral-to-local+Y mean1.109473degrees; no observed fixed90degree or left/right axis reversal. Native reset world errors remain microscopic; original0/22 scores unchanged. Control-learning cause and quality remain unproven |
| Terminal actor trajectory response | Actual CPU exit0; matched real S2 validation11982/training8192, two terminal actors/six conditions/242088 forward rows;27 checks and Ruff pass | [Policy response](docs/style_improvement_20261006/shared_social_trajectory_response_CPU_v1/README.md): change only20 trajectory coordinates; nonzero actor response without active action clipping. Correct gait/steering and closed-loop quality remain unproven; no metric/default changes |
| Procedural reference anchoring and episode candidate | Actual CPU exit0; all2848 frame9 roots, two declared kinematic motions, three mechanisms and2563200 reward rows;49 checks and Ruff pass | [Actual probe](docs/style_improvement_20261006/shared_social_procedural_anchoring_CPU_v1/README.md): stationary periodic reward .964443 versus same-path episode .234614; not policy/physics causation. New reset-only reference passes all2848/455680 target queries; World CPU admission is prepared in the next row; SDK/training remain unverified |
| Episode reference mixed curriculum / World entry | Actual private-entry CPU exit0 with2848 clips and8/256 actors;66 tests and Ruff pass; original AMASS branch and4096 expert source windows per cohort exact | [CPU runtime preparation](docs/style_improvement_20261006/shared_social_episode_reference_runtime_preparation_v1/README.md): reset-only path ownership, persistent episode clock and episode-plus5s reference; unchangedAMASS31, future81 separate.31 artifacts sealed and cold checked,41 sources/4 inputs/23 S3 guards preserved. Actual SDK smokes, fresh2000 training and quality evaluation pending whole-S3 completion |
| Complete S1/S2 PRE command consistency | Actual CPU exit0; each119820 future samples, original PRE phase/clock/root/episode/raw20 prefix verified;38 tests and Ruff pass | [Readonly audit](docs/style_improvement_20261006/shared_social_tracking_commands_CPU_v1/README.md):14.9040% endpoint holds; offline reference root formula .296733/.343848 is not training reward. No captured phase defect found; episode/source horizon observation follows below |
| Actual S1/S2 episode/source horizon | Actual CPU exit0; completed group episodes average1.699144/1.933537s after final-age censoring; original2848 bank IDs/counts/timing verified;32 tests pass | [Readonly diagnosis](docs/style_improvement_20261006/shared_social_episode_horizon_CPU_v1/README.md): canonical50-knot beyond-source fractions48.6179%/49.6418%; validation complete-sequence mean18.154545s. No duration histogram, reset-source ledger or causal conclusion |
| Opt-in minimum future81 curriculum | Independent type/schema;37 CPU tests pass; actual256-actor CPU bank probe exit0, same legal binding native plans/ten targets and4096 expert source windows exact | [Readonly CPU preparation](docs/style_improvement_20261006/shared_social_minimum_future81_CPU_v1/README.md): all2848 retained, minimum2.7s real reset future. New explicit World/entry CPU wiring is prepared below; SDK/training/evaluation remain. No performance or full-training RNG equivalence claim |
| Minimum future81 World/entry wiring | Actual private-entry CPU probe exit0 with2848 clips/256 actors/40 sources;23 tests and Ruff pass; three original profiles remain byte-identical | [CPU runtime preparation](docs/style_improvement_20261006/shared_social_minimum_future81_runtime_preparation_v1/README.md): four reversible World admission changes preserve original physics bytes; SDK/training blocked by pending whole-S3 completion. Planned8/256-actor smokes and fresh2000 candidate have not run; quality unmeasured |
| Episode candidate terminal / independent evaluation entry | 15 CPU tests and11 subtests, Ruff and source preflight pass; actual completed S2 metadata passes original exit/audit gate then is rejected by candidate selection | [CPU preparation](docs/style_improvement_20261006/shared_social_episode_evaluation_preparation_v1/README.md): repeated private-helper import fixed without changing the original loader.10 artifacts cold sealed;57 bindings/3 actual S2 metadata inputs/23 S3 guards preserved. Original evaluation numerics and child paths retained; new candidate training, terminal acceptance and SDK results pending |
| Shared Social CNN AMP S3 / longer budget | Incomplete at12454/12500 updates (99.632%),102023168 transitions; last log2026-10-07 20:55:08 Asia/Shanghai | [Session-unavailable observation](docs/style_improvement_20261007/shared_social_S3_session_unavailable.json): original training28927 and owner581 handles unavailable; actual exit/cause unknown. No result/step012500/completion certificate; last saved step012000. Terminal audit/evaluation pending; complete SDK/curriculum/history resume unsupported; no restart in this status check |
| Shared Social CNN AMP S3-R / optimizer recovery | Completed2026-10-07 23:46:12 Asia/Shanghai: training and terminal CPU audit actual child exits0;500 recovered updates,12500 effective total /102.4M transitions | [Completed recovery](docs/style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/README.md), [external cold verification](docs/style_improvement_20261007/shared_social_S3_optimizer_recovery_v2/owner_capacity_v3/external_verification.json): preselected step012500 and final rollout verified;28 outputs readonly,202 tensors audited,16 sources/4 inputs plus original weights/logs/fixed2848 unchanged. World/private curriculum/history restarted, unsaved454 updates rolled back; no continuous-S3 certificate, GPU exclusivity or independent quality result claimed |
| S3-R recovered terminal independent evaluation | Completed2026-10-08 01:34:43: original successful prefix retained; independent navigation200s and original raw CPU audit actual exits0 | [Strict S2 comparison](docs/style_improvement_20261008/shared_social_S3R_comparison_v1/README.md),72 bound artifacts: full22 success0→14/22,zero falls,position.276747m/rotation.749612rad;45D MMD².0230231 but coverageK50=.58/K100=.52 declines. Navigation falls9→0,peer pair-controls32→0,vertices1→4; persistent route stagnation. Original post1/navigation-11 and GDB diagnostic0 retained; cause unidentified. No original-post success or paper/default promotion claim |
| Public SFM for the same AMP | Joint150-step predictor; fresh30/6000 controls and native raw audit exit0 | SFM200s:20 falls/21 static actor-controls/31 peer pair-controls/6 vertices; quality fails |
| Fresh N4 high-level pairing | Both200s GPU/raw CPU/initial/style processes exit0; all32 live initial channels/resources/RNG match | [Complete result](docs/style_improvement_20261005/shared_social_S1_paired_actual_v1/README.md); DP5 falls/37 peer pair-controls/3 vertices, SFM20/31/6; both fail quality |
| Closed-loop navigation style | Eight CPU regression tests and actual paired measurement pass | [Style entry](scripts/evaluate_shared_social_navigation_style.py): fixed validation / K50 / K100; failures retained; five common balanced90D bins DP .112125/SFM .131224, DP coverage/route lower; no overall superiority |
| Native shared Social AMP preparation | Four measured native maps,52 fixed resource cases:48 admitted/4 Hospital rejections;2058 A* segments;48 CPU SFM snapshots exit0;30 CPU checks pass | [Native entry](docs/SHARED_SOCIAL_NATIVE_EVALUATION_ZH.md) supports N3–15 and explicit DP/SFM/Pop-Art source preparation; native SDK smoke, paired initial capture and full physical/style admission pending |
| Native pre-planner initial capture preparation | Same hook for DP/SFM; explicit N3–15, full body/history/RNG plus four native map channels;79 relevant CPU checks and both strict resource preflights pass | [Final CPU evidence](docs/style_improvement_20261005/shared_social_native_initial_preparation_v3/README.md); mutation detected and exact source restored; actual native World captures, exit and200s quality remain unverified |
| Native N3–15 raw audit preparation | Explicit30/6000 controls;62 CPU checks and4 source checks pass; historical actual DP/SFM N4 metrics exactly preserved | [Readonly CPU evidence](docs/style_improvement_20261005/shared_social_native_raw_audit_preparation_v2/README.md); exit-guard mutation detected21 failures and restored; driver ABI corrected; actual new native capture/style/quality unverified |
| Native N3–15 independent style preparation | Full6000 controls and both external raw CPU exits required;53 CPU checks pass; actual historical DP/SFM pose45/motion90/common-bin fields unchanged | [Readonly style evidence](docs/style_improvement_20261005/shared_social_native_style_preparation_v2/README.md); frozen validation/basis unchanged,92 files sealed; tuple/list runner failure retained; actual new native style/quality unverified |
| Explicit-seed native entry and local registration | 48 CPU checks and17 actual child exits pass; original paired driver bytes preserved; local seeds7/19/42 register156 cohorts,144 resource-admitted/12 rejected | [Readonly seeded evidence](docs/style_improvement_20261006/shared_social_native_seeded_preparation_v1/README.md):88 files verified;288 full200s arm runs are planned, none executed; actual SDK/raw/style admission pending; author eight-scene protocol unmatched |
| Explicit-seed native raw/style interfaces | 195 checks and33 actual CPU child exits pass;24 raw/6 style cold source checks; seed-equality mutation causes13 failures and exact restore | [Readonly CPU evidence](docs/style_improvement_20261006/shared_social_native_seeded_admission_preparation_v1/README.md):119 files verified; all six kernels and historical actual DP/SFM raw/45D/90D metrics unchanged; actual native SDK quality unmeasured. [Corrected test inventory](docs/style_improvement_20261006/shared_social_native_seeded_admission_preparation_v1/test_inventory_correction.json):95 new +100 original tests |
| N4 smoke/full raw admission | Seven CPU contracts; historical30/full6000 actual exit0; all24 body positions match the native fourth substep, full S1 metrics unchanged | [Registered-duration audit](docs/style_improvement_20261005/shared_social_high_level_raw_audit_v1/README.md); historical runs have no new paired initial record and are not fresh SFM/DP evidence |
| Fixed2848 expert support | Five CPU contracts; actual train494792 centers and S1 terminal4096 IDs/frame centers verified; fast original clip-uniform prior1.5801% | [Prior/exposure diagnosis](docs/style_improvement_20261005/shared_social_expert_coverage_CPU_v1/README.md); no resampling or causal quality claim; terminal batch does not prove entire per-ID training exposure |
| Actual PRE task context | Eight CPU contracts; all8192 terminal control rows and4096 experts observed with phase/episode/native velocity checks | [Task context diagnosis](docs/style_improvement_20261005/shared_social_task_context_CPU_v1/README.md); future displacement rates are not instantaneous velocity commands; no expert/task pairing, resampling or causal quality claim |
| BABEL dense span convention | Six CPU contracts and actual fixed2848 metadata extension exit0;399 clips/304 single-action dense parents | [Explicit assumption layer](docs/style_improvement_20261005/babel_dense_sequence_convention_CPU_v1/README.md), separate from844 timed full-clip unions; no extra fallback, dataset selection or training change |
| Optional arm-style blend | Native alpha .25 / 30 s: zero falls, 8 arrivals; pose MMD improves slightly, speed W1 and K100 coverage worsen | Mixed short-run tradeoff; default unchanged |

These local results do not establish the paper's eight-scene performance or superiority. Tracking, pose style, speed, falls and contacts are reported separately, including failed frames.
Completed Social comparisons also require [supplemental evidence admission](scripts/verify_shared_social_evidence_admission.py): actual child ownership, complete tracking protocol, frozen underlying style code and equal sample/reference subsets for directional 90D scores. See [physical support diagnostics](docs/style_improvement_20261005/shared_social_locomotion_CPU_v1/README.md) and [negative terminal-D action sensitivity](docs/style_improvement_20261005/shared_social_D_input_diagnostic_v1/README.md). CPU diagnostics are separate from training ablations and physical performance claims.

## Execution entry points and readiness

| Task | Entry / instructions | Verified state |
| --- | --- | --- |
| Current results and active training | [Reproduction status](docs/CURRENT_REPRODUCTION_ZH.md) | Formal Diffusion40000 and fixed open-loop comparison closed0; same-AMP 200 s physical comparison closed0 but fails quality: F1/peer45/routes24 |
| Run the selected pedestrian simulation | [Baseline command](#run-the-selected-physical-navigation-baseline) | Existing physical baseline and native-scene evidence |
| Train the new shared social CNN + AMP | [Current commands and evidence](docs/SHARED_SOCIAL_AMP_CURRENT_ZH.md) | S0/S1/S2 complete fresh2000/16.384M and all independent measurements with actual exit0. All remain0/22 tracking; [S1 comparison](docs/SHARED_SOCIAL_POPART_RESULTS_ZH.md) and [S2 comparison](docs/SHARED_SOCIAL_FULL_EPOCH_RESULTS_ZH.md) retain separate style/route/physics axes. Quality fails; no default promotion |
| Compare fixed validation motion style | [Validation-only metric entry](docs/style_improvement_20261004/formal_EMA40000_style_metrics_preparation_v1/README.md) | Actual 90D and 45D CPU runs closed0; MMD² .05592538 / .02297688; fixed references and centers, no Test34 array read or default promotion |
| Inspect same-AMP style and physical tradeoffs | [Historical fixed-protocol comparison](docs/style_improvement_20261004/formal_EMA40000_style_historical_comparison_v1/README.md) | OLD→EMA 90D .07238060→.05592538, 45D .02015677→.02297688; fall0→1 and peer0→45. No overall improvement or paper Table III equivalence claimed |
| Evaluate the new AMP on every validation motion | [Current full22 entry](scripts/evaluate_shared_social_tracking.py) | S0/S1 all22/11982 physical frames complete; both0/22 success; frozen style separately reports coverage/speed/common bins |
| Evaluate the new AMP with Diffusion | [Current four-person entry](scripts/evaluate_shared_social_navigation.py) | S0/S1 full200s actual0 and independent raw audits complete; falls956→5, peer pair-controls0→37, route vertices0→3 |
| Compare completed independent results | [CPU evidence comparison](scripts/compare_shared_social_evidence.py) | Actual exits, artifact identities, full22 windows, frozen reference/basis, speed bins and physical budgets required; separate axes, no automatic promotion |
| Evaluate the new AMP in a native 3–15-person scene | [Native evaluation entry and explicit seeds](docs/SHARED_SOCIAL_NATIVE_EVALUATION_ZH.md) | CPU interfaces, fixed resource cohorts and independent raw/style kernels prepared; actual new SDK captures, initial pairing and200s quality pending |
| Observe actual serial native tool exits | [Registered case owner preparation](docs/style_improvement_20261006/shared_social_native_tool_owner_preparation_v2/README.md) | 42 pytest and13 in-memory V8 contracts pass; runtime commands match readonly registration before any SDK call. Actual pending S3 gate exits1 with no SDK launch; complete21-stage native execution still pending |
| Check actor loading and value gradients | [Future CPU inference checks](docs/style_improvement_20261004/social_inference_future_CPU_v1/README.md) | Four actual numerical checks passed; synthetic fixture, real checkpoint/physics still pending |

Use the entry's declared state when interpreting its commands. The shared social policy
has been trained and measured in physics; its current quality remains insufficient.

<details>
<summary>Method components, resources and complete evidence index</summary>

- [Current reproduction status, active experiments and actual evidence](docs/CURRENT_REPRODUCTION_ZH.md)
- [Actual paired evaluation with independent OLD snapshot](docs/style_improvement_20261004/paired_validation_readonly_baseline_v3/README.md): actual CPU91992/23e92b closed0; all181 output files and owner sealed. Main200 ADE/FDE improve53.77%/51.95%, but peer-point collision increases3.06%→3.20%. Diagnostic504 remains separate (20 overlap). First import-path failure preserved; explicit PYTHONPATH corrects the direct-file launch. Fixed same-AMP physical GPU89246/824a99 and raw CPU92420/8e8952 closed0: F1/static0/peer45/routes24 vs OLD0/0/0/21. No paper-performance or default-promotion claim.
- [Prepared paired evaluator with recipe-supported EMA constants](docs/style_improvement_20261004/paired_validation_EMA_normalization_v2/README.md): source-only inverse/AST checks, independent review and root freeze pass. It retains the original fixed selection and terminal admission. Strict terminal checks now pass; actual paired evaluation awaits the independent baseline snapshot binding.
- [Complete validation loss diagnostic](docs/style_improvement_20261004/full_validation_loss_diagnostic_v1/README.md): actual CPU inventory verifies Maps10,566 + Interact19,386 =29,952 unique windows. Scalar tail-weighting checks and source review pass; model execution follows the fixed40k main comparison and uses common per-window noise without promoting a historical checkpoint.
- [Actual-body tracking versus recorded contact](docs/style_improvement_20261004/actual_tracking_contact_CPU_v1/README.md): all 72,000 actor-controls from three sealed 200s captures analyzed on CPU; source phase and complete CSV checks pass. Reference clearance and actual contact disagree, so geometric improvement alone is not physical success. No new simulation or model was run.
- [Actual peer-force severity distribution](docs/style_improvement_20261004/contact_force_severity_CPU_v1/README.md): complete36,000 pair-controls per capture, closed0 CPU execution and source/count checks pass. NEW S1/S10 have48/202 observations above50N, with different median/tail changes. This paper-inspired threshold differs from PACER's masked-body net-force termination rule; it does not promote the candidate or replace the paper's collision metric.
- [Actual same-control velocity response](docs/style_improvement_20261004/reference_response_CPU_v1/README.md): CPU session27686 closes0, all72,000 rows retain failures, low-speed and reset cases. Actual Pelvis velocity and reference use the same completed control phase; contact-frame errors exceed each new model's full-capture mean. This is association, without delay fitting or causal claims.
- [Social raster primitive](docs/style_improvement_20261004/social_raster_staging_v1/README.md): four synthetic CPU cases and independent review pass. Implements64×64×3 height/relativeXY velocity with explicit scene, frame and mirror contracts. The12308-task input requires a new policy/normalizer; shared-group training and physical validation remain pending.
- [Torch social raster backend](docs/style_improvement_20261004/social_raster_torch_staging_v1/README.md): seven actual CPU cross-backend cases cover1024 complete egos and the mixed256 layout; tested owners match exactly and raster atol2e-6 passes. Same-device tensors and scene chunks are implemented; GPU equivalence, synchronization cost and performance are not yet measured.
- [Raw social task assembly](docs/style_improvement_20261004/social_task_observation_staging_v1/README.md): four actual CPU checks passed in.012s, with CUDA uninitialized. Assembles12308 raw task, explicit world/root/grid contracts and one batch terrain query; mirrors raw task and replaces only candidate trajectory prefix. Coupled self/action mirror is now a separately checked stage below; SDK/current-clock wiring remains pending.
- [Social CNN actor/critic source](docs/style_improvement_20261004/social_policy_staging_v1/README.md): standalone shared CNN/map256+trajectory20,633-input separate actor/critic and differentiable normalized value interface authored. Compile/Ruff only; construction/forward/backward remain unexecuted; the necessary CPU gates follow the running primary physical comparison. Old1044 weights/RMS/loader are incompatible.
- [Shared-training integration contract](docs/style_improvement_20261004/social_training_integration_v1/README.md): connects actual four-actor scene filtering, flat-state mapping, group resets/terminal values, histories/curriculum, full-task RMS, mirror and future value-guidance boundaries. It is not a runnable shared-training entry.
- [Four-articulation scene factory](docs/style_improvement_20261004/group_articulation_scene_staging_v1/README.md): lazy SDK config factory follows installed queued replication; four independently copied actors, N/4 envs and local/global collision groups. Two stdlib mock checks/.001s passed; no SDK scene construction or actual collision validation.
- [Four-articulation tensor bridge](docs/style_improvement_20261004/group_articulation_tensor_bridge_staging_v1/README.md): five CPU tests/.010s passed for four buffers↔canonical actor rows,69-coordinate targets and whole-group reset indices. No body/joint/quaternion conversion or SDK writes occurred.
- [Social training settings](docs/style_improvement_20261004/social_training_settings_staging_v1/README.md): four stdlib checks/.001s passed for12308/50/full RMS/57-effective-action contracts and exact configuration roundtrip. Declares a fresh local variant; current old trainer/loaders remain unchanged.
- [Coupled raw social symmetry](docs/style_improvement_20261004/social_symmetry_staging_v1/README.md): four CPU tensor/autograd checks/.020s passed5b2c14 for raw self357/task12308 and exact action69 reflection,57-mask reduction and two-branch gradients. Root sealed4f94bf. No normalized policy or physical symmetry experiment.
- [Independent social CNN/PPO/AMP trainer](docs/style_improvement_20261004/social_trainer_staging_v1/README.md): new CNN+D2019 factory, whole-task RMS, raw social mirror and strict new checkpoint admission authored; original PPO/D/GAE math reused through bounded source derivations. Compile/Ruff/inverse bytes/AST and review passed. No NN construction, update/resume test or physical runner execution yet.
- [Shared-group World driver](docs/style_improvement_20261004/shared_group_world_staging_v1/README.md): four actual-future substeps, current normal contacts, terminal consumption and complete-group history/curriculum reset authored. The three geometry declarations match the trainer contract. Final35097B/b608cae6 compile/Ruff/review and root seal0780fd passed; four fake-boundary CPU driver checks passed once (.207s,2d9f97→1de562 exit0) and test/receipt were sealedf3b74f. Actual SDK/physics remain unverified.
- [Exclusive social rollout collector](docs/style_improvement_20261004/social_rollout_collector_staging_v1/README.md): connects the prepared World and trainer, with pre-state_action history, raw/applied actions, terminal next values, mean-substep power, frozen four RMS and failure preservation. Final27252B/f7d27270 is reviewed/sealed; four fake-boundary CPU collector checks passed8dd007 exit0/.034s. Real NN/physics linkage, full training CLI/expert/SDK resume remain pending.
- [Fresh shared SocialCNN PPO+AMP CLI](docs/style_improvement_20261004/shared_social_training_entry_staging_v1/README.md): reviewed/sealedf12a5004/aa119d; fixedtrain2848,8-actor bounded smoke and256-actor local2000 pilot are explicit profiles. OuterAMP1/task.5; fresh only, finite floor, original collector/expert/update ordering and source-bound logs. No CLI/NN/SDK/train execution yet; physical resume unsupported.
- [Bounded real-SDK shared World probe](docs/style_improvement_20261004/shared_group_world_SDK_probe_staging_v1/README.md): reviewed/sealed87600495 (1f82af); one App, actual train2848,8 actors/two groups, control/reset archives and optional explicitly manipulated fall. Source-only; run serially after formal main comparisons and the separate collision smoke. Zero actions/synthetic nextV are interface probes, not trained policy results.
- [Native net-normal view factory](docs/style_improvement_20261004/social_normal_force_view_factory_staging_v1/README.md): manager-owned per-actor24 paths/reporting/no-filter bindings and stale-lifecycle refusal. Three stdlib mocks69328e pass/.012s; source/test/receipt sealed9c7538. Real SDK force/reset/performance remain pending.
- [AMASS expert batch connection](docs/style_improvement_20261004/social_amass_expert_staging_v1/README.md): reuses original uniform draw/ten-state history and named next-frame PD action proxy, with idle/live decoder and post-draw interrupt guards. Four synthetic CPU cases567839→33fabb closed0/.674s; source/test/receipt sealed. No real AMASS values, NN, physics or positive optional q path executed.
- [Social actor inference snapshot](docs/style_improvement_20261004/social_deployment_integration_v1/README.md): source-only05261849/6e22a2; owns existing trained encoder/actor/two RMS/mask and supports arbitrary positive inference batch. No NN execution; independent checkpoint-reader source is prepared, actual loading/evaluation remain pending;22/200s source producers are now authored, native producer/entry source is prepared; actual execution remains pending; per-call full weight checks require real latency measurement.
- [Strict social checkpoint inference](docs/style_improvement_20261004/social_checkpoint_inference_staging_v1/README.md): reviewed/sealedcd227f5d/6707b2, mandatory readonly snapshot SHA, Linux descriptor-bound CPU/mmap/weights_only, full model/four RMS admission then actor-only storage. Source-only; no weight load or NN execution, and no optimizer/replay/RNG/SDK resume validation.
- [Same-phase evaluation observation producer](docs/style_improvement_20261004/social_evaluation_observation_staging_v1/README.md): reviewed/sealed2379d658/b992fb; originalN1 or sharedN4 COMMON snapshot, raw357/12308, exact policy geometry, explicit real reset notifications and actual clock/quaternion/fallen-heading diagnostics. No policy/SDK/data execution.
- [Social full22 tracking entry](docs/style_improvement_20261004/social_tracking_evaluation_staging_v1/README.md): reviewed/sealed79fd24b2/e0dba3; strict social loader and producer replace old interfaces while all22/11982 steps, original physics/reference/errors and failures remain. Exact source inverse/AST/compile/Ruff pass; no actual model or tracking execution.
- [Social four-person navigation entry](docs/style_improvement_20261004/social_navigation_evaluation_staging_v1/README.md): source-prepared independent evaluation profile and actual shared-training receipt/model/USD admission, original Diffusion/reference/6000-control/contact loop, same-phase social inputs and failure cleanup. Static compile/Ruff/derivation and independent review pass; c31c8739/e207b1 is sealed, no preflight/NN/SDK run.
- [Differentiable frozen social critic snapshot](docs/style_improvement_20261004/social_value_adapter_staging_v1/README.md): owns only existing encoder/critic/three RMS; keeps current body/raster fixed and varies20 local trajectory entries, retaining input-gradient/denormalization semantics. Final12573B/9875c58b is reviewed/sealed source only, with no NN execution or collision-quality claim.
- [Four-articulation state adapter](docs/style_improvement_20261004/group_articulation_state_adapter_staging_v1/README.md): five fake-asset CPU checks/.120s passed for named COMMON conversion, owned link snapshots, buffered position targets and complete-group partial resets. Root sealed source/tests/proof6e49d6; actual SDK writes, contact/power and history/clock integration remain pending.
- [Eight-actor/two-group PhysX smoke entry](docs/style_improvement_20261004/group_articulation_physics_smoke_staging_v1/README.md): single-App source checks exact contact-path ordering, a positive same-group probe and strict zero cross-group raw forces, with complete substep records. Compile/Ruff and source review passed; source sealed6e49d6. No SDK or physics execution yet.
- [Named nonfoot net-normal contact predicate](docs/style_improvement_20261004/nonfoot_net_normal_contact_staging_v1/README.md): four CPU tensor checks/.025s passed61ce0b. Excludes four named feet, sums remaining body vectors and tests norm>50N/progress>1; actual all-partner net-normal SDK sampling and full early termination are pending. Historical peer-pair maxima cannot supply this input.
- [Current all-partner net-normal sampler](docs/style_improvement_20261004/social_net_normal_sampler_staging_v1/README.md): four fake CPU cases/.027s passeddb516f; root sealed052ba9. Validates existing per-actor view paths/names, clones current forces and maps bodies/actor rows. Native SDK execution, force synchronization and grouped-view extension remain unverified.
- [BABEL train semantics](docs/style_improvement_20261004/babel_train_semantics_v1/README.md): actual CLI68758 exited0 and preserved all2848 v7 rows/order. Exact raw-source joins match1646 clips;844 have complete timed union coverage,1202 have no train annotation match. The sealed37.5MB overlay preserves multi-annotator/invalid/unknown evidence; style5, sampling and weights remain unchanged. Semantics-aware training is pending.
- [Four-actor groups and reset masks](docs/style_improvement_20261004/social_group_reset_staging_v1/README.md): five NumPy cases, Ruff and independent review pass. Preserves flat256 rows and mixed2+2 sources per group; genuine falls remain terminal and collateral resets truncate GAE carry. Collision construction, runner and episode/history/plan integration are pending.
- [Common five-second raw / adapted-reference S10 ranking](docs/style_improvement_20261004/reachable_candidate_selection_staging_v1/README.md): isolated entry and driver ready; 10 CPU cases and both actual SDK-free resource preflights pass. The independently reviewed candidate/physical reader and [additive source plan v2](docs/style_improvement_20261004/reachable_candidate_selection_staging_v1/commands_and_audit_plan_v2.json) are frozen; new physical runs and numerical audit execution remain pending. This does not replace deployment defaults.
- [Four native scenes: complete independent map review](docs/style_improvement_20261002/native_map_v9_four_scene_independent_review_v1/README.md)
- [Usable native Warehouse indoor recorded video](output/diffusion_amp/native_warehouse_recorded_replay_indoor_30s_v1/pedestrian_motion.mp4) and [media/source audit](docs/style_improvement_20261002/native_recorded_replay_reader_v1/README.md)
- [Current project structure and resource/evaluation index](docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md)
- [Actual current entry-point and AMASS split audit; rebuilt source package](docs/current_structure_review_20261003/README.md)
- [AMASS, training experiments, fixed checkpoints and physical evidence](docs/AMP_STYLE_PROGRESS_20261002_ZH.md)
- [Fixed 15-keypoint pose coverage / entropy / MMD](docs/style_improvement_20261002/pose_diversity_candidate_v1/README.md)
- [Actual terrain-normalizer and saved-navigation diagnostic](docs/style_improvement_20261002/navigation_terrain_rms_review_v1/README.md)
- [Actual pre-action input audit: 30 samples, physics arrays equal to baseline](docs/style_improvement_20261002/pre_action_actual_observation_reader_v1/README.md)
- [Recorded SMPL mesh / physical joint correspondence audit](docs/style_improvement_20261002/recorded_smpl_skin_audit_v1/README.md)
- [Registered fresh terrain-statistics protection experiment](docs/style_improvement_20261002/terrain_protected_experiment_registration_v1/registration.json): fresh2000 / 16.384M transitions complete; 2,046 terminal checks pass; full22 tracking 15/22, position 0.248 m, rotation 0.775 rad; four-person200s navigation: zero falls/contacts,25 arrivals, idle93.87 actor-seconds; style MMD0.09014 worsens, candidate rejected
- [Complete protected-statistics navigation/style review](docs/style_improvement_20261002/terrain_protected_navigation2000_complete_review_v1/README.md)
- [Native arm blend, same 900-frame comparison](docs/style_improvement_20261002/native_arm_alpha025_30s_style_v1/README.md)
- [Read-only AMASS style coverage audit](docs/style_improvement_20261002/amass_style_coverage_v7_read_only_v1.md)
- [Registered full 22 / 200 s state-action evaluation](docs/style_improvement_20261002/state_action_evaluation_registration_v1/README.md)
- [Historical implementation and training stage](docs/PEDESTRIAN_AMP_DIFFUSION_PROGRESS_20261001_ZH.md)

</details>

## Run the selected physical navigation baseline

Run from this checkout using the pinned environment and the existing supplied
assets. Select a new output directory for every run.

```bash
source ../IsaacLab-Protomotions/.venv/bin/activate
python scripts/simulate_paper_pedestrians.py --inference-only \
  --amp-checkpoint output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt \
  --config configs/diffusion_amp/amp_style_amass_backward_penalty.yaml \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --duration-seconds 200 --output output/diffusion_amp/navigation_NEW_RUN
```

Add `--preflight-only` to inspect the resource contract without starting Kit.
For live textured SMPL RGB and optional depth capture, use the separate
[`capture_paper_pedestrians.py` entry](docs/style_improvement_20261004/live_pedestrian_capture_v1/README.md).
Its SDK-free preflight and CPU interfaces pass; actual RGB-D rendering remains pending.
For native Warehouse, add `--native-dynbench` and
`--scene-config configs/diffusion_amp/native_warehouse_pedestrians_clearance08_v1.yaml`.
Its completed four-person, 200-second test had zero falls, zero measured
static/peer contacts, 57 vertex arrivals and 12 full route cycles. The
[983-check independent audit](docs/style_improvement_20261002/native_scene_200s_actual_review_v1/actual_report.md)
verified the live stage, floor rays and full physical capture. The completed 15-person test also had zero falls and measured contacts, 211 arrivals and 49 full cycles; [all 1,325 raw checks pass](docs/style_improvement_20261002/native15_200s_independent_review_v1/actual_report.md). For that configuration add `--actors 15` and select `configs/diffusion_amp/native_warehouse_pedestrians_15_clearance08_v1.yaml`. Neither this authored fixture nor the available four native
USD scenes supplies the authors' complete eight-scene evaluation protocol.

## Paper pedestrian method: Diffusion + AMP

The new method entry points are `scripts/train_paper_diffusion.py` and
`scripts/train_paper_amp.py`, with `configs/diffusion_amp/method_reconstruction.yaml`.
They train a five-state trajectory x0 diffusion model and PPO/AMP using direct
SMPL PhysX simulation. The original method profile supports an action-conditioned
discriminator; the selected public-PACER transfer baseline uses a state-only discriminator
adaptation. A separate history-and-action training run completed full evaluation
and failed navigation selection. The standalone pedestrian command
in the later section retains its existing SFM/MaskedMimic configuration.
See [the Diffusion/AMP implementation and training guide](docs/DIFFUSION_AMP_RECONSTRUCTION_20261001_ZH.md)
and `output/diffusion_amp/` for actual training artifacts and declared assumptions.

```bash
source ../IsaacLab-Protomotions/.venv/bin/activate
python scripts/prepare_diffusion_amp_data.py \
  --output data/diffusion_amp/amass_NEW_manifest.json
python scripts/train_paper_diffusion.py \
  --manifest data/diffusion_amp/amass_NEW_manifest.json \
  --output-dir output/diffusion_amp/diffusion_NEW_RUN --steps 1000 --batch-size 32
python scripts/train_paper_amp.py \
  --manifest data/diffusion_amp/amass_NEW_manifest.json \
  --diffusion-checkpoint output/diffusion_amp/diffusion_NEW_RUN/last.ckpt \
  --updates 1000 --num-envs 64 --physics-only \
  --synchronized-replanning --output output/diffusion_amp/amp_NEW_RUN
```

Use new names for the manifest and both output directories. This example rebuilds
the basic trajectory corpus; the selected 2,848-segment style bank is the separate
`data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json`.

The derived AMASS manifest, training parameters and new weights do not establish
equivalence to the authors' unpublished 2,848 clips, eight scenes or checkpoints.

The AMP configuration and file checks can run before importing Torch or Kit:

```bash
../IsaacLab-Protomotions/.venv/bin/python -m CrowdSim.humanoid.application.preflight \
  --config configs/diffusion_amp/method_reconstruction.yaml \
  --diffusion-checkpoint output/diffusion_amp/diffusion_training/step_1000.ckpt \
  --output output/diffusion_amp/PREFLIGHT_NEW_RUN
```

This applies the real CLI and configured local resource paths without launching
simulation. Checkpoint signatures/file presence do not establish tensor/model
compatibility; the runtime retains its dedicated loading gates. CLI paths use
cwd; relative AMASS package paths in a manifest use that manifest's directory.

The historical three four-person 200-second obstacle evaluations each had 16
falls. Later independently trained AMP and explicit deployment changes improved
this fixture; see the latest guide above for the fixed models and full command.
The separate bounded 32-person
50-second straight-path diagnostic had no falls for either public PACER or the
corrected trained AMP controller. These are distinct physical tasks. Exact
filtered normal contacts are now verified on controlled wall/peer/floor probes;
historical captures without contact channels retain unknown physical contacts.
See [the evaluation index](docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md#已有物理评估能说明什么)
for the actual reports and comparison limits.

The standalone pedestrian path has been migrated and tested with Isaac Sim 6 / IsaacLab 3.
Start with [the current completion report](docs/PROJECT_COMPLETION_20261001_ZH.md)
for the environment, fixes, packaging, resources and physics/render evidence.

The repository provides two complementary training paths:

1. `src/crowdsim_ppo/`: a simulator-free reference pipeline for quickly reading
   and testing the algorithm.
2. `CrowdSim/` + `protomotions/`: the original Isaac Lab crowd-navigation
   pipeline with humanoids, robot sensors, maps and differential-drive control.

## Included dependency closure

```text
CrowdSim/ppo/train_ppo.py
├── CrowdSim/ppo/ppo_policy.py
├── CrowdSim/world/builder.py
│   ├── CrowdSim/protomotions_runtime.py
│   │   └── protomotions/
│   ├── CrowdSim/scene_setup.py
│   ├── CrowdSim/nav_manager.py
│   │   ├── CrowdSim/world/{map,planning,sfm}.py
│   │   └── CrowdSim/control/drive.py
│   └── CrowdSim/utils/
├── CrowdSim/tools/render_navigation_fast.py
└── CrowdSim/config/env.yaml
```

Preference PPO additionally uses `CrowdSim/pref/`. Expert trajectory collection
uses the selected files under `CrowdSim/flow/`. PPO-specific tests and evaluation
utilities are included under `CrowdSim/tests/` and `CrowdSim/tools/`.

## External files

The following large or separately distributed files are intentionally not stored
in this Git repository:

- MaskedMimic checkpoint, normally
  `data/pretrained_models/masked_mimic/smpl/last.ckpt`
- `resolved_configs_inference.pt` in the same directory as that MaskedMimic
  checkpoint (required by the runtime)
- motion file referenced by `CrowdSim/config/env.yaml`
- warehouse or other scene USD referenced by `CrowdSim/config/scenes/*.yaml`
- trained robot PPO and reward-model checkpoints

Put those files at the configured paths or update the YAML files. ProtoMotions
robot definitions and mesh assets required by its simulator backends are included
under `protomotions/data/assets/`.

Some media and USD files in the source checkout were available only as Git LFS
pointer files. Their pointer metadata is retained, but the corresponding binary
objects must be downloaded from the upstream ProtoMotions distribution when a
simulator configuration needs them.

## Environment

The tested path uses **Isaac Sim package 6.0.0.1**, **Python 3.12**, the PhysX
backend, and IsaacLab commit `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`.
That checkout has product `VERSION=3.0.0` and Python package `isaaclab==12.0.0`.
It is a pinned development commit; the product and package numbers differ.

For an existing checkout with the specified Sim and Lab already installed,
apply and check the recorded dependency profile from this repository:

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/isaaclab3_environment.py apply \
  --lab-root ../IsaacLab-Protomotions --sync
../IsaacLab-Protomotions/.venv/bin/python -I scripts/isaaclab3_environment.py check
source ../IsaacLab-Protomotions/.venv/bin/activate
```

The [environment profile](env/isaaclab3_sim6/README.md) includes the complete
manifest/lock, pytest and a check for the repaired USD provider conflict. The
script validates the existing installation, backs up its dependency files and
does not perform a new Sim installation. Run the commands below with that Python.

The scene builder checks required files before launching Isaac Sim and reports
missing files and unresolved Git LFS pointers. Scene-only `--empty` mode does
not require humanoid checkpoints, motion files or a navigation map.

## Standalone pedestrians

The default uses existing SMPL MaskedMimic weights and AMASS validation motions,
SFM targets, a local neutral SMPL mesh, and a shared PhysX world. It requires no
robot policy. GUI mode runs until the window closes:

```bash
python -m CrowdSim.humanoid.simulate --num-humanoids 4
```

Bounded headless execution saves physical states, a result report, RGB frames
and an MP4 before Kit shuts down:

```bash
python -m CrowdSim.humanoid.simulate \
  --headless --steps 240 --num-humanoids 4 --reset-at 120 \
  --capture --output-dir output/pedestrian_simulation
```

The earlier four-person 8-second run passed for SFM + MaskedMimic.
The Diffusion + AMP implementation, new training artifacts and its independent
evaluation commands are documented in the guide linked above. Author experiment
equivalence still requires the unpublished motion subset and evaluation protocol.
Robot training/evaluation below also requires its configured assets and policies;
it was not included in the standalone pedestrian runtime acceptance.

## Train

```bash
python CrowdSim/ppo/train_ppo.py \
  --env-config CrowdSim/config/env.yaml \
  --total-steps 5000000 \
  --headless
```

Resume:

```bash
python CrowdSim/ppo/train_ppo.py \
  --env-config CrowdSim/config/env.yaml \
  --resume output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt \
  --headless
```

Collect expert trajectories:

```bash
python CrowdSim/flow/collect_ppo_buffer.py \
  --env-config CrowdSim/config/collect/ppo.yaml \
  --ppo-ckpt output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt
```

Preference-reward fine-tuning:

```bash
python CrowdSim/ppo/train_ppo_finetune.py \
  --env-config CrowdSim/config/env.yaml \
  --train-config CrowdSim/config/reward_model_ppo.yaml \
  --base-ppo PATH/robot_ppo_latest.pt \
  --reward-ckpt PATH/reward_model.pt \
  --headless
```

## Evaluate and test

Grid evaluation preserves the configured humanoid count, runs one robot, and
disables the training curriculum. Goal distances come from `navigation.path`,
so evaluation with the default config uses 7–10 m goals.

```bash
python CrowdSim/tools/eval_policy.py \
  --ckpt output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt \
  --env-config CrowdSim/config/env.yaml --num-episodes 200 --seed 42 --headless
```

Cached plots can be regenerated without starting Isaac Sim or supplying a
checkpoint:

```bash
python CrowdSim/tools/eval_policy.py --load-data PATH/policy_eval.pkl
```

Core tests do not start Isaac Sim but require PyTorch, pytest, and the map/plot
Python dependencies in the environment profile. The simulator-free package is
tested directly from this source checkout:

```bash
PYTHONPATH=src:. python -m pytest --import-mode=importlib tests CrowdSim/tests
```

Tests under `protomotions/tests` additionally cover other simulator backends.
The earlier completion-stage full regression passed 1,886 tests with two skips and six LFS
deselections, including Newton FK. Two local SMPL inference configuration tests
were also run separately and passed. This historical result predates the latest
pedestrian changes. The earlier AMP/Diffusion core gate passed 463 tests in 20
files; see [the final focused log](docs/amp_improvement_20261001/final_core_tests.log).
The old G1 motion cache is inconsistent with
the current asset; the backend agrees with independently computed current-asset
FK. SMPL exp_map is verified on PhysX; Newton/MuJoCo explicitly reject that
three-axis contract unless configured for matching hinge data and policies.

The simulator-free package can also be installed independently; its default
configuration ships in the wheel and works outside a source checkout:

```bash
python -m pip install -e ./src
crowdsim-ppo-train --device cpu --total-steps 4096 --output output/point_goal
crowdsim-ppo-eval --device cpu --checkpoint output/point_goal/latest.pt --episodes 20
```

### Public AMP state prior: current opt-in trial

The state-action discriminator can now initialize its history weights and AMP statistics from the verified public PACER checkpoint; action columns start at zero and train normally. The flag defaults to false. [Merged-source CPU verification](docs/style_improvement_20261003/merged_action_state_prior_CPU_v1/README.md) passed44 new and224 regression tests. The actual pilot4 completed32,768 physical transitions and was sealed after exit; actual4→8 complete recovery passed2,120 checks and fresh2000 completed16,384,000 physical transitions with training and watcher exit0. Independent terminal admission passed2,062 checks. Full22 tracking is17/22, position0.209859m, rotation0.810943rad; four-person200s navigation completes with zero falls/contacts,40 arrivals and0.1333 idle actor-seconds. Frozen90D MMD²0.06811695 passes the registered minimum, but support narrows from5 speed bins to3 and fixed45D coverage regresses toK50=.58/K100=.52. The common three-bin secondary is0.04098256 for the baseline versus0.06811695 for the new model, so this does not show matched-speed style improvement. Native Warehouse four-person and15-person200s runs are complete: zero falls/contacts,44/160 arrivals and8/32 cycles. Common-speed-bin90D and fixed45D pose distances improve against the old same-scene runs, but route progress decreases and15-personK50 coverage drops from.76 to.70; default remains unchanged. The30-second indoor recorded-pose video passes media QC. [Complete actual review](docs/style_improvement_20261003/state_prior_complete_review_v1/README.md).

### PACER reward-ratio experiment: completed comparison

The separate profile `amp_style_mixed_state_action_state_prior_pacer_reward.yaml` changes only the outer AMP reward weight from 2.0 to 0.5; the internal scale remains 2.0. This follows the public PACER numerical recipe. NavIsaacLab does not specify its internal discriminator-reward transform, so this is a registered local experiment. Training resume now rejects changes to the saved reward contract instead of silently retaining old values. Fresh training and inference keep their existing behavior.

The separate fresh 2,000-update run completed 16,384,000 transitions and passed 2,073 complete-state checks. Full tracking regressed from 17/22 to 15/22, with position/rotation errors 0.2511 m/0.8917 rad; tracking90D MMD² increased from 0.0421 to 0.1101. The complete four-person200s run recorded3 falls and8 unordered pedestrian-contact pair-frames. Some pose45D scores improved, but the registered physical admission failed; native4/15 were not executed. The selected `step_1000.pt` remains unchanged. [Completed comparison and evidence](docs/style_improvement_20261003/state_prior_pacer_reward_complete_review_v1/README.md).

A separate1,000-layout ORCA corpus is generated, independently checked and sealed. Fresh5,000-step Diffusion training uses the registered original architecture/initialization/learning rules; existing AMP inputs remain unchanged. Its effect requires paired new-validation and fixed-controller physical tests. [Data](docs/style_improvement_20261003/orca_maps_expansion_v1/README.md), [training registration](docs/style_improvement_20261003/orca_maps_diffusion_experiment_v1/registration.json), [actual execution](docs/style_improvement_20261003/orca_maps_diffusion_actual_runs_v1/training_actual_launch.json).
