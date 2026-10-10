# Shared neutral SocialCNN PPO + AMP entry

**Source preparation only.** This directory now provides a callable fresh-training CLI and two explicit profiles. Only source compilation and Ruff have run. No CLI invocation, Torch import, network construction, AMASS loading, SDK/App construction, physics, training, checkpoint load or new test execution has occurred. Production185, defaults, formal40k, q937 and every sealed component remain unchanged.

## Connection and limits

`train_shared_social.py` checks and executes the exact sealed sources: SDK setup helper876004, shared Worldb608ca, current native-normal factory517afb through that setup, trainer81a631, collectorf7d272 and expert48bc49. Each component retains its own exact dependency checks. The setup reuses the probe's bootstrap, source-origin, asset identity and output helpers; it does **not** call the probe's zero-action main. Actual class/module origins are checked before corpus/robot/Scene configuration construction, with converter and loaded D6-workaround origins recorded. An unexecuted cached workaround is identified as unexecuted.

There is one caller App before Torch, one owned SimulationContext and four articulation slots per scene. `N` actors occupy `G=N/4` independent environments, with canonical rows `[2g,2g+1,N/2+2g,N/2+2g+1]`: two AMASS and two procedural actors per group. The unchanged original mixed curriculum uses `bounds_xy=None`, seed7 and linear-velocity tail; no hidden bound-based clip filtering or Diffusion NN is introduced. Physics is120Hz with four substeps/control, original solver4/4 and render disabled. The group filter, actual normal force readiness, body/DOF mapping and reset behavior still require future SDK execution.

Grid declaration is explicitly root-center/root-heading/static-center-ground-Z, include_ego=False. The finite Cuboid floor has topZ0, thickness0.2m and global collision group−1. Every ground and raster query outside its XY square fails rather than returning fictitious infinite ground. Smoke uses64m;256 actors/G64 with spacing8 use512m. Neither extent is a proof that all sampled motion remains inside it. Fixed offsets are not a body-AABB non-overlap proof. This is a local flat-floor experiment, **not** the paper's complete eight-scene training.

Raw self357 and task12308 feed the new shared CNN/actor/critic through the trainer's new complete-task RMS. AMP remains state_action2019: newest-first pre-step history10×195 paired with the actual clamped/masked69 action. PPO likelihood uses the unclamped Gaussian action and57 effective coordinates. The collector computes all actors' true critic next values on post-step observations **before** whole-group reset; group collateral actors remain truncated, not physical falls. Frozen RMS and model guards, original reward arithmetic, four-substep signed-power mean and failure retention are reused unchanged.

Only fresh random policy/D and moments are supported. The v7 manifest is the exact fixed path and SHA, all2848 train segments, child_joint velocities. Uniform expert sampling means uniform train clip followed by a legal center; it is not window-uniform or q-stratified. The same caller corpus is shared with reset sampling. No BABEL overlay or held-out data is loaded. AMP RMS is explicitly **per_frame195**, not per_history1950; D still consumes2019 features. The two profiles use AMP internal scale2/outer weight1 and task weight0.5, symmetry50 once,57 mask and the original PPO/D numerical update. The outer total `r_amp +0.5*r_task` follows Nav's reported weighting; the internal BCE/softplus/clamp style-reward definition remains the local PACER adaptation, not a recovered private Nav implementation.

Both profiles explicitly choose local `backwards_penalty` direction reward. The public Eq6 sign and textual direction description differ; this mode is not asserted to be a strict Eq6 implementation. A future separately recorded config/output may use the already-supported `direction_mode="paper_equation"` for that comparison. No third profile or default change is made here; the actual chosen mode is saved in configuration/provenance.

## Loop, counts and artifacts

Each update runs `collector.collect(T) → expert.sample(E) → trainer.update(rollout, expert_history, expert_actions)`, with `B=T*N` and `E=min(B,4096)`. Different B/E are accepted by the original update. Fresh self/task/value RMS counts must equal cumulativeB. Per-frame AMP count must equal cumulative`10*(B+E)`. Metrics also record expert draws, actual D minibatches and expert/live/replay exposure; exposure is the original per-minibatch sample-count mean multiplied by the actual D minibatch count. Replay draws do not add independent RMS samples, and mirrored policy observations do not update RMS twice.

| Profile | Actors/scenes | Controls/update | Updates | Expert E | PPO/D batch | Floor/cap |
| --- | --- | --- | --- | --- | --- | --- |
| smoke.json |8/2|4|1|32|32/32, one epoch and one D update|64m/4controls|
| training.json |256/64|32|2000|4096|4096/4096, six epochs and two D updates|512m/300controls|

The2000-update profile is a **local pilot**, not a recovered author training budget. Smoke completes32 physical transitions and expects AMP RMS count640. A local training update completes8192 transitions and AMP RMS exposure122880. These are declared expectations, not observed runtime results.

Outputs must be a new or empty independent directory under repository `output` or `/tmp`. Configuration and metrics use exclusive creation; step-specific checkpoints and tensor archives are never overwritten. Smoke independently archives each control before reset and its completed rollout. The longer profile archives rollouts/checkpoints every500 updates and at its final update. `result.json` is created only by an actual invocation and reports real counts, source/profile before/after identities, partial failures and cleanup. Only a completed loop with matching sources/profile and returned simulation-stop/App-close calls exits0.

Updates are not transactional. On collect/expert/update/log/save failure the process stops, retains available owned partial/complete rollout data and failure counters/native snapshot, and attempts independent SimContext/App cleanup. A partial new SimulationContext is looked up even if World construction did not return. No failed update is retried, no output is reused and no synthetic successful state is produced. OS kill or SDK process abort can bypass Python finally; returned cleanup fields cannot prove unobserved external cleanup.

Trainer checkpoints contain the strict new model, four RMS, two Adam states, replay and global RNG. They deliberately pass `runner_state=None`: **SDK state, corpus/private curriculum RNG, physical history, reference plans and group episodes are absent**. Consequently `--resume` always rejects before App bootstrap, even for this new schema. These checkpoints do not claim resumable physical training. Old1044/warm-start migration and `strict=False` loading are unsupported. No inference, critic-guidance, promotion or evaluation framework is added here.

## Future invocation, not yet run

After root review/sealing and serial runtime authorization (formal50502 currently owns the GPU):

```bash
PYTHONDONTWRITEBYTECODE=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261004/shared_social_training_entry_staging_v1/train_shared_social.py --config docs/style_improvement_20261004/shared_social_training_entry_staging_v1/smoke.json --device cuda:0 --output output/diffusion_amp/shared_social_CNN_training_smoke_v1
PYTHONDONTWRITEBYTECODE=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261004/shared_social_training_entry_staging_v1/train_shared_social.py --config docs/style_improvement_20261004/shared_social_training_entry_staging_v1/training.json --device cuda:0 --output output/diffusion_amp/shared_social_CNN_flat_pilot2000_v1
```

Profiles must be readonly before actual invocation. Required local resources are the existing SMPL MJCF/USD cache and fixed v7 train bank; bank size/declaration/source identity are checked by the existing expert adapter, not a newly claimed full-bank rehash. Actual native-group probe and pending bounded trainer NN checks are separate future runtime work. This source preparation does not establish their success or physical quality.

## Checks actually performed

Final source byte compilation only: toold14cd3, exit0. Ruff: toolbbd0ad, exit0, all checks passed. These commands did not import the new entry or any Torch/SDK/data module. No authored tests or additional runtime checks were executed. Independent whole-source review completed with Required=[] / Optional=[] (source-only f0c371). Root read the full entry (be61f9/90796e), both final profiles (2d691c/88592b) and this README (153326); source/configs are sealed readonly after these checks. Actual CLI, NN and SDK execution remain pending.
