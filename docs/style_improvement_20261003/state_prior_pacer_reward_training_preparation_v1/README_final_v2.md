# Prepared AMP reward-factor experiment — final v2

Prepared only; this directory did not launch GPU/Kit or training. Final code is
`output/isolated_training/mixed_state_action_terrain_state_prior_pacer_reward_v2`.
Its source manifest SHA is `1fd9b5e14dc377f753b04ecfe6ce93054b300d6f46c5cb341aa466f6ac887a21`.

The fresh training mechanism changes only `training.amp_reward_weight: 2.0→0.5`.
`reward.amp_weight` is the consistent metadata alias. Internal AMP scale remains
2.0, so the effective coefficient on the existing capped softplus is1.0.
Task weight0.5, negative-direction engineering reward, mixed commands, public
coupled state-prior initialization, protected terrain moments, replay, PPO/PD,
seed7, bank2848 and expert_sampling=null are retained. Corrected two inherited
assumptions and the profile label describe this new experiment truthfully.
This compares a PACER code coefficient; it is not a claim of author-equivalent
NavIsaacLab data, full reward formula or paper metric.

Code contains935 independently copied regular files:932 exact from immutable
934, two minimal training-resume guard deltas, one added absolute profile.
The new guard rejects changed saved/requested reward objective fields for
training resume and leaves fresh math/RNG and evaluation paths unchanged.
The production runner's later native changes were not copied into this bundle.
`launch_isolated.py`6d56 and `watch_snapshots.py`be59 remain byte identical.
Code files and source metadata are0444, code directories0555; shared resources
are used read-only by this experiment but not OS chmod'ed.

Final actual CPU readiness is `final_actual_CPU_readiness.json`, schema
`pacer_reward.actual_preparation_admission.v1`, statuspassed. Its owned imports
are in `final_actual_CPU_imports.json`; all actual execution paths are the final
v2 isolated code. It checked real profile load, saved2/requested.5 rejection,
same.5 acceptance, cold model/normalizer/RNG identity, actor action identity and
weighted AMP reward quarter, with empty optimizers/replay and0 updates/transitions.
The actual SDK-free CLI preflight also closed0. Guard peer and final source/loader
peer evidence are linked by the source freeze; old268 tests were not rerun.

Root plan `root_GPU_launch_commands_v2.json` keeps old commands/cwd/env/watch
layout plus the complete final source-manifest artifact. The only GPU phases
are freshpilot4 and independently fresh2000; neither has--resume. Pilot4 uses
32768 transitions, full2000 uses16384000, each with256 actors×32 rollout steps.
The watcher targets500/1000/2000 and uses CUDA_VISIBLE_DEVICES=''. Root alone
starts the single GPU process. CPU preparation does not claim any future phase
has executed, any checkpoint has passed, or any style quality has improved.

The unchanged watcher checks serialization, payload presence, budgets, recipe
and dimensions; it has no old transferred=False/AMPcount=transitions gate and
does not audit reward weight. It publishes snapshots with the original file
mode. Root must observe real process closure, verify bytes/nlink, chmod selected
snapshots0444 and use the separate new full-payload auditor before evaluation.
No mutablelast.pt may serve as evaluation admission; missing milestones remain
missing in the ledger and do not pass.

History is retained: first prepared bundlev1 (`f3c2…`) was never trained. Its
inherited prose incorrectly described prior initialization as the sole new
mechanism and AMPweight2.0; v2 corrects that metadata in a separate bundle.
First actual CPU check failed when a relative bundle path was resolved after
the launcher changed cwd. All numerical assertions had completed, but report
publication failed; the failed import ledger and first failure report remain.
The docs helper uses absolute paths and the final evidence was actually executed
against v2. Initial v1 cold success is not presented as final v2 execution.

No default promotion. No GPU/Kit, optimizer update, physics rollout, future
checkpoint or sealed34 output was consumed by this preparation.
