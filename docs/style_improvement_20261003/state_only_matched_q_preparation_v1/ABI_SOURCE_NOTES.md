# q935 matched state-only D preparation

Status: source-only prepared. No bundle, model read, SDK, GPU, training or future checkpoint claim. Root decides execution after the ongoing q full evaluation.

## Exact mechanism

Parent is the frozen q935 recipe (`2e14550a…`, profile `bcbc8942…`), with the same stratified expert sampling, reward scale2 and outer weight2, public actor/critic, fixed-half commands, protected terrain, seed7 and fresh2000 budget. Only three training fields change:

```yaml
discriminator_input: state
warm_start_discriminator: true
warm_start_action_discriminator_from_state_prior: false
```

The existing `PaperAMPNetwork` builds D as1950→1024→512→1. The existing warm-start function strictly copies the complete public D and1950 AMP history mean/variance/count, then uses the same count cap200000 (public AMP count15482880001). Policy self357/task1044,69 action outputs with57 active coordinates, value RMS and terrain preservation remain unchanged. This is a source-derived cold-load contract; no public weight tensor was loaded here.

## Necessary staged runner change

The current runner automatically uses post-step history for state D. That would combine action removal with a history timing change. The independently staged runner instead computes reward from the same h[t] before `world.step`, retains the same rollout/replay history, removes the post-step overwrite and records the correct pre-step timestamp. Three exact fragment substitutions reverse to the entire parent runner bytes and AST. No PPO, loss, gradient penalty, replay, expert sampler, AMP feature, normalization or physics function was rewritten. The original frozen runner and production PACER state-only defaults are untouched.

The actual isolated config loader accepted the candidate with seven owned imports from q935 and no Torch/SDK imports. The executed AST check observes exactly one loop history assignment and one reward assignment before physics, with no state-specific post-step replacement. Ruff F passed.

## Retained data ABI and interpretation

History remains batch×10×195, newest first; history-coordinate RMS has1950 entries. Existing `_amp_inputs` still validates batch×69 finite normalized action arrays with12 frozen coordinates zero, and expert batch/replay retain these arrays. State D ignores them in logits, BCE and gradient penalty. Their construction is retained to isolate the discriminator mechanism and keep loader/RNG interfaces. q expert sampler configuration and saved private RNG schema remain exact.

Removing69×1024 random initial D weights changes constructor RNG consumption. Same seed and complete saved RNG restoration do not imply paired bitwise stochastic samples or physics. D reward state history and expert-window semantics remain the q pre-step convention.

This is an action-input mechanism ablation relative to NavIsaacLab Eq2 and a PACER-compatible state D adaptation. It is not a strict NavIsaacLab final model or an author training reproduction. It does not add future reference poses, latent style conditions or modify the reward.

## Concrete execution dependencies

Use fresh training with no `--resume`. The original q runner uses saved config on resume and does not compare requested D fields; a2019-state-action checkpoint must never be used as this1950-state branch. Future same-branch resume requires its saved state config. No generic resume guard was added during this preparation.

If root later executes, the existing q complete-payload auditor needs a small identity/shape adaptation: full state D1950 and `warm_start_discriminator=True`, direct public-D/RMS transfer provenance, and no `state_prior_action_extension`. Existing model/optimizer/replay/RMS/RNG/terrain/q sampler checks should be retained. The old q reader hardcodes the action extension strategy and cannot attest this branch unmodified. No new reader or launch protocol has been built here.

The profile changes only the three training fields and descriptive initialization/profile/assumptions metadata. Dataset, reward section and every other training field are equal to the parent. Validation22 and sealed test34 outputs were not read.
