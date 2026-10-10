# NEW b563 / default62c4 / S1 variance fixture

Prepared only. The single method variable is `diffusion_guidance_mode: trace_clean → trace_clean_variance`. The existing `paper.py` branch multiplies the same clean-guidance goal/peer/scene gradient by its posterior variance. This is a local TRACE variance ablation; official clean-code default omits that scaling. No physics or model result is claimed.

Fixed source185/V12 (`9ddb…`), NEW DP `b563…`, default AMP `62c4…` original0664, S1/100-step DDPM/FP32, guide20/peer5/scene10, radii0.6/0.8m, alpha0/eta0, 15-control replan, four people/200s/6000controls/seed7 and the original fixture/routes/reference causal speed1.5/accel2/PD/contacts are unchanged. AMP training provenance stays historical OLD63e; real deployment NEW identity remains the existing strict planner-change ABI. Failed/idle/reset samples stay in the full original raw audit. No MMD or route count gate is used.

## Three executable sources

- `run_variance.py`: reuses the S10 external forwarded-argv pattern, substitutes only the mode token, and invokes the original185 driver/factory. Real SDK-free parser preflight was exit0 in0.572s; parent preflight differs only in mode/output and the explicit extension identity.
- `launch_registered.py`: uses the exact reviewed original `bc617…` main with metadata substitutions and removes the already-completed parent ordering block. Whole inverse bytes preserve Popen, wait/TERM/KILL, owner_error/cleanup_error and actual return-code/finally handling.
- `variance_evidence.py`: renders the parent NEW plan, keeps the original `96d9…` strict historical-vs-deployed planner guard and S1 runtime function unchanged, adds real producer mode/goal/peer/scene equality, and exposes raw CLI. It calls the exact `0c08…` raw body and `f851…` raw mathematics. Its source guard uses both actual `result.arguments.diffusion_guidance_mode` and `result.diffusion_guidance.mode`, plus the exact deployment entry/owner command. No missing mode observation is inferred from a label.

[commands.json](commands.json) contains exact arrays. Root alone runs GPU serially after reviewing these three files:

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/default_NEW_variance_preparation_v1/launch_registered.py --phase fixture_new
```

After the actual child closes0 and the owner seals complete captures:

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/default_NEW_variance_preparation_v1/launch_registered.py --phase fixture_new --cpu-audit
```

Output: `output/diffusion_amp/default_newDP_variance_fixture200s_v1`. Actual receipts/raw audit: `docs/style_improvement_20261003/default_NEW_variance_actual_runs_v1/fixture_new{_actual_launch,_CPU_actual_launch,_audit}.json`. No future output exists or has been read. Any fall/contact failure stays recorded. This packet does not register a future native run or promote defaults. Whole-source before/after and opaque model hashes remain runtime-owner checks; preparation did not deserialize or execute models, start SDK/GPU, or read34 outputs.
