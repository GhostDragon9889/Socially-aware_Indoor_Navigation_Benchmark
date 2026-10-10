# Actual native Warehouse first probes: independent CPU review

Both parent-executed GPU runs completed all 150 controls. This reader started no SDK/GPU and changed no run artifacts or runtime source. Each report passes **492** independently checked source/raw contracts, including all 154 archived source files, 150×4 actor-controls, 600 shared physics substeps and 2,400 actor-substeps. Actual completed-capture counters run consecutively from 0 to 600; none of the 150 controls is padded, dropped or treated as a successful route merely because the process exited successfully.

| Actual five-second run | Falls | Reached vertices | Static actor-controls /600 | Peer unique-pair controls /900 | COM recomposition max |
|---|---:|---:|---:|---:|---:|
| Original .35m clearance/spawns | 3 | 0 | 78 (13%) | 0 | 1.90735e-6m |
| New .8m clearance and new spawns/routes | 0 | 0 | 0 | 0 | 9.53674e-7m |

The actual artifacts and full-array dtype/byte hashes are in [actual_first5s_audit.json](actual_first5s_audit.json) and [actual_clearance08_first5s_audit.json](actual_clearance08_first5s_audit.json). All raw floating fields are finite; floor-filter normal forces and same-instant body COM are real captured channels. Rig names, 69 DOF limits, joint control, common masses and live native ordering exactly match the loaded baseline actor's saved physical contract. Selected AMP/Diffusion bytes are independently rehashed as `62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b` / `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`.

## First-run failure timing

All three falls belong to actor2 and satisfy pelvis-height <.45m above the declared physical floor0; none triggers the root-up tilt threshold.

| Control /time | Episode length | Pelvis height | Root-up Z | Static contact at fall |
|---|---:|---:|---:|---|
| 94 /3.1333s | 94 | .438205m | .605979 | No |
| 112 /3.7333s | 18 | .449892m | .576419 | Yes |
| 135 /4.5s | 23 | .448862m | .959494 | Yes |

Actor2 contacts authored static geometry from frame4 (.1667s), often through frame52 (1.7667s). Its first fall occurs at frame93 after a contact-free interval; no static contact is recorded in the preceding 16 controls. After reset the second episode collides during frames97–111. The maximum actor2 static force is 5,845.04N at frame133. Actor1 has 10 earlier static-contact controls and no fall; actors0/3 have no static contacts or falls. This timing makes an environment-local collision/controller/command or feature-distribution investigation reasonable, but does not identify a cause.

There is **no observed through-floor collapse**: pelvis minima are [.87265,.72520,.43820,.88526]m. The floor-only channel records substantial nonzero ankle/toe loads. Original no-support substep fractions are [.00833,.11333,.33833,.04], with actor2 losing support far more frequently. The floor force matrix excludes static obstacles/peers and uses the single declared `/__PaperMapProbeReferenceGround` filter. COM geometry/positive masses/common order and force/support reductions are independently recomputed, rather than accepted from the summary.

## New bounded observations and interpretation

The second run uses clearance .8m **and** different origins/routes, so it is not a single-variable clearance ablation. Pelvis minima remain [.86134,.86642,.82508,.86128]m, with within-episode distances [5.69395,4.72173,5.87120,5.97939]m. No-support substep fractions are [.00667,0,.005,.00167]. It has no static/peer contacts or falls within five seconds, but no reached vertices or completed cycles. It does not establish 30/200-second stability, gait quality, safe clearance, or author-method equivalence.

Both use strict deterministic Torch with the new four-corner SDF sampler and actual clean-planner inference. No global deterministic fallback was invoked. PhysX/Warp bitwise reproducibility remains explicitly unguaranteed.

## Remaining observed-evidence gaps

The first two runs save observed force/COM/counters and source-bound material/config descriptions, but do not retain a fresh stage floor inventory/all-hit ownership record, App/context historical construction counters, or every height query's event row. Thus source-reviewed single-world injection and the original native resource probe are distinguished from newly measured runtime ownership. Native engine time is unavailable and remains null; Lab submitted-step counters are not renamed native time. The newly authorized live-verification hook is a separate next source epoch, not retrospectively claimed by either five-second artifact.

An initial second-run audit had a generic failure status label despite zero raw falls. That erroneous label-only report is preserved as [actual_clearance08_first5s_initial_status_label_error.json](actual_clearance08_first5s_initial_status_label_error.json); the corrected audit distinguishes bounded zero-fall observation from navigation success. No physical array changed.
