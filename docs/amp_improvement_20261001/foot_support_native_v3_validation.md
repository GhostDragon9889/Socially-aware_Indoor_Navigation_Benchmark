# Native feet-floor v3 validation: sensor channels pass, fixed-PD stance fails

The actual standalone v3 run completed **all three phases, 68 controls / 272 physics substeps**, then correctly exited with failure. **Filtered floor/wall force and live COM channel criteria passed. Fixed-PD no-fall stance failed. The original combined `native_physical_probe_passed` remains false.** No business source or criterion was changed during this CPU audit.

## Results independently recomputed from native captures

| Phase | Controls / substeps | Filtered floor force max¹ | Foot-wall normal force max | Live COM error max² | Sensor criterion | Original complete criterion |
|---|---:|---:|---:|---:|---|---|
| Standing | 60 / 240 | 810.0793 N | 0 N | 4.76837e-7 m | PASS | **FAIL: one fall control** |
| Airborne | 4 / 16 | Exactly 0 N | 0 N | 4.76837e-7 m | PASS | PASS |
| Airborne wall | 4 / 16 | Exactly 0 N | 573.0038 N | 9.53674e-7 m | PASS | PASS |

¹ Maximum absolute vector component as defined by the original QA criterion; the standing maximum sum of four foot-vector magnitudes is 869.9858 N. These values are filtered normal-vector sums, not scalar sums of per-patch normal forces or friction.

² Original synchronized native/float32 composition error. Independent float64 NumPy recomputation gives 2.83987e-7 / 2.78681e-7 / 4.75629e-7 m. Mass-weighted native COM independently agrees within 1.26e-6 m.

Standing last-20-control side support fractions are **[1, 1]**. Nevertheless, control **60** has root height **0.4460741 m**, below the unchanged 0.45 m threshold. Its up-Z is 0.3273296 (70.89° tilt); the separate up-Z < 0.1 termination criterion was not crossed. At control 40, COM X = 6.1996956 m exceeds even the projected all-foot collider front X = 6.1398649 m. At the final control, support has shifted to toe forces while the body falls. This is real fixed-PD instability; foot contact presence does not imply balanced stance. The projected collider AABB is a geometric proxy, not an active contact-patch support polygon or measured COP.

The explicit candidate started at root height **0.9377000 m** with right sole clearance 1 mm / left 2.8 mm. Initial reset geometry agrees with the verified neutral bind within 4.77e-7 m for standing/airborne and 1.91e-6 m for wall placement. Each phase has one initial reset, zero initial velocity, then a constant PD target. The declared static gravity/support estimate and actual native stiffness successfully produced bounded fixed targets, but **did not establish a stable stance**. No further stance-controller fix or gate relaxation is part of this report.

## Native mapping and temporal contract

Actual foot sensor order and complete exact paths are captured in the JSON report. They form a verified bijection with:

- `/World/envs/env_0/Robot/Geometry/Pelvis/L_Hip/L_Knee/L_Ankle`
- `/World/envs/env_0/Robot/Geometry/Pelvis/L_Hip/L_Knee/L_Ankle/L_Toe`
- `/World/envs/env_0/Robot/Geometry/Pelvis/R_Hip/R_Knee/R_Ankle`
- `/World/envs/env_0/Robot/Geometry/Pelvis/R_Hip/R_Knee/R_Ankle/R_Toe`

The floor view's four native filter rows each contain only `/World/ground/terrain/mesh`. The independent wall witness filters only `/World/PaperValidation/Obstacle_0/geometry/mesh`. Its 573 N normal force alongside exactly zero floor force demonstrates that wall contact did not contaminate this floor channel. Neither obstacles nor peer forces are counted as floor support.

The existing public `SimulationContext.physics_sim_view` is identical to the actual registered manager's `get_physics_sim_view()` result. Its runtime owner is recorded as the `isaaclab.utils.string.ResolvableString` wrapper. There was no legacy singleton, guessed private attribute or new SimulationView fallback.

Submitted-step counters are exactly **1–240**, **241–256**, **257–272** for the three phases. Every control has exactly four valid 120 Hz substeps. Before/after physical/contact read counters are equal, no phase reset advances the physics counter, and each phase's episode ID is constant (2, 3, 4). Fall flags match the original height/tilt predicate; no time limit occurred. All numeric physical/force arrays are finite. Body rotations are unit quaternions, masses are positive, native sensor-row reordering and per-foot/side contact masks match raw forces.

**83 independent NumPy integrity checks pass**, with no SDK or CUDA imports. These checks verify saved physical evidence and the report's classification; they do not turn the failed stance result into PASS. The counter is a submitted Lab step counter; a native PhysX engine timestamp was not observed. There is no PhysX/Warp bitwise determinism guarantee.

## Frozen artifacts and scope

| Artifact | SHA-256 |
|---|---|
| Actual `result.json` | `a93951b2b67586720d8a25a50c4dd501a4944b1932a5e86091c06d30808fa94d` |
| `standing.npz` | `0b200f2c356ad15ff1e12e0655eb65fab00adf110bf9d47028535348003d824b` |
| `airborne.npz` | `21a9d8c76f559eac7eb702969f5456b1219d54a7ffd7058beccc142da4d4e237` |
| `airborne_wall.npz` | `1bcf0f0ab77e0b0d3499fe68abdedb3464a8409904996eb195350b80345cfdb7` |
| Actual probe driver | `0ada325dffa97d353da0df2f2ac49fee0b6d215f328f4593f43e5d183a005f02` |
| Stable foot monitor | `8c9cb88ddcbe55599ebe79abf80ea7a77a85eedfaa17324b937a862b284267e8` |

All five captured business-source hashes match their current files at review time; the complete source manifest, metadata, per-phase checks, numeric errors and COM/footprint snapshots are in [foot_support_native_v3_validation.json](foot_support_native_v3_validation.json). The actual raw run remains under `output/diffusion_amp/foot_support_native_qa_v3`; v1/v2 failures remain preserved.

Recompute the CPU audit with [audit_foot_support_native_v3.py](audit_foot_support_native_v3.py):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/amp_improvement_20261001/audit_foot_support_native_v3.py
```

This bounded test supports using the exact filtered floor/COM measurement channels in later learned-AMP evaluations. It does not validate learned gait quality, long-run navigation, contact-point slip, paper locomotion success, or a stable static controller. Body-origin tangential velocity remains a clearly labelled slip **proxy**; normal contact and foot rolling cannot be disentangled by that proxy alone.
