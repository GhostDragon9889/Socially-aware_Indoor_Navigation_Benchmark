# Feet-floor v2 failure and explicit v3 stance candidate

## Actual v2 result remains FAIL

The actual native test in `output/diffusion_amp/foot_support_native_qa_v2` completed 60 standing controls / 240 physics substeps and failed its original no-fall requirement. Controls 59 and 60 had pelvis heights 0.423354 m and 0.345678 m, below the unchanged 0.45 m termination threshold. The final root tilt was 77.18 degrees; the tilt termination threshold itself was not crossed. Two true fall controls must not become a successful standing result simply because feet still produce floor forces.

| Observation | Actual v2 evidence |
|---|---:|
| Body mass / gravitational weight | 63.313949 kg / 621.10986 N |
| Lowest initial collider sole with root at 1 m | 0.0633001 m above ground |
| First force over 1 N | Physics substep 13 |
| Peak sum of four foot force-vector magnitudes | 5998.62 N |
| Last 20 controls: left/right support fraction | 1 / 1 |
| Last 20 controls: mean sum of foot loads | 411.15 N |
| Maximum live COM composition error | 4.76837e-7 m |
| Wall forces during standing | Exactly zero |
| Monotonic read counters | Exactly 1–240; reads never advance them |

The first three control quaternions are exactly XYZW identity. At control 40, COM X is 6.21990 m, beyond the front of the projected all-foot collider AABB at approximately 6.13997 m; the body continues moving forward and downward. This is real loss of fixed-PD stance stability. Initial free fall and its impact are a concrete setup defect, but these data do not prove impact was the sole cause of the later collapse. The collider AABB is a geometric projection proxy, not an observed active support polygon or pressure COP.

The native view lookup and filtering did work. Actual `native_sensor_paths` match the four exact semantic ankle/toe rigid paths. All four `native_filter_paths` rows contain only `/World/ground/terrain/mesh`. The separate wall view contains only `/World/PaperValidation/Obstacle_0/geometry/mesh`. Native body ordering differs from common SMPL ordering and was explicitly mapped. No floor force was inferred from unfiltered obstacle/peer forces.

The v2 script stopped after this failed phase, so it did **not** validate suspended feet or wall-force isolation. Its original source is preserved in [foot_support_probe_v2_source.py.txt](foot_support_probe_v2_source.py.txt). Actual result SHA is `9f2505d764fdc7aa66578f191a623f0d2522dfcc1b2daf1647b3b8858b06bd42`; standing capture SHA is `d9f51ebd116bdf9d21526c9db6b4235fd72f1d48a7503277d964ed0b558a2965`.

Detailed CPU reconstruction: [foot_support_probe_v2_stance_diagnosis.json](foot_support_probe_v2_stance_diagnosis.json). This reads the real capture and the exact converted USD; it does not execute another physics test.

## v3 changes, explicit and unvalidated physically

[probe_paper_foot_support.py](../../scripts/probe_paper_foot_support.py:436) now offers two explicit options while retaining the old defaults (`neutral`, `fixed_height`, root height 1 m):

- `--standing-placement collision_grounded`: derive the initial root height from actual enabled collision vertices/cube corners in the converted neutral rig. Nested toe colliders remain owned by the toe, and transformed geometry is used instead of body origins. A 1 mm minimum sole clearance gives root height **0.9377000 m**, left sole clearance **2.8 mm**, and right clearance **1 mm**. A single initial reset must match the verified neutral USD bind offsets within 5e-5 m; mismatch fails closed.
- `--pose-anchor gravity_balanced_neutral`: keep the initial reset DOF pose exactly zero, then apply a constant fixed PD target offset estimated from a declared static two-ankle vertical support load and all actual body masses/local COM offsets. The whole floating-base wrench is checked before deriving child-axis joint torques. Native PD stiffness and effort limits are read through verified installed getters; semantic DOF names must form an exact native/common bijection. Required effort, normalized action and a 0.25 rad offset bound fail closed. Frozen toe/hand target semantics remain active. There is no online balancing feedback, extra actuator force, root fixation or pose write during a phase.

The candidate's maximum required joint torque is 14.8695 Nm, and its analytical floating-base force/moment residual is below 1.2e-13. The neutral template agrees with the first unloaded v2 physical substep's body offsets to 6.20e-7 m. These are analytical checks; they do **not** establish a stable physical stance. The assumed ankle support locations and loads are not measured contact COPs. Frozen joint biases are omitted and recorded. A true v3 run must pass the original no-fall criterion.

All three independent finite phases now complete after a criterion-only failure. An acquisition exception or nonfinite channel still aborts safely and retains the available raw prefix/partial rows. The final result reports `sensor_channel_criteria_passed` and `standing_no_fall_criterion_passed` separately; `native_physical_probe_passed` is true only when **all original criteria pass** across standing, airborne and airborne-wall phases. No criterion was relaxed.

## Verification and next actual run

Focused CPU tests: **55 passed in 2.24 s** (27 standalone probe tests + 28 monitor tests), with CUDA disabled and no Kit startup. Tests cover transformed cube geometry, nested rigid ownership, sole placement, a known 2 kg / 20 cm gravitational moment, floating-base wrench failure, actual-gain semantic reordering, neutral reset vs nonzero fixed target, phase continuation after a true mocked stance failure, and exception/missing-phase failure. These mock tests validate contracts, not native physics. Ruff passes for the two modified files. Actual metadata/header/XML preflight also passes without importing Torch or SDK; the package SHA was declared, not rehashed by that preflight.

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m pytest \
  tests/test_paper_foot_support_probe.py tests/test_foot_support.py -q
```

Parent-owned next native run (not executed by this CPU audit):

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/probe_paper_foot_support.py \
  --output output/diffusion_amp/foot_support_native_qa_v3 \
  --standing-placement collision_grounded \
  --pose-anchor gravity_balanced_neutral --deterministic-evaluation
```

Frozen driver SHA: `0ada325dffa97d353da0df2f2ac49fee0b6d215f328f4593f43e5d183a005f02`. Frozen standalone test SHA: `61848e0127809c83612bd497a51d2fb4d3b8a279a8886ceceb5c5d549d6a2f80`. The foot monitor source is unchanged (`8c9cb88ddcbe55599ebe79abf80ea7a77a85eedfaa17324b937a862b284267e8`). SDK-free static candidate evidence is in [foot_support_probe_v3_static_candidate.json](foot_support_probe_v3_static_candidate.json), CPU results in [foot_support_probe_v3_cpu_tests.xml](foot_support_probe_v3_cpu_tests.xml).

No policy, diffusion checkpoint, AMP training, shared runner, world, environment or historical result was modified. These finite contact-channel checks are not a paper gait or locomotion success benchmark.
