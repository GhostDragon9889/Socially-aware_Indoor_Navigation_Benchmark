# Current net-normal force sampler: source preparation

This isolated prototype passed one root-approved invocation of its four pure
Torch CPU mock tests. No SDK/native getter, NN, weights or physical termination
ran here. Production185, existing views/smoke, defaults and frozen proofs are
unchanged. Mock success is not evidence of native force shape or floor contact.

## Interface and ownership

`sample_current_net_normal(views, layouts, body_names_by_view, group_actor_rows,
bridge=..., device=..., physics_dt=..., physics_step=..., control_step=...)`
returns `(forces[N,24,3], metadata)` in COMMON body order and canonical actor
rows. Pass the existing smoke's per-actor views and layout dictionaries
(`actor`, `sources`), and a 24-name sensor-order tuple for each view. View
order may differ from actor rows; all actors and body paths must occur once.
Each native ordered `sensor_paths` must equal its declared sources before any
getter runs. The body/path association is supplied by the caller's reviewed
USD/name resolver; this function does not reopen assets or infer names from
width or path basenames.

Current prepared use is eight per-actor views / two four-person groups.
Group rows can describe another complete N divisible by four, but this does
not create native views, enlarge filter lists or prove larger SDK operation.
Future four-slot `[G×24]` grouped views require a separate exact-path/order
adaptation. No view factory or SimulationView is created.

The only force call is `view.get_net_contact_forces(dt=physics_dt)`. It reads
all contact partners, including floor, **independently of filter patterns**.
`get_contact_force_matrix` is never called; no filtered matrix sum, vector
magnitude, foot mask or four-substep maximum is used. Real Warp output is
lazily converted with `warp.to_torch`; a pure-Torch mock takes no Warp import.
Every returned view buffer is immediately detached and cloned, checked for
finite float32 `[24,3]` on the explicit device, then reordered by semantic
names. Four slot tensors pass through the sealed original actor-row bridge.

Use actual physics dt: the current runner uses **1/120 s**, not the 1/30 s
control interval. Native dt converts impulses to Newtons. The explicit finite
positive dt is caller responsibility; this function cannot discover actual
solver dt. Sample once at the current control end (after the fourth substep),
without advancing physics. Physics/control stamps are retained as declared
integers, not independently verified timestamps. All layouts are checked
before a getter; native failures may leave earlier reads completed. No writes,
rollback or transaction is promised.

`rows.tolist()` transfers group indices to the host, and boolean validations
may synchronize the device. Force vectors are not transferred to CPU; this
prototype makes no all-GPU performance claim and performance was not measured.

Call [the sealed nonfoot predicate](../nonfoot_net_normal_contact_staging_v1/README.md)
in another layer with the runner's real episode progress. This sampler does
not change terminated/truncated/reset flags, history, clocks or RNG and does
not claim that early termination or shared training is implemented.

## Fixed source semantics

- The installed native API is
  `/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.1.11+110.1.1.lx64.r.cp312.u7f4/omni/physics/tensors/api.py`,
  SHA `b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903`.
  Lines4918–4953 document all-partner scope, shape and impulse-to-force dt;
  `sensor_paths` is public at4891. The installed Lab3 ContactSensor declares
  its net force channel **NORMAL**; separate friction data are not added here.
- [Public API documentation entry](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)
  returned403 on this session's fetch. Getter semantics above were actually
  read from the installed source; this link is not a successful web-read claim.
- Original readonly `tensor_bridge.py`:6035B,
  SHA `2f5dc721f739d26ace02cc0831864523e605587f7e960f0fd32b82326ec8c9ec`.
  The supplied bridge must have the exact source location/bytes; no actor or
  semantic body conversion is silently delegated to it.
- Original observation tuple source:10438B,
  SHA `a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba`.
  Only `PACER_BODY_NAMES` is decoded with AST; observation/rotation/NN code
  is not imported by this sampler.

This is a local **net-normal/all-partner** adaptation. Equivalence to the
friction content or temporal aggregation of PACER's Gym contact tensor is
not established. No finite mock tensor can establish real native contacts,
collision filtering or force synchronization.

## Actual bounded mock checks

Four bounded tests check: eight reordered views plus one body permutation
through the true sealed row bridge; cloned output/RNG/input preservation and
no pair-matrix or Warp/SDK import for mocks; all layout/name failures before
the first getter; dt/stamp/partition/source-force failures and transparent
partial reads. No policy is constructed and no existing suite is repeated.
Root whole-source and independent source reviews found no required blocker.
The sole approved invocation completed directly: tool chunk `db516f`, exit0,
four tests in0.027s; shell timing1.015s and tool wall0.887031448s are separate
measurements. Source04d21596 and tests301abbe9 remained unchanged afterward.
[Actual command, output and scope](actual_CPU_checks.json) record that CUDA
remained uninitialized and Warp/SDK were not imported. Original source-only
AST/compile chunk `1a24f5` and Ruff chunk `649277` both exited0.
The completed command below is recorded for reproducibility, not another run:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/social_net_normal_sampler_staging_v1 -p test_net_normal_sampler.py -v
```
