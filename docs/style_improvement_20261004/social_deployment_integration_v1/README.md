# Social policy evaluation and deployment boundary

Source-only integration review and actor snapshot preparation. No Torch/NN/SDK
import, weights/data loading, test, simulation or training occurred here.
Production185 and sealed stages are unchanged. This page identifies the
smallest remaining work after training; it does not claim deployed paper effects.

## Implemented source API

```python
actor = FrozenSocialAMPInference.from_trainer(trained_social_trainer)
prediction = actor.act(raw_self_B357, raw_task_B12308, deterministic=True)
# Pass prediction['applied_actions'] to the physical decoder once.
actor.describe()
```

[inference_adapter.py](inference_adapter.py) imports stdlib only. It accepts the
already constructed, quiescent exact SocialAMPTrainer81a, with positive update/
physical counters and original positive RMS counts matching those transitions.
It owns one tuple deepcopy of **encoder, actor, self RMS357, task RMS12308 and
canonical69 mask**. It seeds/moves/restores nothing and constructs no network,
critic, discriminator, optimizer or replay. Exact source/type/RMS predicates
reuse the sealed critic snapshot9875; its critic constructor is never called.

Inference accepts arbitrary positive B, including isolated B=1 and shared B=4;
the saved training actor_count remains provenance, not inference batch size.
Raw task is `[trajectory20,HWC64×64×3]`, normalized entirely with saved task RMS
before the CNN; encoded order is `[map256,trajectory20]`, giving633 policy inputs.
The actor head alone returns deterministic Gaussian means; applied actions are
clamp[-1,1] times the57-active mask. No Normal, critic/value, PD decode, moment
update, or planner runs. Inputs must already share the snapshot's explicit
float32 device; all retained modules remain frozen/eval with float64 RMS.
Grid center, grid heading, height reference and include_ego are captured from
the trained configuration and must also govern the external producer.

**Checkpoint loader remains unimplemented.** There is no from_checkpoint or
ONNX export in this slice. A later loader must admit exact social format
`local.neutral_smpl.social_CNN_PPO_AMP_training.v1`, its complete source/config
contract, immutable SHA/stat identity and restricted CPU/mmap payload. It must
strictly validate all declared model keys/shapes/dtypes and moments before
retaining the actor path. If it temporarily constructs SocialAMPNetwork for
validation, record that NN construction and preserve caller CPU RNG; discard
critic/D afterwards. Calling SocialAMPTrainer.load instead constructs training
networks, Adam/replay and seeds through the derived original constructor.
Positive counters prove neither completed training nor motion quality.

## Concrete compatibility gaps

| Existing entry | Source-bound incompatibility | Small independent adaptation |
| --- | --- | --- |
| `runtime/checkpoint_contract.py:78`, `amp_inference.py:103,201` | Old format and357/1044 MLP input; PaperAMPNetwork has613 actor inputs | New social-only snapshot/loader; retain strict old loaders unchanged |
| `benchmark_paper_motion_tracking.py:191,327,417` | PaperAMPTrainer.load, old world.observations32-height task, isolated N=1 physics | Derive evaluation entry changing actor admission and raw social observation producer only |
| `train_paper_amp.py:134,359,747` | Actor factory cannot replace old observations; reports also assume legacy RNG preparation/profile | Derive pure evaluation entry with explicit social producer/config metadata and fresh provenance |
| SharedGroupWorldb608:167,279,373,401 | Exact train mixed curriculum, contact50 termination, whole-group reset after archival | Keep as training driver; separate reference/navigation lifecycle rather than silently reusing it |
| `dynbench_scene.py:408` | Existing native injection requires filter_collisions=False, disables replication; b608 requires bothTrue | First use its existing single-world binding; grouped native scene wiring is a separate pending SDK adaptation |
| `deployment/export_tracker_onnx.py:202,337` | Tracker resolved_configs/`_actor` layout and its tracking observations | Future social actor-only export with own RMS/raw-layout metadata; no tracker weight conversion |

## Preserve the original full22 and four-person comparisons

**Full22:** the thin derived benchmark can retain the original AMPWorld N=1,
named physical decoder, per-clip source-frame9 reset, ten true future XY samples
at+.5..5s, endpoint command hold, complete post-physics frames, no mid-clip fall/
timeout reset, same SO(3)/body metrics and inclusive22 success denominator.
Replace only actor loading and task assembly using the frozen6975/234e producer,
one stable actor ID/scene ID and the selected training grid/reference/include_ego
contract. Do not add three peers to satisfy training actor_count. Preserve all
failed/idle/initial rows, native/reference quaternions and original clock fields.
Singular fallen task headings require explicit same-episode continuation and
recorded fallback/actual quaternion; do not change physical rotation metrics.

**Fixed four-person200s:** retain the registered scene/routes, seeds, Diffusion
guidance/replan/noise schedule, continuous reference adaptation and6000 controls
of the old shared-world physics loop. Assemble raw social observations from all
four current COMMON states, stable distinct actor IDs and a single shared scene
ID; retain source control phase and raw/adapted target distinction. Reuse actual
peer/static monitors and their original threshold/measurement provenance.
b608's all-partner nonfoot normal50 predicate is a different channel; substituting
its termination/whole-group resets changes this comparison. The old reader's
frozen checkpoint/source admission cannot certify new runs: use independent
social metadata admission while reusing the existing metric mathematics,
complete captures, failures/resets/idle and primary reference without refitting.

**Nav native:** retain the admitted source USD/map/floor and pre-context query
hook, exact dynamic robot roots and original world coordinates. Existing
provider.height_at supplies floor or conservative floor+1.8 for blocked/unknown
cells, a policy feature rather than true obstacle tops; explicitly declare this
sampler choice before applying the social height reference/+1.7/clip×5 once.
Do not call the provider's bounded native_surface_height_at a full-top sampler,
or treat social owner/rectangle proxies as collision truth. Current standalone
grouped Scene factory/World cannot consume this provider unchanged.

This source slice therefore connects **an in-memory trained actor** only.
Independent checkpoint inference, derived22/200s/native producers and captures,
full native grouped lifecycle, ONNX/export and actual model/physics validation
remain to be implemented and executed. Source review is not a test receipt.
