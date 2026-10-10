# Owned Articulation SimulationView release, revision v9

This independent preparation derives from sealed v8 `smoke_group_physics.py` (49961 bytes, SHA256 `a9da29022b7f514ffb6d4034f152190cd1cbd302601542a1418fcc764c207ba3`). It adds a terminal instance compatibility release. Production, installed SDK files, v8 and prior failed runtime records are preserved.

## Actual evidence and source boundary

ROOT's v13 process exited 139 after the public scene-backend setter returned with both cache fields None. One existing SimulationView was still alive after GC, although its public validity was false; the eight contact wrappers were dead and Base/Core/Timeline cleanup flags passed.

ROOT then executed the separate target-only diagnostic v14 (`049247`, session 25082, completion `798bc4`, exit 139). Its sealed owner JSON is `../social_actual_runtime_v1/group_physics_backend_referrer_v14_actual_root_owner.json`, 11703 bytes, SHA256 `8a18c8abcd9934caeaeced6e462b5c3158798ee3e7eb1ab5285fdfc54593f902`. The diagnostic identified eight dictionaries with `_physics_sim_view`, owned by four Articulation and four ArticulationData instances. Diagnostic temporary lists are not additional production-owner evidence. These observations locate retained Python aliases; they do not establish that this revision fixes the native finalization crash.

The installed pinned Articulation initialization stores the current manager view in `_physics_sim_view` (`articulation.py:3859`) and constructs its data object (3880). ArticulationData stores the manager view too (`articulation_data.py:87`). Articulation invalidation clears `_root_view` (3976–3980), while these SimulationView aliases remain. Their exact whole-file pins are recorded in `source_derivation.json`: Articulation 224858 bytes / `2cbd53ccbebb7a7b038bed2b612b65e41d6674a4791beec68d84336d7bd00a87`; Data 95901 bytes / `6579e5b3efcd13d113cda141945cd055a4d0b86a069f0a80fc2c73c4cfe4163b`.

STOP callback ordering is not inferred from manager invalidation alone. AssetBase registers PhysicsEvent.STOP; the installed Lab subscription maps that event to a timeline STOP subscription, and Base.close also dispatches STOP. The helper records the actual `_root_view` presence and type after returned stop. It neither requires that view to be absent nor queries its native validity, count, paths or body names.

## Reusable helper and call order

`_release_owned_asset_simulation_views(owned_sim, assets, report)` accepts exactly four distinct caller-owned vectorized assets in a list or tuple. It looks up already loaded SDK modules and verifies their actual module paths, whole bytes/SHA, exact canonical classes, descriptor bindings and compiled function source origins. It imports no additional SDK module.

Before any instance write, it requires:

- The exact current SimulationContext singleton, a returned `sim.stop()`, and actual boolean stopped=True / playing=False observations from the pinned bound methods.
- Four exact Articulation objects and their four distinct exact existing ArticulationData objects.
- The canonical manager's existing exact scene backend and actual existing exact SimulationView.
- All eight existing instance `_physics_sim_view` fields to reference that same backend view.

Only then does it assign these eight **private fields on owned terminal instances** to None. It records actual counts separately for four Articulation and four Data writes, verifies all eight fields are None, and leaves the backend view intact for v8's existing public setter and weakref-death check. It does not write SDK files, class methods, Base state, `_root_view` or native buffers, and does not initialize or step physics. Strong native/object/class/method aliases are cleared in `finally`, including when validation or a write fails. Native-owner checks use direct loops so an exception cannot retain a separate asset/data generator frame; the temporary data list is cleared explicitly.

The original order becomes returned stop → this helper → original contact/asset container clearing and GC → original owned-manager normalization / context clear → original backend public setter / Core / Timeline cleanup → original post-context GC weakref gate → App close. SDK STOP dispatch during close remains part of its existing lifecycle. The revision does not promise identical teardown callbacks or successful native unload.

Successful helper flags are:

```python
report["owned_asset_simulation_views_release_returned"] is True
report["owned_asset_simulation_view_release_counts"] == {
    "articulation_fields": 4, "data_fields": 4
}
```

These are also required by the entry's passed status. A guard failure preserves false/partial counts and the cleanup error; no missing view, incomplete allocation or failed release is accepted. Actual process exit remains separately observed by ROOT.

## Exact preserved source and preparation status

Four recorded literal changes add the helper, call it after stop, initialize its flags and require them. Full inverse bytes and AST reproduce sealed v8. Ten original functions retain identical source bytes and AST, including all separated/overlap physics, force thresholds, snapshots, targets and the entire original v8 terminal helper. Only `_run` and `main` have the declared boundary changes.

The first reviewed source is preserved byte for byte as `before_native_generator_revision.py` (61945 bytes / `609992a7dbc79bebf42ac8a80e9b2badc51cd06068a08eda5d8284b3fc956d32`). The peer-required five native-owner generator removals and final list clearing are recorded in `native_generator_lifetime_exact.patch`; Python 3.12 compilation confirms no native-owner generator frame remains in this helper. The remaining generator reads only serializable ownership metadata.

Source-only construction, inverse/AST checks and stdlib compilation initially returned exit 0 in `55b02c`. Final full inverse, ten untouched-function byte/AST checks, compilation and generator-frame inspection returned exit 0 in `583aeb`; final Ruff `e5073d` returned exit 0. No candidate module, SDK, Torch, NN, checkpoint, dataset or array was imported/executed and no numerical tests ran. This source remains mutable until ROOT and independent peer review; v9 runtime repair, weakref death and real exit 0 are not yet demonstrated.
