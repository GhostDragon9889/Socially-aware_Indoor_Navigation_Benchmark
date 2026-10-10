# Native contact-point velocity: available data and proposed strict contract

## Conclusion

The pinned SDK can supply **actual filtered contact positions, normal loads and normals**, allowing a rigid-body velocity estimate at each reported contact position. This removes the angular lever-arm omission in the current body-origin tangential-speed proxy and can distinguish ideal rolling from translational sliding. However, the native tensor API does **not** return solver slip velocity, contact-event timestamps or persistent point IDs. The defensible label is **completed-step rigid-body velocity evaluated at native contact positions**, with contact geometry time explicitly unspecified. An exact solver-instant slip claim is unsupported.

This is a read-only source/API study. No stable monitor, capture, probe, world, runner or training source was modified. Detailed native contact-point capture has not yet been tested in Kit; the successful v3 test validated force matrices and COM channels, not these detailed buffers.

## Verified installed API

The decisive source is installed `omni.physics.tensors-110.1.11+110.1.1/.../api.py`, SHA `b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903`, lines 4986–5147. The published 108 reference supports the same buffer layout. Its prose incorrectly says five returned normal-data arrays; installed code and the documented example actually return **six**. [Official buffer reference](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidContactView.get_contact_data).

| Getter | Returned tuple, actual component shapes |
|---|---|
| `get_contact_data(dt)` | normal force `[C,1]`, position `[C,3]`, normal `[C,3]`, separation `[C,1]`, count `[S,G]`, start `[S,G]` |
| `get_friction_data(dt)` | friction force `[C,3]`, friction position `[C,3]`, count `[S,G]`, start `[S,G]` |

Float buffers are float32; counts/starts are uint32. `C=max_contact_data_count` is the view capacity. Filters and positive capacity are required. For pair `(s,g)`, only `[start, start+count)` is valid. `dt` converts impulses to forces in the backend: pass the actual **1/120 s**, and do not divide returned force by `dt` again. Friction samples have their own positions/counts; they cannot be joined to normal samples by array index. [Official normal/friction API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidContactView.get_friction_data).

### Critical shared-buffer hazard

Both getters use the **same count and start buffers**. Calling the friction getter overwrites metadata previously returned by the normal getter. Clone all normal outputs, particularly counts/starts, **before** reading friction. The pinned Lab sensor expressly stages normal metadata before the friction call; it also checks allocation pointer stability. This behavior is source-confirmed, not speculation. [Pinned official Lab implementation, lines 477–518](https://github.com/isaac-sim/IsaacLab/blob/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py#L477).

A CPU harness executed only the real installed Python `RigidContactView` class with an explicitly fake NumPy backend/frontend. It confirms the exact six/four tuple lengths and same-object alias: normal count/start **2/1** become **1/4** after the friction call, while their clones stay **2/1**. This validates Python ownership semantics, not native contact physics. Unused slots deliberately contain NaN; active slots remain finite. Evidence: [sdk_contact_point_velocity_contract.json](sdk_contact_point_velocity_contract.json).

## Proposed acquisition interface

Use a new optional capture component; preserve the stable existing force monitor. Suggested boundary:

```python
sample_completed_substep(
    dt, *, episode_ids, physics_step,
    foot_link_pos_w, foot_link_lin_vel_w, foot_ang_vel_w,
    exact_static_floor_paths,
) -> ContactPatchSample
```

The component receives an already verified existing native floor view. It must share the same post-physics hook as the force monitor, with no extra `step`, `forward`, renderer update or reset. Reuse initialized Lab-owned access. [Official Lab native access lifecycle](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/native-physics-api/physx.html).

Operational contract:

1. Verify each sensor path exactly matches one actor's `L_Ankle`, `L_Toe`, `R_Ankle`, `R_Toe` rigid path. Resolve actual `sensor_paths` and per-row `filter_paths`; do not assume multi-actor body-major/actor-major order or reshape it into a guessed order. Filter only explicit enabled floor colliders, excluding wall/peer/self. Keep their exact native-to-common bijection.
2. Record the current episode IDs, control/substep indices and public submitted-step counter before acquisition. This must be a newly completed step in the same episode. A reset invalidates old contact data until another actual physics step completes.
3. Immediately convert and `detach().clone()` all six normal buffers. Only then clone the four friction buffers if requested. Warp `.torch` / `wp.to_torch` conversion must occur before checking vector-component shapes. Getter outputs are refreshed in place; retaining a view is not an archival snapshot.
4. Promote count/start arithmetic to int64. For **positive** counts, require a bounded slice `0 <= start < start+count <= C`, and prohibit overlapping active slices between pairs. For zero count, ignore the start field entirely: it may be stale or a sentinel. Ignore all unused float-buffer slots, even if stale or NaN. Validate finite forces/positions/normals/separations only on active slices. Keep near-unit-normal checks explicit; invalid normals fail rather than being silently accepted.
5. Record capacity and active counts. The reviewed interface exposes no public overflow flag. Treat `sum(counts) >= C` as possible capacity saturation, rather than proving completeness. Fail or mark the sample incomplete and explicitly increase capacity for a new run. Reconstruct `sum(f*n)` per pair and compare with a same-step normal force matrix using declared absolute/relative tolerances; mismatches invalidate the detailed sample. This cross-check is necessary, but alone cannot prove that missing zero-load geometry was not truncated.
6. Require the floor to be static, with no moving/kinematic ancestor or enabled surface-velocity material. A stationary collider on a conveyor is not a zero-velocity contact surface. Dynamic ground needs its own matched point velocity; do not subtract fabricated zero.
7. Clone actual completed-step body data in the same world frame, including angular velocity. The pinned manager initializes tensor subspace roots at `/` (`physx_manager.py:799`); a future transformed subspace needs an explicit frame conversion. Use **link-origin velocity with link-origin position**, or **COM velocity with COM position**. Installed `get_link_velocities()` explicitly reports its linear part at COM; pairing it with link-origin position would be wrong.
8. Check the submitted-step counter after every read group. It must be unchanged. Emit contact geometry timestamp **None / unspecified**, body state time convention **completed-step**, and `exact_solver_contact_velocity_observed=False`. Four substep samples remain separate before control-rate aggregation.

The above is a proposed fail-closed protocol; the new detailed contact getter still requires a real SDK probe before integration is labelled validated. Saving compact active ragged slices plus original pair/count/start metadata avoids recording `C` unused entries on every 120 Hz step.

## Derived velocity and load-weighted statistics

For a reported position `p`, consistent reference point `o`, velocity `v_o`, and world angular velocity `ω`:

```text
v_foot(p) = v_o + ω × (p − o)
v_relative(p) = v_foot(p) − v_floor(p)
v_tangent(p) = v_relative − n (n · v_relative),  |n| = 1
```

For the current static flat floor, `v_floor=0`. Normal direction sign does not affect tangent projection. Load-bearing floor support additionally needs positive upward normal force; zero-load geometry may exist inside collision contact offsets and must not become support merely because count is nonzero.

Recommended derived outputs per foot/substep:

- Native normal sample positions, force, normal and separation; valid sample count and semantic pair identity.
- Load-weighted **mean of tangent-speed magnitudes**, `sum(w*|v_tangent|)/sum(w)`, weighted RMS, and loaded-point peak. Mean vector and its norm should be recorded separately: opposite patch velocities can cancel while both are sliding.
- A force-weighted contact-position centroid and its lever arm relative to the foot origin/COM. On the declared flat floor it is a solver-sample pressure-centroid estimate, not continuous pressure distribution ground truth.
- Separate contact/no-contact and support validity masks. Empty or zero-load groups are **invalid/None** for slip, not successful zero slip. Temporal summaries must report the supported-substep denominator and their time/load weighting.
- Optional friction force and anchor position as an independent measurement. Do not use friction magnitude as normal weight or match friction/normal points by index. Work using completed-step velocity is a labelled approximation, not exact solver dissipation.

For ideal rolling, origin height 0.1 m, velocity `[0.5,0,0]`, angular velocity `[0,5,0]`, and point `[0,0,0]` give **zero** point tangent speed despite origin speed 0.5 m/s. Removing angular velocity gives 0.5 m/s sliding. The CPU calculation also verifies the equivalent COM formula and shows why averaging speed magnitudes is required. These analytical cases are included in the evidence JSON; they do not establish rolling behavior of the learned actor.

Do not finite-difference contact positions or their centroid across frames as a material slip measurement: sample identities can change and a rolling contact patch can migrate over stationary material points.

## Timestamp and solver limits

PhysX's broader contact-report API distinguishes impact/event pose and pre-/post-solver velocities through optional extra-data streams. Those fields are not exposed by the reviewed `RigidContactView` tuples. Normal contacts and friction anchors also have distinct solver roles. This is why an exact instant cannot be recovered merely by calling detailed tensor getters after `fetch_results()`. [PhysX extra contact data and friction anchors](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/AdvancedCollisionDetection.html#extra-contact-data).

The point-based estimate evaluates completed-step kinematics at the latest native contact geometry. Geometry may correspond to a solver contact configuration rather than the end pose; its exact generation timestamp remains unknown. A future real static/sliding/rolling probe should quantify force reconstruction, lever arms and timing sensitivity at 120 Hz. No SDK-private timestamp or C++ introspection fallback is proposed.

The previous navigation body-origin proxy value (~0.726 m/s) cannot be retroactively corrected without recorded contact points. A larger proxy may reflect rotation, true slip or both; neither this source study nor a formula change establishes improved gait.
