# Four-Articulation actor tensor bridge

Root verified source2f5dc721/test116db45f/CPUrecord39ceb2ef after full reads and sealed all three readonly in actual tool`f55b13`. CPU results remain synthetic actor-row evidence; no body/DOF format conversion or physical scene is inferred.

Independent data layer: the sole approved CPU invocation actually passed **five tests in0.010s**, after independent source peer Required=[] and root whole-read. Tool`daeb79` returned exit0 directly (wall0.887s; no running session was returned). Source/test bytes stayed unchanged. [CPU checks](actual_CPU_checks.json) records their identities and the exact invocation. This module imports only Torch; it does not import Warp/Isaac Lab, construct Articulations, step physics, load models or alter production185.

## Actor rows and four slots

Canonical here means **actor batch order only**, not `StateConversion.COMMON`. The first128 of256 actor rows remain AMASS; the final128 remain procedural. `mixed_group_rows(256, device="cpu")` returns64 rows:

`group_actor_rows[g] = [2g, 2g+1, 128+2g, 129+2g]`.

Columns0..3 correspond respectively to prims `Robot_0..Robot_3`, exposed by the independent scene factory as asset keys `robot_0..robot_3`. Supply tensors in this slot order. Each asset has64 group environment rows; all four assets' row g belongs to the same cloned environment. The helper supports other positive actor counts divisible by4 using N/2, but does not create stable actor IDs, scene IDs or origins. **Actor-row indices are not stable identities.** The caller must keep existing IDs alongside the explicit map.

Every supplied `group_actor_rows` is int64 `[G,4]` on the explicit device and must cover every canonical row `0..4G−1` exactly once. Noncontiguous actor membership is normal; an explicit reordered group map is allowed only with the corresponding asset row order. Wrong width, duplicate, missing, negative or out-of-range membership is rejected.

| Function | Input → output |
| --- | --- |
| `assets_to_canonical` | Four same-shape float `[G,...]` fields → new `[4G,...]` in canonical actor rows |
| `canonical_to_assets` | One `[4G,...]` field → four new `[G,...]` fields in fixed slot order |
| `gather_asset(..., slot=...)` | One `[4G,...]` field → new `[G,...]` for integer slot0..3 |
| `actions_to_assets` | Exact `[4G,69]` → four `[G,69]`; preserves the entire69-coordinate order |
| `reset_group_env_ids` | Unique whole-group canonical actor-row request → four separately copied, identical ascending SDK group env-index lists |

State/action fields must be nonempty finite float32/64 with matching shape/dtype/device among the four assets. Trailing dimensions are opaque: root `[G,3/4]`, joint `[G,69]`, body `[G,24,3/4]` are examples, not validated physics schemas. No world XYZ/Z/origin offsets, quaternion permutations/normalization, body/joint name mapping, action clipping or PD conversion occur. Output dtype and values are preserved. Inputs and RNG stay unchanged; outputs have new storage. Explicit `device="cpu"` or indexed CUDA is mandatory; no implicit transfer or device selection. Scalar validation may synchronize GPU; GPU use/performance is untested.

## Reset boundary

An empty int64 reset list is valid. Duplicate/out-of-range rows or **any partial group** are rejected, never expanded. For a complete group request, all four assets receive the same group env indices, not the canonical actor indices. Future writes must use each asset's corresponding selected data `[K,...]`, or deliberately choose an SDK full-data API; a `[G,...]` full asset buffer is not silently treated as partial reset data.

The original separate group-reset primitive expands a member termination/truncation into a group reset and marks collateral truncations. This bridge instead accepts an already complete reset request and changes only its indexing. It creates no fall/termination flag and does not reset histories, curricula, plans, contacts or clocks. SDK/world reset integration and those coupled states remain pending.

## Locally inspected Isaac Lab 3 / ProtoMotions source interfaces

All items below were read and fully byte-hashed with ordinary CPU filesystem tools; no SDK was imported or executed. They are **source context**, not runtime dependencies or proof of a running64×4 scene. Installed checkout root is `/project/EXP/SangforAndSaut/IsaacLab-Protomotions`. The factory `source/isaaclab/isaaclab/assets/articulation/articulation.py` dispatches to the active backend; the relevant local implementation is PhysX.

- PhysX `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py` (95901B, SHA`6579e5b3efcd13d113cda141945cd055a4d0b86a069f0a80fc2c73c4cfe4163b`): lines48–61 document timestamped **pull-to-refresh** and stable ProxyArray views. `root_link_pose_w`708 and `body_link_pose_w`824 are world link/actor poses with **XYZW** orientation. Link velocities725/843 are world velocities of link frames, not COM velocity. `joint_pos`1040/`joint_vel`1054 are `[G,J]` buffers. Root/link subfields1223–1275 and body subfields1327–1384 have Torch shapes `[G,3/4]` and `[G,B,3/4]` after access.
- Base `source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py` (72567B, SHA`629379d8fb61a1651e1279573bb3576b8b8c8d1b42135995eb6a23c658329fdb`): root/body position and quaternion aliases1322/1328/1370/1376 select **link** fields, whereas `root_vel_w` and `body_vel_w` aliases select **COM** velocity. Use explicit `root_link_lin_vel_w`, `root_link_ang_vel_w`, `body_link_lin_vel_w`, `body_link_ang_vel_w` for the current ProtoMotions contract.
- `source/isaaclab/isaaclab/utils/warp/proxy_array.py` (13884B, SHA`5d20232ff07cb4d333b8f3e9e68b152568b884e935a1f672bf864ce3942937df`): `.torch`123–151 is a cached **zero-copy** `warp.to_torch` view; `.warp` provides the underlying array. A future SDK adapter must fetch each property at its current timestamp and clone its Torch view before a refresh/physics step can overwrite shared storage. The bridge inputs are already Torch snapshots; it does no Warp access. Merely retaining a previous `.torch` tensor is not a new timestamped snapshot.
- PhysX `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py` (224858B, SHA`2cbd53ccbebb7a7b038bed2b612b65e41d6674a4791beec68d84336d7bd00a87`): `write_root_link_pose_to_sim_index`452 takes XYZW pose `[K,7]`; `write_root_link_velocity_to_sim_index`803 takes link velocity `[K,6]`; `write_joint_state_to_sim_index`906 takes joint position/velocity `[K,J]`. All take explicit `env_ids`; `full_data=False` expects partial data. Position targets2498 fill buffers and require `write_data_to_sim` to apply. Current legacy root-link/joint wrappers4911/4929 still delegate but are deprecated. Target values require the simulator joint ordering and SDK float32 contract; this bridge's float64 support does not change that contract.

Current project source inspected and pinned:

- `protomotions/simulator/isaaclab/simulator.py` (53687B, SHA`207ff0cbfb65fd48092e73e36c97b2e1b8e2ab55cae79bc2e7edf562528e2ad5`): getters752–779/817–828/936–955 convert data via `wp.to_torch(...).clone()` for body/root **link** fields and joints. `_get_sim_body_ordering`715–740 derives semantic joint names from the actual backend names and kinematic map. Writes690–697 use root-link and joint state plus position targets. New four-asset wiring must validate names and use per-asset maps; a69-width tensor alone does not establish joint-name compatibility.
- `protomotions/simulator/base_simulator/simulator.py` (68053B, SHA`6b080dd1b63885e34fcad523b8468ab6b63e838a3eaac765e7a230a2ca3ac782`):245–282 build name-based body/DOF maps; root/body getters979/1023 convert to common; reset770 converts back to simulator; action1333 uses `dof_convert_to_sim` before applying PD targets.
- `protomotions/simulator/base_simulator/simulator_state.py` (34638B, SHA`31cf14fab319151602f43fcf6ad2751b71088cda3a3f4feefadfa61262102731`): `DataConversionMapping`28 holds body/DOF maps and `sim_w_last`; `RobotState`, `RootOnlyState` and `ResetState` conversion methods438/607/692 carry format conversion separately. These methods mutate their own state object; future wiring must own snapshots rather than mutate shared views. Local IsaacLab config sets `w_last=True` for Lab3 XYZW; **the bridge does not convert quaternions**.

The future state adapter must validate each asset's body/joint names, construct the correct COMMON/SIMULATOR mapping and then consistently gather actor batches. Each asset's joint maps must also be respected on actions/reset writes. No direct production calls have been replaced. World origins, four nonoverlapping initial slot offsets, group collision filtering, static/peer contact sampling, terminal logic, terrain sampling and physics stepping belong to the future world integration. Four synthetic slot tensors are not proof of64 cloned physical environments or peer collisions.

## Actual bounded CPU verification

Five tests passed: exact256-row mixed partition/slot labels; field round trips and input/RNG preservation; tagged slot/group world values and reordered-group provenance; all69 action coordinates; whole/empty reset acceptance and partial/invalid reset, partition, float dtype, finite, shape, slot and device rejection. CUDA was asserted uninitialized before and after. No prior tests were rerun; no captures/models were used. Ruff`afc018` and AST parse`186c5a` had passed before execution, and no source changes were needed. Exact approved and completed command (one invocation):

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/group_articulation_tensor_bridge_staging_v1 -p test_tensor_bridge.py -v
```
