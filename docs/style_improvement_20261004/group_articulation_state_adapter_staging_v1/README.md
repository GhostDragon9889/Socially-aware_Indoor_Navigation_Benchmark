# Lazy four-Articulation state and write adapter

The sole approved invocation actually passed **five fake-asset CPU tests in0.120s**, following independent source peer Required=[] and root whole-read. Tool`d7ee93` returned observed exit0 directly (wall0.989s; no running session was returned). Source/test bytes stayed unchanged. [CPU checks](actual_CPU_checks.json) records their exact identities, command and observed combined output. No Isaac Lab/Warp/MuJoCo import, real asset access, SDK writer or physics step has been executed for this stage. Production185 and all previous stages remain unchanged.

## Initialization and named conversion

Construct `FourArticulationAdapter(assets, group_actor_rows, kinematic_info, device=...)` with four actual assets or minimal mocks in `robot_0..robot_3` / `Robot_0..Robot_3` slot order. The explicit map must partition every canonical actor row; the frozen tensor bridge supplies row conversion. For256 actors,64 cloned environments contain rows `[2g,2g+1,128+2g,129+2g]`. No IDs or origins are created.

Use the actual SMPL kinematic description:24 unique common body names beginning with Pelvis,69 unique semantic DOF names, and zero root DOFs plus23 three-axis hinge groups. Each asset's actual `body_names` and `joint_names` must match that description through the current Isaac Lab semantic resolver. Width69 alone cannot establish joint order. Four assets may have different native body/DOF orderings; each receives its own original `DataConversionMapping`, with `sim_w_last=True` for locally inspected Lab3 XYZW. No legacy WXYZ input is accepted as if it were XYZW.

The original `build_isaaclab_joint_name_map` follows kinematic hinge counts and maps a multi-axis semantic triple to backend names such as `L_Hip_x:0`, `L_Hip_x:1`, `L_Hip_x:2`. It is the same function called by current `IsaacLabSimulator._get_sim_body_ordering`. Importing its whole module would import `pose_lib`→MuJoCo/dm_control. Instead, the adapter verifies complete source bytes and compiles the **unchanged original class and builder AST only**, with future annotations. Runtime builder globals were inspected by pure stdlib compilation/disassembly: `IsaacLabJointNameMap`, `ValueError`, `len`, `range`, `zip`; dataclass decoration and Mapping annotations are explicitly supplied. No helper implementation is rewritten or copied into this directory.

## Link snapshots

`snapshot()` retrieves every property anew, accesses its zero-copy `.torch` view, and immediately **clones** it before retaining any state. Minimal mocks may expose a Torch tensor directly. Exact properties:

| Output | Property |
| --- | --- |
| Native root XYZ/XYZW | `root_link_pos_w`, `root_link_quat_w` |
| Native root world linear/angular velocity | `root_link_lin_vel_w`, `root_link_ang_vel_w` |
| Body world XYZ/XYZW | `body_link_pos_w`, `body_link_quat_w` |
| Body world linear/angular velocity | `body_link_lin_vel_w`, `body_link_ang_vel_w` |
| Native joint position/velocity | `joint_pos`, `joint_vel` |

All snapshots are finite float32 on the explicitly declared CPU or indexed CUDA device; quaternion norms must be unit within `atol=1e-3,rtol=0`, without normalization. Root/body velocity COM aliases are not used. Original `RobotState` and `RootOnlyState` start as SIMULATOR and convert using each asset's name map, then the frozen bridge gathers their four asset batches into canonical actor rows. Conversion mutates only newly owned state objects.

Return `CommonSnapshot.robot_state` (COMMON bodies/DOFs; its root properties view mapped Pelvis) and `CommonSnapshot.root_state` (separately gathered COMMON native root-link state). Both are preserved; no unmeasured equality between native root and mapped body0 is asserted. Caller must resolve any physical mismatch before choosing an AMP root contract.

The PhysX timestamp must already be coherent across all four assets. Property access pulls cached/current data but does **not** call `update`, refresh the simulator clock or step physics. The future world loop must update assets at the correct actual physics/control phase and exclude concurrent stepping during these sequential field reads. Clone ownership prevents later refreshes overwriting snapshots; it cannot make asynchronous reads atomic. Contact, applied torque, signed power and terminal masks are **not present** and remain pending.

## Position targets and resets

`set_common_position_targets(common_pd_targets[N,69])` receives already constructed COMMON **PD position targets**, not raw policy actions. It fully validates the tensor, gathers four `[G,69]` batches, reorders DOFs with each original `dof_convert_to_sim`, and calls each asset's real `set_joint_position_target_index(target=..., joint_ids=None, env_ids=group_indices, full_data=False)`. It fills target buffers only. Caller owns `action_to_pd`, limits, actuator settings, `write_data_to_sim` and120Hz stepping; no flush or step is hidden in this adapter.

`write_common_reset(COMMON_ResetState, canonical_rows)` requires full canonical `[N,...]` root/DOF reset fields and a unique **whole-group** actor-row request. Partial groups are refused, never expanded; a valid empty request calls no writer. All four datasets and float32/unit/dtype/device/shape contracts are validated and prepared **before the first SDK writer**. Each new per-asset ResetState selects `[K,...]` using the same group env IDs and converts DOFs to that asset's native order. Calls are explicit:

1. `write_root_link_pose_to_sim_index(root_pose=[K,7], env_ids=..., full_data=False)`.
2. `write_root_link_velocity_to_sim_index(root_velocity=[K,6], env_ids=..., full_data=False)`.
3. `write_joint_state_to_sim_index(position=[K,69], velocity=[K,69], joint_ids=None, env_ids=..., full_data=False)`.
4. `set_joint_position_target_index(target=[K,69], joint_ids=None, env_ids=..., full_data=False)`.

The inspected installed methods accept Torch inputs. They convert int64 Torch env IDs to int32 Warp IDs internally; Warp kernels ingest position/velocity tensors directly. The adapter does not invent a Warp conversion or import Warp. All values are already world coordinates; no environment-origin, worldZ or terrain offset is added. It does not reset fall flags, contact caches, trajectory/history/curriculum/episode clocks or coupled AMP state.

SDK failures may occur after earlier writers succeeded. They propagate directly: this is **not a transaction**, rollback or safe automatic retry. No physics quality, shared-group reset completeness outside these fields, actual64×4 Articulation population or collision filtering follows from fake writer tests.

## Byte pins and source-only API basis

Dependencies are checked by full byte length/SHA once before local modules are loaded. The bridge additionally must be an ordinary readonly single-link file. Current state and rotations are imported only from their checked repository paths; they contain Torch tensor utilities and no model construction, weight loading or SDK import. The semantic resolver selected AST avoids its unrelated heavy imports. SDK source is hashed but not imported:

| Local source | Bytes | SHA256 |
| --- | ---: | --- |
| `../group_articulation_tensor_bridge_staging_v1/tensor_bridge.py` |6035|`2f5dc721f739d26ace02cc0831864523e605587f7e960f0fd32b82326ec8c9ec`|
| `protomotions/simulator/base_simulator/simulator_state.py` |34638|`31cf14fab319151602f43fcf6ad2751b71088cda3a3f4feefadfa61262102731`|
| `protomotions/utils/rotations.py` |28136|`ef1475d33d6f941ccdea4a612477483db4fad8236ba7786fbe8bf91a65c1cef8`|
| `protomotions/simulator/isaaclab/utils/actuator_groups.py` |4511|`fe1aeaca98a0548e3dd6847e3831b232689a41137a196db2fa65f68683bcdafa`|
| Installed PhysX `articulation.py` |224858|`2cbd53ccbebb7a7b038bed2b612b65e41d6674a4791beec68d84336d7bd00a87`|
| Installed PhysX `articulation_data.py` |95901|`6579e5b3efcd13d113cda141945cd055a4d0b86a069f0a80fc2c73c4cfe4163b`|

Installed PhysX sources are under `/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/`. Actual source reads show write signatures at452/803/906/2498, environment-index handling4636–4656, and pull-to-refresh/link data contracts48–61/708–747/824–878/1040–1068/1223–1384. `write_data_to_sim`248 applies buffered commands; it is deliberately left to the future caller. Original ProtoMotions getter/conversion/writer locations are recorded in the already sealed bridge README. These pins bind this local inspected revision, not any arbitrary future SDK release.

## Actual single CPU check

Five new tests passed:64-group named body/joint remapping plus fresh-property/owned-snapshot preservation; invalid names/hinge triples; COMMON PD target mappings without flush; whole/partial/empty reset and prevalidation before writes; transparent partial SDK failure with no SDK imports. CUDA stayed uninitialized before and after. Fake buffers and methods only; these are not physical execution. No previous tests were rerun. Original dependencies emitted a TorchScript deprecation warning and an invalid escape SyntaxWarning; both are retained in the receipt and caused no failure. They were not suppressed and no dependency was edited. Ruff`73171d` and AST parse`0c775b` passed before execution. Exact approved and completed command (one invocation):

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/group_articulation_state_adapter_staging_v1 -p test_state_adapter.py -v
```
