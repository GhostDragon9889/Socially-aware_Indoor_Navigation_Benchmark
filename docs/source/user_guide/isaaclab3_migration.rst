IsaacLab 3 Migration Pin
========================

The benchmark targets IsaacLab 3.0 EA / Isaac Sim 6.1.0.0. Product VERSION,
package version and source commit are distinct identities. The original
nested-body contact-sensor fix remains included in the selected source.

Default revision and environment
--------------------------------

* Source: ``ae37b028ea415c91ea2bc32609efcd759ed2b974`` (``v3.0.0-EA``)
* Product VERSION: ``3.0.0``; package: ``isaaclab==17.0.2``
* Python: ``3.12.13``; Sim package: ``6.1.0.0``
* Project interpreter: ``.runtime/isaaclab3_ea_sim61/.venv/bin/python``
* SDK and overlay locks: ``env/isaaclab3_ea_sim61/``

``scripts/bootstrap_reproduction.py prepare-runtime --profile isaaclab3_ea_sim61
--lab-root /absolute/path/to/IsaacLab`` builds project additions without changing
an installed SDK. New installations use the same script's ``install`` action.

The old source ``4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`` / package 12 / Sim 6.0
profile and its original evidence remain historical. Its frozen manifest refers
to the old project extra and is not reapplied to migrated source.

Humanoid assets
---------------

Humanoid robot configs declare only MJCF via ``RobotAssetConfig.asset_file_name``.
At IsaacLab scene construction, ``convert_robot_mjcf_to_usd()`` invokes IsaacLab
3 ``MjcfConverterCfg`` / ``MjcfConverter`` and spawns through ``UsdFileCfg``.
Contact-sensor prim paths are resolved from the converted USD hierarchy
(flat or nested). Non-IsaacLab simulators continue to load the same MJCF files.

Temporary MJCF importer compatibility
--------------------------------------

The IsaacLab 3 converter currently collapses multiple single-axis MJCF joints
between the same bodies into a PhysX D6 joint. ProtoMotions installs a narrow
compatibility repair at first conversion that preserves the shared source
frame, restores MuJoCo spring gains, and locks unused D6 axes. IsaacLab still
owns the conversion; the repair is removed once the upstream importer includes
the corresponding fix. The converter cache includes the repair version and SDK/importer metadata, so a
USD generated with another SDK is not reused silently.

Sim 6.1 importer output defaults to the generic ``Physics=physics`` payload,
which retains parallel MJCF hinges. Robot spawn and USD path inspection
explicitly select ``Physics=physx`` to consume the repaired D6 layer. CPU USD
inspection confirmed 24 bodies / 23 D6 joints / 69 rotational DOFs for SMPL;
this does not replace live PhysX dynamics testing.

For SOMA and SMPL, IsaacLab exposes the collapsed axes with ``:0``, ``:1``, and
``:2`` suffixes. ProtoMotions maps those backend names back to the semantic MJCF
DOF names before configuring actuators and simulator state.

Kit-free dry run
----------------

Unit tests inject a converter factory or set
``PROTOMOTIONS_ISAACLAB_MJCF_DRY_RUN=1`` to exercise path/config/caching without
starting Kit.

Current Sim 6.1 acceptance (2026-10-11)
---------------------------------------

The full current CPU suite passes 4,286 nodes with seven explicit skips. Five
of those CUDA nodes pass on real GPU; the remaining two require legacy
IsaacGym. The original 354 historical source/SDK guards and their failures
remain separately listed, with no new exclusions.

The project runs real 69-DOF / 24-body SMPL PhysX states. Four original scene
profiles are checked separately for 1,800 controls (60 simulated seconds),
with actual parent exit codes, source hashes and contact archives. Original
profiles use 4/4/1/1 actors; single-actor scenes do not establish crowd
interaction or full-scene coverage. Local proxy gates do not establish the
paper's complete benchmark or visually natural gait.

An eight-environment AMP smoke completes 12 updates / 3,072 physical
transitions and samples all 2,848 training clips. Export/checkpoint CPU
roundtrip passes. This is compatibility evidence, not convergence or a
promoted production checkpoint; GPU mid-episode resume is not validated.

RTX smoke renders three real H264 frames and exits normally. It replays
accepted physical states; it is neither fresh physics nor a 60-second video.
Earlier 60-second videos retain their Sim 6.0 identity. Detailed receipts and
unverified scopes are documented in the repository's
``docs/ISAACLAB3_EA_SIM61_MIGRATION_20261011_ZH.md`` and its English summary.

Historical headless smoke (Sim 6.0)
-----------------------------------

The following earlier smoke facts apply to the old 4ecd0b/source12/Sim 6.0 SDK,
not to the new EA pin. A new run requires separate physical and exit receipts::

   python protomotions/inference_agent.py \\
     --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \\
     --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \\
     --simulator isaaclab --num-envs 1 --headless

That path converts the robot MJCF, builds the articulation/contact sensors, and
steps physics. In the local verification, Kit reached scene setup and policy
materialization before a ten-minute bound expired during cold Torch warm-up.
The direct simulator tutorial (``examples/tutorial/0_create_simulator.py``)
using the same pinned runtime initialized, reset, stepped, and read state for
more than 3,800 action/physics cycles before its 180-second guard expired. Allow
more time on a cold machine for the first policy compilation.

.. _#5085: https://github.com/isaac-sim/IsaacLab/issues/5085
.. _#6259: https://github.com/isaac-sim/IsaacLab/pull/6259
