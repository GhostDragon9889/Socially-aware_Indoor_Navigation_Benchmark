#!/usr/bin/env python3
"""Prepared single-App real-SDK probe; no policy, model checkpoint or training.

Only main executes SDK/data/Tensor code. Root must schedule any future invocation
after the formal GPU job and its registered comparisons. Import is stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import stat
import sys
import time
import traceback
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
STAGES = ROOT / "docs/style_improvement_20261004"
DT, SUBSTEPS, ACTORS, CAP = 1 / 120, 4, 8, 4
OFFSETS = ((-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5))
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
PINS = {
    "World": (STAGES / "shared_group_world_staging_v1/shared_group_world.py", 35097,
              "b608cae6cd8249e2c66d841813b4291b8bd229ca7beae45af625c33daf28fceb", True),
    "normal_factory": (STAGES / "social_normal_force_view_factory_staging_v1/normal_force_views.py", 11810,
                       "517afb01605e5ce90a417f14c663c233d443630e4cec29f54d6169e94358132b", True),
    "eight_actor_smoke": (STAGES / "group_articulation_physics_smoke_staging_v1/smoke_group_physics.py", 21236,
                          "d000200abf55545d0583f896fcfe48f6ebba673afd78b8820782a6a25b6c063d", True),
    "corpus": (ROOT / "CrowdSim/humanoid/amass_corpus.py", 12126,
               "2d4e4f7549aa38f5fd6763de4d3924a9cdd969746682c649ab773f033d747cf7", False),
    "bootstrap": (ROOT / "protomotions/utils/simulator_imports.py", 3118,
                  "7f236c0a5bd071996437ce1298c7911975a1527bea5a33bad61d68f7724918e8", False),
    "scene_config": (ROOT / "protomotions/simulator/isaaclab/utils/scene.py", 11303,
                     "b721a0e2d4f4b5529e191182a5bb152619dde65de36d400627c92ed95731d6f1", False),
    "robot_config": (ROOT / "protomotions/robot_configs/smpl.py", 4067,
                     "505e6e909f5b36a93e5449c75ee10a87b11ce30a4f2052a965ac65d273ae7b2e", False),
    "simulator_config": (ROOT / "protomotions/simulator/isaaclab/config.py", 2589,
                         "f615a5f0a9e7e1457399b2dbdee2ba3fc77b545ff3d25e7427671a49bd21254d", False),
    "MJCF_converter": (ROOT / "protomotions/simulator/isaaclab/utils/mjcf_to_usd.py", 14127,
                       "e8e383b425c3a40b277b4375e031235ef4a3d545d4c6a469ce8528c12e80cae2", False),
    "D6_workaround": (ROOT / "protomotions/simulator/isaaclab/utils/mjcf_d6_workaround.py", 10158,
                      "9e1ee16aa5d3646e4eb4ba3e10060b7537063185fe9450a3612ed53d143ef371", False),
}
GEOMETRY = {"grid_center": "root", "grid_heading": "root", "height_reference": "static_center_ground_z"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="The fixed v7 train2848 manifest")
    parser.add_argument("--num-actors", type=int, choices=(8,), default=8)
    parser.add_argument("--control-steps", type=int, default=8)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--seed", type=int, choices=(7,), default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fall-probe", action="store_true", help="One extra control after an explicitly manipulated root pose")
    args = parser.parse_args()
    if not CAP <= args.control_steps <= 12 - int(args.fall_probe):
        parser.error("Require 4..12 total controls, including the optional one-control fall probe")
    return args


def _record(path, *, readonly=False):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or readonly and info.st_mode & 0o222:
        raise ValueError("Expected regular exclusive source/artifact: " + str(path))
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _sources():
    records = {}
    for key, (path, size, sha, readonly) in PINS.items():
        record = _record(path, readonly=readonly)
        if record["bytes"] != size or record["sha256"] != sha:
            raise ValueError("Reviewed source changed: " + key)
        records[key] = record
    records["execution_entry"] = _record(Path(__file__))
    return records


def _asset_identity(path):
    """Observe the actual local asset bytes; aliases/links are recorded, not hidden."""
    resolved = Path(path).resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Actual local USD/MJCF must be an ordinary file")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Asset changed while its runtime byte identity was observed")
    return {"configured_path": str(path), "path": str(resolved), "bytes": after.st_size,
            "sha256": digest.hexdigest(), "nlink": after.st_nlink, "mode": oct(stat.S_IMODE(after.st_mode))}


def _origin(value, key, records):
    """Bind the actual imported module, not only the desired file on sys.path."""
    module = value if isinstance(value, ModuleType) else sys.modules.get(getattr(value, "__module__", ""))
    if module is None or not getattr(module, "__file__", None):
        raise ValueError("Missing actual imported source origin: " + key)
    path = Path(module.__file__).resolve()
    if path != PINS[key][0].resolve():
        raise ValueError("Actual imported source origin differs from reviewed source: " + key)
    record = _record(path)
    if record != records[key]:
        raise ValueError("Actual imported source bytes changed: " + key)
    return {"module": module.__name__, **record}


def _load(path, name, record):
    data = Path(path).read_bytes()
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("Source changed before execution: " + str(path))
    if name in sys.modules:
        raise RuntimeError("Refuse an already loaded private probe module: " + name)
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _output(path):
    if path.is_symlink():
        raise ValueError("Output cannot be a symbolic link")
    destination = path.resolve()
    allowed = (ROOT / "output", Path("/tmp"))
    if not any(parent in destination.parents for parent in allowed):
        raise ValueError("Output must be a new/empty child of repo output or /tmp")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Refuse nonempty output; failures and partial captures must be retained")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _json(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item, torch) for item in value]
    return value


def _write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return _record(path)


def _write_npz(path, arrays):
    # NumPy is imported only in the future App-owned run. Never pickle payloads.
    import numpy as np

    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return _record(path)


def _tensor_arrays(values):
    return {key: value.detach().clone().cpu().numpy() for key, value in values.items()}


def _state_arrays(snapshot):
    result = {}
    for key in ("root_pos", "root_rot", "root_vel", "root_ang_vel", "rigid_body_pos", "rigid_body_rot",
                "rigid_body_vel", "rigid_body_ang_vel", "dof_pos", "dof_vel", "dof_forces"):
        value = getattr(snapshot.robot_state, key, None)
        if value is not None:
            result["common_" + key] = value
    for key in ("root_pos", "root_rot", "root_vel", "root_ang_vel"):
        result["native_link_" + key] = getattr(snapshot.root_state, key)
    return result


def _observation_arrays(observation):
    return {key: getattr(observation, key) for key in
            ("self_obs", "task_obs", "amp_history", "history_episode_ids", "episode_ids", "age", "plan_age", "owners")}


def _check_observation(observation, world, torch):
    for key, shape in (("self_obs", (ACTORS, 357)), ("task_obs", (ACTORS, 12308)), ("amp_history", (ACTORS, 10, 195))):
        value = getattr(observation, key)
        if (tuple(value.shape) != shape or value.dtype != torch.float32 or value.device != world.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Actual observation shape/dtype/device/finiteness failure: " + key)
    if not torch.equal(observation.history_episode_ids, observation.episode_ids[:, None].expand(ACTORS, 10)):
        raise ValueError("History crosses an actor episode")


def _control(world, output, label, report, torch):
    before_counter, old_episodes = world._counter(), world.episode_ids.clone()
    pre = world.prepare_control()
    _check_observation(pre, world, torch)
    report["artifacts"].append(_write_npz(output / (label + "_pre.npz"), _tensor_arrays({
        **_state_arrays(world.current), **_observation_arrays(pre), "group_actor_rows": world.rows,
        "scene_ids": world.scene_ids, "physics_step": torch.tensor(before_counter, dtype=torch.int64)})))
    actions = torch.zeros((ACTORS, 69), dtype=torch.float32, device=world.device)
    mask = torch.tensor([name.rsplit("_", 1)[0] not in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}
                         for name in world.deps.observations.PACER_DOF_NAMES], dtype=torch.bool, device=world.device)
    if int(mask.sum()) != 57:
        raise ValueError("Expected canonical57 effective actions")
    actions *= mask  # Explicit zero applied action; no Gaussian/actor policy is constructed.
    pending = world.step(actions)
    _check_observation(pending.post, world, torch)
    if (pending.physics_step_before != before_counter or pending.physics_step_after != before_counter + SUBSTEPS
            or world._counter() != before_counter + SUBSTEPS or not torch.equal(world.episode_ids, old_episodes)
            or not torch.equal(pending.post.age, pre.age + 1) or not torch.equal(pending.post.plan_age, pre.plan_age + 1)):
        raise ValueError("Control clock or premature episode reset differs from the driver contract")
    forces, force_metadata = world.deps.sampler.sample_current_net_normal(*world.bundle, world.rows,
        bridge=world.deps.bridge, device=world.device, physics_dt=DT,
        physics_step=world._counter(), control_step=world.control_step)
    # All original episodes, including terminal/collateral rows, get distinct positive values.
    # This is an archival handoff probe, not a learned critic or value-quality test.
    next_values = torch.arange(1, ACTORS + 1, dtype=torch.float32, device=world.device) + 100 * world.control_step
    archive_returned = False
    def archive(transition, values):
        nonlocal archive_returned
        if (transition is not pending or world._counter() != before_counter + SUBSTEPS
                or not torch.equal(world.episode_ids, old_episodes) or not torch.equal(values, next_values)
                or not bool((values > 0).all())):
            raise ValueError("All-N synthetic terminal nextV must be archived on original episodes before reset")
        reset = transition.reset_mask
        if not torch.equal(reset[world.rows].any(1)[:, None].expand(-1, 4), reset[world.rows]):
            raise ValueError("Reset must cover complete four-actor groups")
        raw_terminal = transition.physical_fall | transition.nonfoot_normal_contact.contact_flag
        original_trunc = transition.cap_truncated | transition.source_end_truncated
        collateral = reset & ~(raw_terminal | original_trunc)
        if (not torch.equal(transition.terminated, raw_terminal)
                or not torch.equal(transition.collateral_truncated, collateral)
                or not torch.equal(transition.truncated, original_trunc | collateral)):
            raise ValueError("Real terminal and group-induced truncation meanings changed")
        arrays = {**_state_arrays(transition.snapshot), **_observation_arrays(transition.post),
            "pre_amp_history": transition.pre.amp_history, "applied_actions": transition.applied_actions,
            "synthetic_next_values": values, "signed_substep_mean_power_w": transition.signed_power_w,
            "target_xy": transition.target_xy, "terminated": transition.terminated, "truncated": transition.truncated,
            "reset_mask": reset, "collateral_truncated": transition.collateral_truncated,
            "physical_fall": transition.physical_fall, "cap_truncated": transition.cap_truncated,
            "source_end_truncated": transition.source_end_truncated,
            "nonfoot_normal_contact_flag": transition.nonfoot_normal_contact.contact_flag,
            "nonfoot_resultant_force_w": transition.nonfoot_normal_contact.resultant_force_w,
            "nonfoot_resultant_magnitude_n": transition.nonfoot_normal_contact.magnitude_n,
            "net_normal_force_w": forces, "actual_root_quat_xyzw": transition.actual_root_quat_xyzw,
            "used_task_heading": transition.used_task_heading,
            "task_heading_defined": transition.task_heading_defined,
            "singular_heading_fallback": transition.singular_heading_fallback,
            "physics_step": torch.tensor(world._counter(), dtype=torch.int64)}
        report["artifacts"].append(_write_npz(output / (label + "_post_before_reset.npz"), _tensor_arrays(arrays)))
        report["controls"].append({"label": label, "control_step": world.control_step,
            "physics_step_before": before_counter, "physics_step_after": world._counter(),
            "physical_fall": transition.physical_fall.cpu().tolist(), "terminated": transition.terminated.cpu().tolist(),
            "truncated": transition.truncated.cpu().tolist(), "reset_mask": reset.cpu().tolist(),
            "cap_truncated": transition.cap_truncated.cpu().tolist(), "synthetic_nextV_archived_before_reset": True,
            "nextV_is_actual_critic": False, "normal_force_metadata": force_metadata})
        archive_returned = True
    reset_ids = world.complete_transition(next_values, archive)
    if not archive_returned or world._counter() != before_counter + SUBSTEPS:
        raise ValueError("Completion/reset advanced physics or did not return the archive callback")
    reset_mask = pending.reset_mask
    if (not torch.equal(reset_ids, reset_mask.nonzero().flatten())
            or not torch.equal(world.episode_ids, old_episodes + reset_mask.to(torch.int64))
            or bool((world.age[reset_mask] != 0).any()) or bool((world.plan_age[reset_mask] != 0).any())
            or not torch.equal(world.age[~reset_mask], pending.post.age[~reset_mask])
            or not torch.equal(world.plan_age[~reset_mask], pending.post.plan_age[~reset_mask])
            or world._prepared is not None or world._pending is not None
            or not torch.equal(world.amp_history_episode_ids, world.episode_ids[:, None].expand(ACTORS, 10))):
        raise ValueError("Group reset episodes/ages/history/pending identity failed")
    report["artifacts"].append(_write_npz(output / (label + "_after_reset.npz"), _tensor_arrays({
        **_state_arrays(world.current), "episode_ids": world.episode_ids, "age": world.age,
        "plan_age": world.plan_age, "amp_history": world.amp_history,
        "history_episode_ids": world.amp_history_episode_ids,
        "physics_step": torch.tensor(world._counter(), dtype=torch.int64)})))
    report["completed_controls"] += 1
    return pending


def _manipulate_fall(world, output, report, torch):
    # An explicit external pose manipulation, not an actual rollout transition.
    # Preserve all other actors' states; whole-group writer IDs satisfy the bridge.
    before = world._counter()
    _, states, _ = world.deps.adapter._dependencies()
    fields = {key: getattr(world.current.root_state, key).clone() for key in ("root_pos", "root_rot", "root_vel", "root_ang_vel")}
    fields.update({key: getattr(world.current.robot_state, key).clone() for key in ("dof_pos", "dof_vel")})
    fields["root_pos"][0, 2] = .3
    fields["root_rot"][0] = torch.tensor([1., 0., 0., 0.], dtype=torch.float32, device=world.device)
    fields["root_vel"][0] = 0
    fields["root_ang_vel"][0] = 0
    reset = states.ResetState(state_conversion=states.StateConversion.COMMON, **fields)
    world.adapter.write_common_reset(reset, world.rows[0].clone())
    world.scene.reset(env_ids=torch.tensor([0], dtype=torch.int64, device=world.device))
    world.sim.forward()
    if world._counter() != before:
        raise ValueError("Explicit pose manipulation/forward must not step physics")
    world.current = world.adapter.snapshot()
    physical_fall = torch.zeros(ACTORS, dtype=torch.bool, device=world.device)
    physical_fall[0] = True
    world._select_task_heading(physical_fall)
    report["artifacts"].append(_write_npz(output / "fall_manipulated_before_control.npz", _tensor_arrays(_state_arrays(world.current))))
    report["fall_manipulation"] = {"actor": 0, "root_world_z_m": .3, "root_quaternion_xyzw": [1., 0., 0., 0.],
        "physics_step_unchanged": True, "history_restamped_as_physical_transition": False,
        "claim": "manipulated pose; next real control must independently flag a physical fall"}


def _run(args, report):
    sys.path.insert(0, str(ROOT))
    if "torch" in sys.modules:
        raise RuntimeError("Fresh process required: simulator bootstrap must precede Torch")
    records = report["source_records"] = _sources()
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    origins = report["actual_import_origins"] = {"bootstrap": _origin(import_simulator_before_torch, "bootstrap", records)}
    launcher = import_simulator_before_torch("isaaclab")
    app, world, sim_type, creation_attempted = None, None, None, False
    try:
        app = launcher({"headless": True, "device": args.device, "enable_cameras": False, "fast_shutdown": False,
            "kit_args": "--/app/renderer/enabled=false --/renderer/enabled=false"}).app
        report["app_constructors_returned"] = 1
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab_physx.physics import PhysxCfg
        from protomotions.robot_configs.smpl import SmplRobotConfig
        from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
        from protomotions.simulator.isaaclab.utils.scene import SceneCfg
        from CrowdSim.humanoid.amass_corpus import AMASSCorpus

        for key, value in (("corpus", AMASSCorpus), ("robot_config", SmplRobotConfig),
                           ("simulator_config", IsaacLabSimulatorConfig), ("scene_config", SceneCfg)):
            origins[key] = _origin(value, key, records)
        converter = importlib.import_module("protomotions.simulator.isaaclab.utils.mjcf_to_usd")
        origins["MJCF_converter"] = _origin(converter, "MJCF_converter", records)
        _origin(SceneCfg.__init__.__globals__["convert_robot_mjcf_to_usd"], "MJCF_converter", records)
        torch.set_num_threads(4)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        sim_type = sim_utils.SimulationContext
        if sim_type.instance() is not None:
            raise RuntimeError("Probe refuses an existing simulation context")
        manifest = _record(args.manifest, readonly=True)
        if manifest["sha256"] != MANIFEST_SHA:
            raise ValueError("Fixed v7 manifest SHA mismatch")
        manifest_document = json.loads(args.manifest.read_text())
        train = manifest_document["splits"]["train"]
        if train["clip_count"] != 2848 or len(train["clips"]) != 2848:
            raise ValueError("Require all2848 train clips; no hidden filtering")
        report["AMASS_manifest"] = manifest
        report["AMASS_bank"] = {"path": train["path"], "declared_bytes": train["bytes"], "declared_sha256": train["sha256"],
            "split_loaded": "train", "full_bank_hash_recomputed": False, "bank_is_reference_data_not_policy_weights": True}
        corpus = AMASSCorpus(args.manifest, "train", device=args.device, seed=args.seed, dof_velocity_frame="child_joint")
        world_module = _load(PINS["World"][0], "_probe_reviewed_shared_world", records["World"])
        normal = _load(PINS["normal_factory"][0], "_probe_reviewed_normal_factory", records["normal_factory"])
        # Original relative imports use a private package path; never execute the
        # NN-bearing CrowdSim.humanoid.training.__init__ or change module source.
        package = ModuleType("_probe_original_training")
        package.__path__ = [str(ROOT / "CrowdSim/humanoid/training")]
        sys.modules[package.__name__] = package
        for key, filename in (("procedural", "trajectories"), ("amass", "amass_trajectories"), ("mixed", "mixed_trajectories")):
            path, size, sha, _ = world_module.PINS[key]
            record = _record(path)
            if record["bytes"] != size or record["sha256"] != sha:
                raise ValueError("Original curriculum source changed: " + key)
            _load(path, package.__name__ + "." + filename, record)
        procedural = sys.modules[package.__name__ + ".trajectories"]
        mixed = sys.modules[package.__name__ + ".mixed_trajectories"]
        curriculum = mixed.MixedTrajectoryCurriculum(corpus, ACTORS, device=args.device,
            bounds_xy=None, tail_policy="linear_velocity",
            procedural_config=procedural.PacerTrajectoryConfig(), seed=args.seed)
        config = IsaacLabSimulatorConfig(num_envs=2, headless=True)
        config.sim.fps, config.sim.decimation = 120, SUBSTEPS
        config.sim.physx.num_position_iterations, config.sim.physx.num_velocity_iterations = 4, 4
        robot = SmplRobotConfig()
        robot.contact_bodies = None
        base = SceneCfg(config=config, robot_config=robot, terrain=None, projectile_config=None,
            num_envs=2, env_spacing=8., filter_collisions=True, replicate_physics=True)
        workaround = sys.modules.get("protomotions.simulator.isaaclab.utils.mjcf_d6_workaround")
        origins["D6_workaround"] = ({"loaded": True, **_origin(workaround, "D6_workaround", records)} if workaround is not None
            else {"loaded": False, "expected_source": records["D6_workaround"],
                  "note": "Not imported on this actual path; a cached USD can bypass converter/workaround execution"})
        base.robot.spawn.activate_contact_sensors = True
        if not Path(base.robot.spawn.usd_path).is_file():
            raise ValueError("Reviewed local SMPL USD must exist; no new remote asset download")
        base.ground = AssetBaseCfg(prim_path="/World/ground", collision_group=-1,
            spawn=sim_utils.CuboidCfg(size=(64., 64., .2),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.1)))
        def static_height(query):
            if bool((query.abs() > 32).any()):
                raise ValueError("Actual query left the finite64m floor; no infinite-ground assumption")
            return torch.zeros(query.shape[:-1], dtype=query.dtype, device=query.device)
        def geometry(snapshot, root_heading, defined):
            del defined  # Heading continuation is the World's explicit declared mechanism.
            xy = snapshot.robot_state.root_pos[:, :2].clone()
            return xy, root_heading.clone(), static_height(xy)
        if sim_type.instance() is not None:
            raise RuntimeError("Scene configuration unexpectedly created a simulation context; refuse ownership")
        creation_attempted = True
        world = world_module.SharedGroupWorld.create_after_app(app,
            sim_utils.SimulationCfg(device=args.device, dt=DT, render_interval=SUBSTEPS,
                physics=PhysxCfg(max_position_iteration_count=4, max_velocity_iteration_count=4)),
            base.robot, base, num_actors=ACTORS, actor_offsets_xy=OFFSETS, env_spacing=8., robot_config=robot,
            curriculum=curriculum, ground_height_at=static_height, terrain_sampler=static_height, grid_geometry=geometry,
            grid_geometry_contract=GEOMETRY, include_ego=False,
            normal_force_bundle_factory=normal.create_normal_force_bundle, episode_steps=CAP)
        report["actual_initialized_physics_baseline"] = world._counter()
        report["actual_robot_MJCF"] = _asset_identity(converter.resolve_robot_mjcf_path(robot.asset))
        report["actual_native_assets"] = [{"name": f"robot_{slot}", "instances": asset.num_instances,
            "body_names": list(asset.body_names), "joint_names": list(asset.joint_names),
            "USD": _asset_identity(asset.cfg.spawn.usd_path)} for slot, asset in enumerate(world.assets)]
        if any(item["instances"] != 2 or len(item["body_names"]) != 24 or len(item["joint_names"]) != 69
               for item in report["actual_native_assets"]):
            raise ValueError("Actual four assets must each provide2 instances/24 bodies/69 joints")
        report["actual_world_origins"] = world.scene.env_origins.detach().cpu().tolist()
        report["initial_world_description"] = _json(world.describe(), torch)
        report["artifacts"].append(_write_npz(args.output / "initial_after_AMASS_reset.npz", _tensor_arrays({
            **_state_arrays(world.current), "world_origins": world.scene.env_origins,
            "spawn_world_xyz": world.spawn_world_xyz, "group_actor_rows": world.rows,
            "episode_ids": world.episode_ids, "age": world.age, "amp_history": world.amp_history,
            "history_episode_ids": world.amp_history_episode_ids,
            "physics_step": torch.tensor(world._counter(), dtype=torch.int64)})))
        for index in range(args.control_steps):
            report["attempted_control"] = index + 1
            _control(world, args.output, f"control_{index+1:02d}", report, torch)
        if not any(any(item["cap_truncated"]) for item in report["controls"]):
            raise ValueError("Episodecap4 was not actually observed; early terminal resets are retained, not hidden")
        if args.fall_probe:
            _manipulate_fall(world, args.output, report, torch)
            report["attempted_control"] = args.control_steps + 1
            pending = _control(world, args.output, "manipulated_fall_control", report, torch)
            if not bool(pending.physical_fall[0]) or not bool(pending.reset_mask[world.rows[0]].all()):
                raise ValueError("Manipulated pose did not produce a real physical-fall flag and whole-group reset")
        if world._counter() - report["actual_initialized_physics_baseline"] != SUBSTEPS * report["completed_controls"]:
            raise ValueError("Final real counter differs from bounded4-substep controls")
        report["final_world_description"] = _json(world.describe(), torch)
        report["source_records_after"] = _sources()
        if report["source_records_after"] != records:
            raise ValueError("Sources changed during the actual probe")
        report["probe_checks_passed"] = True
    except BaseException:
        if world is not None:
            try:
                torch = importlib.import_module("torch")
                report["failure_world_description"] = _json(world.describe(), torch)
                report["failure_actual_physics_step"] = world._counter()
                report["artifacts"].append(_write_npz(args.output / "failure_actual_native_snapshot.npz", _tensor_arrays({
                    **_state_arrays(world.adapter.snapshot()), "episode_ids": world.episode_ids,
                    "age": world.age, "amp_history": world.amp_history,
                    "physics_step": torch.tensor(world._counter(), dtype=torch.int64)})))
                report["failure_snapshot_is_validated_control_endpoint"] = False
            except BaseException as error:
                report["failure_observation_error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        cleanup_errors = []
        sim = None
        try:
            sim = world.sim if world is not None else sim_type.instance() if creation_attempted and sim_type is not None else None
        except BaseException as error:
            cleanup_errors.append("context lookup: " + type(error).__name__ + ": " + str(error))
        if sim is not None:
            try:
                sim.stop()
                report["simulation_stop_returned"] = True
            except BaseException as error:
                cleanup_errors.append("sim.stop: " + type(error).__name__ + ": " + str(error))
        if app is not None:
            try:
                app.close()
                report["app_close_returned"] = True
            except BaseException as error:
                cleanup_errors.append("app.close: " + type(error).__name__ + ": " + str(error))
        report["cleanup_errors"] = cleanup_errors
        report["cleanup_succeeded"] = not cleanup_errors and report["simulation_stop_returned"] and report["app_close_returned"]


def main():
    args = parse_args()
    args.output = _output(args.output)
    started = time.time()
    report = {"schema": "social.shared_group_world.bounded_real_SDK_probe.v1", "status": "running",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "started_unix_s": started, "app_constructors_returned": 0, "simulation_stop_returned": False,
        "app_close_returned": False, "cleanup_succeeded": False, "probe_checks_passed": False,
        "completed_controls": 0, "controls": [], "artifacts": [], "applied_action_source": "untrained zero69, canonical57 mask",
        "value_source": "positive synthetic all-N values; no model/critic", "grid_geometry_contract": GEOMETRY,
        "include_ego": False, "episode_cap_controls": CAP, "rendered": False,
        "real_training_or_policy_quality_claimed": False, "SDK_resume_or_independent_capture_audit_claimed": False}
    try:
        _run(args, report)
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["elapsed_wall_s"] = time.time() - started
    report["output_inventory_before_result"] = [{"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(args.output.iterdir()) if path.is_file()]
    report["status"] = "passed" if report["probe_checks_passed"] and report["cleanup_succeeded"] and "error" not in report else "failed_preserved"
    _write_json(args.output / "result.json", report)
    print(json.dumps({"status": report["status"], "completed_controls": report["completed_controls"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
