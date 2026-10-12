#!/usr/bin/env python3
"""Prepared single-App real-SDK probe; no policy, model checkpoint or training.

Only main executes SDK/data/Tensor code. Root must schedule any future invocation
after the formal GPU job and its registered comparisons. Import is stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import gc
import importlib
import json
import os
from pathlib import Path
import random
import stat
import sys
import time
import traceback
import weakref
from types import ModuleType

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
STAGES = ROOT / "resources/runtime_sources/style_improvement_20261004"
DT, SUBSTEPS, ACTORS, CAP = 1 / 120, 4, 8, 4
OFFSETS = ((-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5))
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
LIFECYCLE_SDK_GATE_READY = True  # Each _sources call must admit ROOT v18 actual closed0 and current exact overlay evidence.
PINS = {
    "World": (STAGES / "shared_group_world_owned_lifetime_revision_v2/shared_group_world.py", 39960,
              'd7400a48364b64971f4e2e942d87ebd9e108b5a8ff018e8a8f404b817bd02038', True),
    "normal_factory": (STAGES / "social_normal_force_empty_filter_revision_v3/normal_force_views.py", 17897,
                       '73d38d6f16af322081ba669649ff19a065c59799a54e92bb0404497e80e2c502', True),
    "terminal_cleanup": (STAGES / "social_native_overlay_entry_revision_v10/smoke_group_physics.py", 66700,
                         '270605992e4e600b84b43517efd6bbba82e67a5c449f7a55bfa5a5763269804e', True),
    "lifecycle_owner": (STAGES / "social_actual_runtime_v1/group_physics_native_overlay_v18_actual_root_owner.json", 13111,
                        '5efe84e3391d52d82f8b14acdb4ff61fff3e75bac9a2e2daab83fe291c879434', True),
    "eight_actor_smoke": (STAGES / "group_articulation_physics_smoke_staging_v1/smoke_group_physics.py", 21299,
                          '1f26a658bbbf2190782d626ab5a28d1ce1e830b1ff9b5baeecca4a27b0950eca', True),
    "corpus": (ROOT / "CrowdSim/humanoid/amass_corpus.py", 12126,
               '2d4e4f7549aa38f5fd6763de4d3924a9cdd969746682c649ab773f033d747cf7', False),
    "bootstrap": (ROOT / "protomotions/utils/simulator_imports.py", 3118,
                  '7f236c0a5bd071996437ce1298c7911975a1527bea5a33bad61d68f7724918e8', False),
    "scene_config": (ROOT / "protomotions/simulator/isaaclab/utils/scene.py", 11546,
                     '3fcf83158b29c998971ee59f9e7370183fda9ad6f892a5d8ffc5d5bca79fff51', False),
    "robot_config": (ROOT / "protomotions/robot_configs/smpl.py", 4067,
                     '505e6e909f5b36a93e5449c75ee10a87b11ce30a4f2052a965ac65d273ae7b2e', False),
    "simulator_config": (ROOT / "protomotions/simulator/isaaclab/config.py", 2799,
                         '67682326c12f9fbc9887c4a87256bb36cac0e46ed77ac04bd3fb7320c3ac6db7', False),
    "MJCF_converter": (ROOT / "protomotions/simulator/isaaclab/utils/mjcf_to_usd.py", 15556,
                       'e472e2397a26b20ae90700ab1e7a68dc4cbe4293c0fe90f4c3265c415f351f57', False),
    "D6_workaround": (ROOT / "protomotions/simulator/isaaclab/utils/mjcf_d6_workaround.py", 10158,
                      '9e1ee16aa5d3646e4eb4ba3e10060b7537063185fe9450a3612ed53d143ef371', False),
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


def _manifest_preflight(path):
    """Validate the fixed data contents before starting Kit; writable data is valid."""
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Fixed manifest must be a regular exclusive data file")
    data = path.read_bytes()
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError("AMASS manifest changed during validation")
    row = {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if row["sha256"] != MANIFEST_SHA:
        raise ValueError("Fixed v7 manifest SHA mismatch")
    document = json.loads(data)
    train = document["splits"]["train"]
    if (document["schema"] != "navisaaclab.amass_reconstruction.v1"
            or document.get("author_clip_list") is not False
            or train["clip_count"] != 2848 or len(train["clips"]) != 2848):
        raise ValueError("Require the complete reconstructed train2848 manifest")
    return row, document


def _verify_lifecycle_owner(records):
    """Current plain metadata admission of ROOT's real v18 closed0; no SDK/arrays."""
    evidence = json.loads(PINS["lifecycle_owner"][0].read_text())
    expected_process = {"initial_chunk": "c8b514", "session_id": 93877,
        "terminal_chunk": "e8ad50", "exit_code": 0, "debugger_used": False}
    if (evidence["schema"] != "local.group_physics.actual_root.native_overlay_v18"
            or evidence["status"] != "physical_smoke_passed_and_actual_process_exit0"
            or evidence["actual_process"] != expected_process
            or type(evidence["actual_process"]["exit_code"]) is not int
            or evidence["execution_source"] != records["terminal_cleanup"]
            or evidence["process_exit_not_inferred_from_report_passed"] is not True
            or evidence["installed_extension_before_after_byte_equal"] is not True
            or evidence["fast_shutdown_used"] is not False
            or evidence["NN_training_executed"] is not False
            or evidence["actual_physics_substeps"] != 64
            or evidence["actual_positive_same_group_peak_force_n"] <= 1.
            or evidence["actual_cross_group_raw_force_absolute_max_n"] != 0.
            or evidence["owned_asset_release_counts"] != {"articulation_fields": 4, "data_fields": 4}
            or evidence["owned_contact_weakrefs_live"] != 0 or evidence["backend_weakrefs_live"] != 0
            or evidence["cleanup_errors"] != []):
        raise RuntimeError("Actual v18 lifecycle owner/source/closed0 evidence does not match")
    overlay = ROOT / "output/native_sdk_overlay/simulation_manager_shutdown_v1/isaacsim.core.simulation_manager"
    loaded = evidence["native_overlay_loaded"]
    if (loaded["extension_id"] != "isaacsim.core.simulation_manager-1.15.4"
            or loaded["extension_path"] != str(overlay) or loaded["dev_paths"] != [str(overlay)]
            or loaded["plugin_mapped_paths"] != [str(overlay / "bin/libisaacsim.core.simulation_manager.plugin.so")]
            or loaded["plugin"]["bytes"] != 9562016
            or loaded["plugin"]["sha256"] != "cc21b76874553f8ca035efdfaa7c57d8879b12df66a06cc6d54c2e6b3fcb208c"
            or evidence["overlay_manifest"]["sha256"] != "713c477db243b9c72081beec906fc30d67143cfd61ec4ca3eb61f95c0489c980"):
        raise RuntimeError("Actual v18 overlay plugin, origins or manifest differ")
    for row in [evidence["overlay_manifest"], loaded["plugin"]]:
        if _record(Path(row["path"]), readonly=True) != row:
            raise RuntimeError("Actual v18 admitted overlay bytes changed")


def _sources():
    if LIFECYCLE_SDK_GATE_READY is not True:
        raise RuntimeError("Pending lifecycle SDK exit0 gate: this prepared entry must not execute SDK/NN")
    records = {}
    for key, (path, size, sha, readonly) in PINS.items():
        record = _record(path, readonly=readonly)
        if record["bytes"] != size or record["sha256"] != sha:
            raise ValueError("Reviewed source changed: " + key)
        records[key] = record
    _verify_lifecycle_owner(records)
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


def _owned_cleanup_defaults():
    return {"lifecycle_runtime_ready": LIFECYCLE_SDK_GATE_READY,
        "owned_world_native_release_returned": False, "owned_native_references_release_returned": False,
        "owned_asset_simulation_views_release_returned": False, "owned_asset_simulation_view_release_counts": None,
        "owned_asset_tracker_captured_count": None,
        "owned_contact_view_count": None, "expected_owned_contact_view_count": None,
        "owned_contact_views_live_after_release": None, "pre_context_clear_gc_returned": False,
        "post_context_clear_gc_returned": False, "owned_manager_normalization_returned": False,
        "core_terminal_shutdown_returned": False, "terminal_timeline_cache_compat_returned": False,
        "terminal_scene_backend_release_returned": False, "terminal_scene_backend_weakrefs_released": False,
        "terminal_scene_backend_existing_view_count": None, "scene_backend_views_live_after_post_context_gc": None,
        "NN_constructor_attempted": False, "NN_constructor_returned": False,
        "training_update_attempted_count": 0, "training_update_returned_count": 0,
        "checkpoint_load_attempted": False}


def _owned_release_callback(factory, records, report, receipts, tracked_assets):
    """Capture weak receipts and private owned assets before exact factory unbind."""
    function = factory.release_normal_force_bundle
    if (function.__globals__ is not factory.__dict__ or function.__qualname__ != "release_normal_force_bundle"
            or Path(function.__code__.co_filename).resolve() != PINS["normal_factory"][0].resolve()
            or type(tracked_assets) is not list or tracked_assets):
        raise RuntimeError("Owned release must delegate to the exact pinned factory function")
    report["owned_release_callback_source"] = _origin(function, "normal_factory", records)
    report["owned_release_wrapper_source"] = records["execution_entry"]
    def release(bundle):
        receipt = views = binding = assets = view = left = right = None
        try:
            if type(bundle) is not tuple or len(bundle) != 3 or type(bundle[0]) is not tuple or not bundle[0]:
                raise RuntimeError("Exact original factory bundle required before owned asset capture")
            views = bundle[0]
            for view in views:
                if type(view) is not factory._NormalView:
                    raise RuntimeError("Only exact pinned factory wrappers may expose owned assets")
            binding = views[0].binding
            if type(binding) is not factory._Binding or binding._released or binding._bundle is not bundle:
                raise RuntimeError("One unreleased original binding required before owned asset capture")
            for view in views:
                if view.binding is not binding:
                    raise RuntimeError("One unreleased original binding required before owned asset capture")
            assets = binding.assets
            if type(assets) is not tuple or len(assets) != CAP or len({id(asset) for asset in assets}) != CAP:
                raise RuntimeError("Complete original four owned SDK assets required before factory unbind")
            if tracked_assets:
                if len(tracked_assets) != CAP:
                    raise RuntimeError("Refuse a second or changed owned asset tracker")
                for left, right in zip(tracked_assets, assets, strict=True):
                    if left is not right:
                        raise RuntimeError("Refuse a second or changed owned asset tracker")
            else:
                tracked_assets.extend(assets)
            report["owned_asset_tracker_captured_count"] = len(tracked_assets)
            receipt = function(bundle)
            receipts.append(receipt)
            return receipt
        finally:
            bundle = receipt = views = binding = assets = view = left = right = None
    return release


def _cleanup_owned_runtime(app, world, sim_type, creation_attempted, release_callback, receipts, tracked_assets,
                           expected_actors, terminal, report, *, collector=None, expert=None):
    """Terminal owned cleanup only; no physics/training retry or quality conclusion."""
    errors, sim, scene_backend_refs = [], None, ()
    receipt = refs = assets = left = right = None
    report["expected_owned_contact_view_count"] = expected_actors
    try:
        try:
            sim = world.sim if world is not None else sim_type.instance() if creation_attempted and sim_type is not None else None
        except BaseException as error:
            errors.append("context lookup: " + type(error).__name__ + ": " + str(error))
        if sim is not None:
            try:
                sim.stop()
                report["simulation_stop_returned"] = True
            except BaseException as error:
                errors.append("sim.stop: " + type(error).__name__ + ": " + str(error))
        try:
            if sim is None or report["simulation_stop_returned"] is not True or type(tracked_assets) is not list:
                raise RuntimeError("Owned asset release requires the stopped owned context and private tracker")
            assets = world.assets if world is not None else tuple(tracked_assets)
            if type(assets) is not tuple or len(assets) != CAP:
                raise RuntimeError("No complete observable four owned assets; partial constructor cleanup remains unverified")
            if tracked_assets:
                if len(tracked_assets) != CAP:
                    raise RuntimeError("Partial-construction asset tracker differs from the owned World assets")
                for left, right in zip(tracked_assets, assets, strict=True):
                    if left is not right:
                        raise RuntimeError("Partial-construction asset tracker differs from the owned World assets")
            terminal._release_owned_asset_simulation_views(sim, assets, report)
            if (report["owned_asset_simulation_views_release_returned"] is not True
                    or report["owned_asset_simulation_view_release_counts"] != {"articulation_fields": CAP, "data_fields": CAP}):
                raise RuntimeError("The pinned helper must release all four Articulation and four Data fields")
        except BaseException as error:
            errors.append("owned asset SimulationView release: " + type(error).__name__ + ": " + str(error))
        finally:
            assets = left = right = None
            if type(tracked_assets) is list:
                tracked_assets.clear()
        try:
            if world is not None:
                world.release_owned_native(release_callback)
            elif creation_attempted and not receipts:
                raise RuntimeError("Failed World construction returned no observable native release receipt")
            report["owned_world_native_release_returned"] = True
        except BaseException as error:
            errors.append("World owned release: " + type(error).__name__ + ": " + str(error))
        finally:
            if type(tracked_assets) is list:
                tracked_assets.clear()  # Clear in place even when cached callback frames retain the list.
        for owner, field in ((collector, "world"), (expert, "_world")):
            if owner is not None:
                try:
                    setattr(owner, field, None)
                except BaseException as error:
                    errors.append("caller World owner release: " + type(error).__name__ + ": " + str(error))
        # Fields are explicitly detached, so external cached frames can retain only inert owners.
        world = collector = expert = owner = None
        try:
            report["pre_context_clear_gc_collected"] = gc.collect()
            report["pre_context_clear_gc_returned"] = True
            if (type(expected_actors) is not int or expected_actors < 1
                    or type(receipts) is not list or len(receipts) != 1):
                raise RuntimeError("Exactly one complete owned native release receipt is required")
            receipt = receipts[0]
            if (type(receipt) is not dict or set(receipt) != {
                    "native_weakrefs", "native_view_count", "module_owned_aliases_released"}
                    or receipt["module_owned_aliases_released"] is not True
                    or type(receipt["native_view_count"]) is not int
                    or receipt["native_view_count"] != expected_actors
                    or type(receipt["native_weakrefs"]) is not tuple):
                raise RuntimeError("Owned release receipt must cover the original full actor count")
            refs = receipt["native_weakrefs"]
            if len(refs) != expected_actors or any(type(ref) is not weakref.ReferenceType for ref in refs):
                raise RuntimeError("Full original actor contact weakrefs are required")
            report["owned_contact_view_count"] = len(refs)
            report["owned_contact_views_live_after_release"] = sum(ref() is not None for ref in refs)
            if report["owned_contact_views_live_after_release"] != 0:
                raise RuntimeError("Owned native contact views survive before context teardown")
            report["owned_native_references_release_returned"] = True
        except BaseException as error:
            errors.append("owned native weakref release guard: " + type(error).__name__ + ": " + str(error))
        if sim is not None:
            try:
                terminal._release_terminal_timeline_caches(sim, report, phase="before_clear")
            except BaseException as error:
                errors.append("owned manager normalization: " + type(error).__name__ + ": " + str(error))
            try:
                if type(sim).instance() is not sim:
                    raise RuntimeError("Refuse to clear a simulation singleton not owned by this entry")
                sim.clear_instance()
                if type(sim).instance() is not None:
                    raise RuntimeError("Simulation singleton remains after clear_instance returned")
                report["simulation_clear_instance_returned"] = True
            except BaseException as error:
                errors.append("sim.clear_instance: " + type(error).__name__ + ": " + str(error))
            try:
                scene_backend_refs = terminal._release_terminal_timeline_caches(sim, report, phase="after_clear")
            except BaseException as error:
                errors.append("terminal SDK cache release: " + type(error).__name__ + ": " + str(error))
        try:
            report["post_context_clear_gc_collected"] = gc.collect()
            report["post_context_clear_gc_returned"] = True
            if (report["terminal_scene_backend_release_returned"] is not True
                    or type(scene_backend_refs) is not tuple or not scene_backend_refs
                    or len(scene_backend_refs) != report["terminal_scene_backend_existing_view_count"]
                    or any(type(ref) is not weakref.ReferenceType for ref in scene_backend_refs)):
                raise RuntimeError("Terminal backend must return its complete existing-view weakrefs")
            report["scene_backend_views_live_after_post_context_gc"] = sum(
                ref() is not None for ref in scene_backend_refs)
            if report["scene_backend_views_live_after_post_context_gc"] != 0:
                raise RuntimeError("Native scene backend views survive before App teardown")
            report["terminal_scene_backend_weakrefs_released"] = True
        except BaseException as error:
            errors.append("post-context backend weakref guard: " + type(error).__name__ + ": " + str(error))
        if app is not None:
            try:
                app.close()
                report["app_close_returned"] = True
            except BaseException as error:
                errors.append("app.close: " + type(error).__name__ + ": " + str(error))
        report["cleanup_errors"] = errors
        flags = ("simulation_stop_returned", "simulation_clear_instance_returned", "app_close_returned",
            "owned_world_native_release_returned", "owned_native_references_release_returned",
            "owned_asset_simulation_views_release_returned", "pre_context_clear_gc_returned", "post_context_clear_gc_returned",
            "owned_manager_normalization_returned", "core_terminal_shutdown_returned",
            "terminal_timeline_cache_compat_returned", "terminal_scene_backend_release_returned",
            "terminal_scene_backend_weakrefs_released")
        report["cleanup_succeeded"] = not errors and all(report[name] is True for name in flags)
    finally:
        if type(tracked_assets) is list:
            tracked_assets.clear()
        app = world = sim_type = sim = collector = expert = owner = release_callback = receipts = terminal = tracked_assets = None
        receipt = refs = assets = left = right = scene_backend_refs = None


def _run(args, report):
    sys.path.insert(0, str(ROOT))
    if "torch" in sys.modules:
        raise RuntimeError("Fresh process required: simulator bootstrap must precede Torch")
    records = report["source_records"] = _sources()
    manifest, manifest_document = _manifest_preflight(args.manifest)
    train = manifest_document["splits"]["train"]
    report["AMASS_manifest"] = manifest
    report["manifest_preflight_before_App"] = True
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    origins = report["actual_import_origins"] = {"bootstrap": _origin(import_simulator_before_torch, "bootstrap", records)}
    launcher = import_simulator_before_torch("isaaclab")
    app, world, sim_type, creation_attempted = None, None, None, False
    normal_release_callback, asset = None, None
    owned_release_receipts, owned_asset_tracker = [], []
    terminal = _load(PINS["terminal_cleanup"][0], "_probe_terminal_lifetime_helpers", records["terminal_cleanup"])
    origins["terminal_cleanup"] = _origin(terminal, "terminal_cleanup", records)
    terminal._verify_overlay_inputs(report)
    try:
        app = launcher({"headless": True, "device": args.device, "enable_cameras": False, "fast_shutdown": False,
            "kit_args": "--/app/renderer/enabled=false --/renderer/enabled=false " + terminal.OVERLAY_KIT_ARGS}).app
        report["app_constructors_returned"] = 1
        terminal._verify_loaded_overlay(report)
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
        report["AMASS_bank"] = {"path": train["path"], "declared_bytes": train["bytes"], "declared_sha256": train["sha256"],
            "split_loaded": "train", "full_bank_hash_recomputed": False, "bank_is_reference_data_not_policy_weights": True}
        corpus = AMASSCorpus(args.manifest, "train", device=args.device, seed=args.seed, dof_velocity_frame="child_joint")
        world_module = _load(PINS["World"][0], "_probe_reviewed_shared_world", records["World"])
        normal = _load(PINS["normal_factory"][0], "_probe_reviewed_normal_factory", records["normal_factory"])
        normal_release_callback = _owned_release_callback(normal, records, report, owned_release_receipts, owned_asset_tracker)
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
        config = IsaacLabSimulatorConfig(num_envs=2, headless=True, experiment_name="diffusion_amp_reconstruction")
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
            normal_force_bundle_factory=normal.create_normal_force_bundle, episode_steps=CAP,
            normal_release_callback=normal_release_callback)
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
        report["AMASS_manifest_after"], _ = _manifest_preflight(args.manifest)
        if report["AMASS_manifest_after"] != manifest:
            raise ValueError("AMASS manifest changed during actual controls")
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
        try:
            asset = None
            _cleanup_owned_runtime(app, world, sim_type, creation_attempted, normal_release_callback,
                owned_release_receipts, owned_asset_tracker, ACTORS, terminal, report)
        finally:
            app = world = sim_type = asset = normal_release_callback = terminal = None  # noqa: F841 - clear the inlined native asset alias.
            owned_asset_tracker.clear()
            owned_release_receipts = owned_asset_tracker = None


def main():
    args = parse_args()
    args.output = _output(args.output)
    started = time.time()
    report = {"schema": "social.shared_group_world.bounded_real_SDK_probe.v1", "status": "running",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "started_unix_s": started, "app_constructors_returned": 0, "simulation_stop_returned": False,
        "app_close_returned": False, "simulation_clear_instance_returned": False, "cleanup_succeeded": False, "probe_checks_passed": False,
        "completed_controls": 0, "controls": [], "artifacts": [], "applied_action_source": "untrained zero69, canonical57 mask",
        "value_source": "positive synthetic all-N values; no model/critic", "grid_geometry_contract": GEOMETRY,
        "include_ego": False, "episode_cap_controls": CAP, "rendered": False,
        "real_training_or_policy_quality_claimed": False, "SDK_resume_or_independent_capture_audit_claimed": False}
    report.update(_owned_cleanup_defaults())
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
