"""Read-only initial conditions for the fixed N4 same-AMP planner comparison.

No Torch or SDK import at module import. Captures precede planner construction;
matching snapshots do not establish process completion or navigation quality.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FOOT_CAPTURE_SHA = "89c48fde37c05eaa57f4eacf2fb3eb072b087c0d9a46acff3f7d23729f28d39b"
SCHEMA = "shared_social.before_planner_initial_condition.v1"
COMMON_FIELDS = {"rigid_body_pos": (4, 24, 3), "rigid_body_rot": (4, 24, 4),
                 "rigid_body_vel": (4, 24, 3), "rigid_body_ang_vel": (4, 24, 3),
                 "dof_pos": (4, 69), "dof_vel": (4, 69)}
WORLD_FIELDS = {"origins": (4, 2), "age": (4,), "episode_ids": (4,), "reference_ids": (4,),
                "amp_history": (4, 10, 195), "amp_history_episode_ids": (4, 10),
                "planar_history": (4, 91, 5), "history_available": (4, 91)}
NATIVE_FIELDS = {"body_link_pos_w": (4, 24, 3), "body_link_quat_w": (4, 24, 4),
                 "body_link_lin_vel_w": (4, 24, 3), "body_link_ang_vel_w": (4, 24, 3),
                 "body_com_pos_w": (4, 24, 3), "local_com_pose_b": (4, 24, 7),
                 "body_masses_kg": (4, 24), "mass_weighted_world_com": (4, 3),
                 "com_composition_max_error_m": ()}


def _identity(path):
    from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record
    return record(path)


def _array(value, name, *, shape=None):
    if hasattr(value, "detach"):
        value = value.detach().clone().cpu().numpy()
    result = np.asarray(value)
    if (result.dtype.kind not in "bifu" or result.size == 0
            or (result.dtype.kind == "f" and not np.isfinite(result).all())
            or (shape is not None and result.shape != shape)):
        raise ValueError("Missing, nonfinite or incorrectly shaped initial channel: " + name)
    return result.copy()


def _description(array):
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def rng_snapshot(corpus_generator):
    """Use documented state getters, without seeding, restoring or drawing.

    https://docs.pytorch.org/docs/2.11/random.html
    https://docs.pytorch.org/docs/2.11/generated/torch.cuda.get_rng_state_all.html
    https://numpy.org/doc/2.3/reference/random/generated/numpy.random.RandomState.get_state.html
    https://docs.python.org/3.12/library/random.html#random.getstate
    """
    import torch
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    if python_state[0] != 3 or numpy_state[0] != "MT19937":
        raise ValueError("Unexpected Python/NumPy global generator state representation")
    arrays = {"rng_python_MT": np.asarray(python_state[1], dtype=np.uint64),
              "rng_numpy_MT": numpy_state[1].copy(),
              "rng_torch_CPU": _array(torch.random.get_rng_state(), "CPU RNG"),
              "rng_corpus_CPU": _array(corpus_generator.get_state(), "corpus RNG")}
    cuda_initialized = torch.cuda.is_initialized()
    if cuda_initialized:
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            arrays[f"rng_torch_CUDA_{index}"] = _array(state, "CUDA RNG")
    metadata = {"python_version": python_state[0], "python_cached_gaussian": python_state[2],
                "numpy_generator": numpy_state[0], "numpy_position": int(numpy_state[2]),
                "numpy_has_gauss": int(numpy_state[3]), "numpy_cached_gaussian": float(numpy_state[4]),
                "CUDA_initialized": cuda_initialized, "CUDA_getter_called": cuda_initialized,
                "CUDA_device_state_count": len([key for key in arrays if key.startswith("rng_torch_CUDA_")])}
    json.dumps(metadata, allow_nan=False)
    return arrays, metadata


def _equal_arrays(left, right):
    return (left.keys() == right.keys() and all(_description(left[key]) == _description(right[key]) for key in left))


def collect_snapshot(world, ground_capture):
    """Clone already owned COMMON and native link/COM channels, no physics step.

    The existing FootCaptureAdapter performs native-order mapping and verifies
    COM composition. This helper never samples contact forces or starts Kit.
    Injectable objects in CPU tests are only contract fixtures.
    """
    if (world.num_envs != 4 or not world.shared_world or world.scene_provider is not None
            or world.validation_scene is None or ground_capture.world is not world):
        raise ValueError("Initial condition requires the original complete shared N4 fixture")
    before = ground_capture.counter()
    if type(before) is not int or before < 0:
        raise ValueError("Initial condition needs an observed Lab submitted-step counter")
    state = world.state()
    if getattr(state.state_conversion, "value", None) != "common":
        raise ValueError("Initial robot state must already use COMMON ordering and XYZW")
    arrays = {"common_" + key: _array(getattr(state, key), key, shape=shape)
              for key, shape in COMMON_FIELDS.items()}
    for name, shape in WORLD_FIELDS.items():
        arrays["world_" + name] = _array(getattr(world, name), name, shape=shape)
    if (arrays["world_age"].dtype.kind not in "iu" or np.any(arrays["world_age"] != 0)
            or arrays["world_episode_ids"].dtype.kind not in "iu" or np.any(arrays["world_episode_ids"] != 1)
            or arrays["world_reference_ids"].dtype.kind not in "iu"
            or arrays["world_history_available"].dtype != np.bool_
            or not arrays["world_history_available"][:, -1].all()):
        raise ValueError("Expected initial reset1, age0, genuine history masks and integer IDs")
    arrays["actor_ids"] = np.arange(4, dtype=np.int64)
    arrays["scene_ids"] = np.zeros(4, dtype=np.int64)
    arrays["scene_routes_xy"] = _array(world.validation_scene.routes, "routes")
    if arrays["scene_routes_xy"].ndim != 3 or arrays["scene_routes_xy"].shape[0] != 4 or arrays["scene_routes_xy"].shape[2] != 2:
        raise ValueError("Expected all four registered routes")
    native = ground_capture._state_snapshot()
    expected = NATIVE_FIELDS
    if native.keys() != expected.keys():
        raise ValueError("Existing native initial snapshot channel set changed")
    arrays.update({"native_" + key: _array(native[key], key, shape=shape) for key, shape in expected.items()})
    for common, measured in (("rigid_body_pos", "body_link_pos_w"), ("rigid_body_rot", "body_link_quat_w"),
                             ("rigid_body_vel", "body_link_lin_vel_w"), ("rigid_body_ang_vel", "body_link_ang_vel_w")):
        if not np.array_equal(arrays["common_" + common], arrays["native_" + measured]):
            raise ValueError("COMMON and native same-instant link channels disagree: " + common)
    after = ground_capture.counter()
    if after != before:
        raise ValueError("Initial condition read unexpectedly advanced physics")
    return arrays, {"physics_step_before": before, "physics_step_after": after,
                    "counter_semantics": "Lab submitted-step count, not native PhysX simulation time",
                    "native_body_names": list(ground_capture.native_names),
                    "common_body_names": list(ground_capture.common_names), "quaternion_order": "xyzw"}


def publish_snapshot(output, arrays, contract):
    """Create new files only; no overwrite and no zero-filled missing channels."""
    output = Path(output)
    npz = output / "paired_initial_arrays.npz"
    document = output / "paired_initial_condition.json"
    if npz.exists() or document.exists():
        raise FileExistsError("Initial condition files already exist")
    validate_channels(arrays, contract)
    with npz.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    receipt = {"schema": SCHEMA, "contract": contract, "arrays": _identity(npz),
               "channels": {key: _description(value) for key, value in arrays.items()},
               "actual_process_exit_verified": False, "navigation_quality_established": False}
    with document.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"document": _identity(document), "arrays": receipt["arrays"],
            "boundary": "after actor admission/reseed, before high-level planner construction",
            "physics_or_RNG_advanced": False, "actual_process_exit_verified": False}


def capture_initial_condition(world, ground_capture, output, report, args):
    """Live entry hook; keep only detached CPU arrays and JSON in the report."""
    from CrowdSim.humanoid.amp_world import AMPWorld
    from CrowdSim.humanoid.foot_capture import FootCaptureAdapter
    if (type(world) is not AMPWorld or type(ground_capture) is not FootCaptureAdapter
            or _identity(ROOT / "CrowdSim/humanoid/foot_capture.py")["sha256"] != FOOT_CAPTURE_SHA):
        raise ValueError("Require exact source-bound original world and live capture adapter")
    rng_before, rng_metadata = rng_snapshot(world.corpus.generator)
    arrays, clock = collect_snapshot(world, ground_capture)
    second, second_clock = collect_snapshot(world, ground_capture)
    if not _equal_arrays(arrays, second) or clock != second_clock:
        raise ValueError("Native or COMMON initial state changed during read")
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    arguments.pop("output")
    actor = report["evaluation_actor"]
    actor_keys = ("checkpoint", "configuration_document", "trainer_source_contract", "RMS_counts",
                  "observations", "action_contract", "normalization", "grid_geometry_contract")
    contract = {"capture_kind": "live_original_world",
                "boundary": "after actor admission/reseed, before high-level planner construction",
                "arguments_excluding_output": arguments, "actor": {key: actor[key] for key in actor_keys},
                "physics": report["physics"], "reference": report["reference_adaptation"],
                "runtime_numerics": report["runtime_numerics"], "rng_preparation": report["evaluation_rng_preparation"],
                "scene": world.validation_scene.description, "clock": clock, "rng": rng_metadata,
                "initial_capture_source": _identity(Path(__file__)),
                "native_capture_source": _identity(ROOT / "CrowdSim/humanoid/foot_capture.py"),
                "contact_forces_sampled": False, "extra_physics_steps": 0}
    rng_after, after_metadata = rng_snapshot(world.corpus.generator)
    if not _equal_arrays(rng_before, rng_after) or rng_metadata != after_metadata:
        raise ValueError("Initial condition read consumed RNG")
    arrays.update(rng_before)
    result = publish_snapshot(output, arrays, contract)
    rng_final, final_metadata = rng_snapshot(world.corpus.generator)
    if (not _equal_arrays(rng_before, rng_final) or rng_metadata != final_metadata
            or ground_capture.counter() != clock["physics_step_after"]):
        raise ValueError("Initial condition publication changed RNG or physics; files retained for diagnosis")
    return result


def validate_channels(arrays, contract):
    """Reject empty/partial pairing receipts before any match can be reported."""
    if contract.get("capture_kind") not in {"live_original_world", "CPU_contract_fixture"}:
        raise ValueError("Missing initial capture kind")
    clock, rng = contract["clock"], contract["rng"]
    if (type(clock["physics_step_before"]) is not int or clock["physics_step_before"] < 0
            or clock["physics_step_before"] != clock["physics_step_after"]):
        raise ValueError("Initial physics counters differ")
    shapes = {"common_" + key: value for key, value in COMMON_FIELDS.items()}
    shapes.update({"world_" + key: value for key, value in WORLD_FIELDS.items()})
    shapes.update({"native_" + key: value for key, value in NATIVE_FIELDS.items()})
    shapes.update(actor_ids=(4,), scene_ids=(4,), rng_python_MT=(625,), rng_numpy_MT=(624,))
    variable = {"scene_routes_xy", "rng_torch_CPU", "rng_corpus_CPU"}
    count = rng["CUDA_device_state_count"]
    if (type(count) is not int or count < 0 or type(rng["CUDA_initialized"]) is not bool
            or rng["CUDA_getter_called"] != rng["CUDA_initialized"]
            or (rng["CUDA_initialized"] and count == 0) or (not rng["CUDA_initialized"] and count != 0)):
        raise ValueError("CUDA initial RNG metadata disagrees")
    variable.update(f"rng_torch_CUDA_{index}" for index in range(count))
    if set(arrays) != set(shapes) | variable:
        raise ValueError("Initial snapshot is partial or contains unregistered channels")
    for key, value in arrays.items():
        _array(value, key, shape=shapes.get(key))
    if (arrays["scene_routes_xy"].ndim != 3 or arrays["scene_routes_xy"].shape[0] != 4
            or arrays["scene_routes_xy"].shape[1] < 2 or arrays["scene_routes_xy"].shape[2] != 2):
        raise ValueError("Initial routes require all four actors and at least two vertices")
    if (arrays["world_history_available"].dtype != np.bool_
            or not arrays["world_history_available"][:, -1].all()
            or not np.array_equal(arrays["actor_ids"], np.arange(4))
            or np.any(arrays["scene_ids"] != 0) or np.any(arrays["world_age"] != 0)
            or np.any(arrays["world_episode_ids"] != 1)):
        raise ValueError("Initial actors, scene IDs, age/reset or history masks differ")
    for key in ("actor_ids", "scene_ids", "world_age", "world_episode_ids", "world_reference_ids", "world_amp_history_episode_ids"):
        if arrays[key].dtype.kind not in "iu":
            raise ValueError("Initial identifiers require genuine integers: " + key)
    for key in variable - {"scene_routes_xy"}:
        if arrays[key].dtype != np.uint8 or arrays[key].ndim != 1:
            raise ValueError("Torch RNG state requires a one-dimensional ByteTensor archive")
    if arrays["rng_python_MT"].dtype != np.uint64 or arrays["rng_numpy_MT"].dtype != np.uint32:
        raise ValueError("Unexpected MT generator key representation")
    float_fields = set(shapes) - {"world_history_available", "actor_ids", "scene_ids", "world_age", "world_episode_ids",
                                 "world_reference_ids", "world_amp_history_episode_ids", "rng_python_MT", "rng_numpy_MT"}
    if any(arrays[key].dtype != np.float32 for key in float_fields | {"scene_routes_xy"}):
        raise ValueError("Initial live floating channels must use the original FP32 contract")


def read_snapshot(path):
    """Validate the archive SHA, then every channel's shape/dtype/bytes."""
    path = Path(path)
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != SCHEMA:
        raise ValueError("Wrong paired initial condition schema")
    if _identity(receipt["arrays"]["path"]) != receipt["arrays"]:
        raise ValueError("Initial archive identity differs")
    with np.load(receipt["arrays"]["path"], allow_pickle=False) as data:
        if set(data.files) != set(receipt["channels"]):
            raise ValueError("Initial archive channels differ")
        arrays = {key: _array(data[key], key) for key in data.files}
        validate_channels(arrays, receipt["contract"])
        for key, value in arrays.items():
            if _description(value) != receipt["channels"][key]:
                raise ValueError("Initial archive payload differs: " + key)
    return receipt


def compare_initial_conditions(left, right):
    """Admission of initial pairing only; still require actual exits/raw audits."""
    baseline, candidate = read_snapshot(left), read_snapshot(right)
    if baseline["contract"] != candidate["contract"] or baseline["channels"] != candidate["channels"]:
        raise ValueError("Planner arms do not share exact recorded initial conditions and resources")
    return {"status": "initial_conditions_matched", "snapshots": [_identity(left), _identity(right)],
            "channel_count": len(baseline["channels"]), "same_initial_array_bytes": True,
            "same_resources_protocol_and_RNG": True, "actual_process_exit_verified": False,
            "capture_kind": baseline["contract"]["capture_kind"],
            "navigation_quality_established": False, "author_experiment_reproduced": False}
