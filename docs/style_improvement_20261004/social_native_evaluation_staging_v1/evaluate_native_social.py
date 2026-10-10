"""Unexecuted N3..15 social actor evaluation in one original native AMPWorld.

Pure source/resource preflight precedes the caller's sole App. Actual model,
training-result and scene identities must be supplied after the serial gates.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import stat
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "docs/style_improvement_20261004/social_navigation_evaluation_staging_v1/evaluate_four_actor_social.py"
PARENT_BYTES = 27829
PARENT_SHA = "c31c87391ab62c2c33369fc437c6b8967546987a505e885b7954ba9349107d8e"
PRODUCER_BYTES = 13020
PRODUCER_SHA = "7d2a51563f59811a6c6d16d82f880e99accaffbfecc3074e44c48bad64666621"


def _record(path, *, readonly=False):
    path = Path(path).absolute()
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or readonly and before.st_mode & 0o222:
        raise ValueError("Use an ordinary single-link source/resource: " + str(path))
    data = path.read_bytes()
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")
    if len(data) != before.st_size or any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError("File changed while reading")
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


_EXECUTION_RECORD = _record(Path(__file__))


def _native_resource_records(request):
    records = {}
    for role in ("configuration_file", "preflight", "native_map", "probe_script_snapshot"):
        selected = request[role]
        if selected is None:
            continue
        current = _record(selected["path"])
        if current != selected:
            raise ValueError("Original selected native resource changed: " + role)
        records[role] = current
    return records


def _derive_parent(source):
    """Six functional seams plus explicit source/profile/report identities.

    The nine literal edits below reverse the complete c31 bytes. Its _derive
    function still makes the identical ten driver substitutions, and no new
    control/planning/contact/reference/reward/cleanup implementation is copied.
    """
    original, patches = source, []
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("Expected one native entry replacement: " + str(source.count(old)))
        source = source.replace(old, new)
        patches.append((old, new))

    replace('"""Unexecuted social-only evaluation derived from the frozen physical driver.\n\n'
            'No Torch/SDK import at module import. Run only after the registered primary\n'
            'Diffusion comparison and serial network/SDK gates. No training or native scene.\n"""',
            '"""Unexecuted N3..15 social actor evaluation in one original native AMPWorld.\n\n'
            'Stdlib at import; explicit immutable artifacts and original native preflight.\n'
            'No grouped-training World or training resume; actual execution is pending.\n"""')
    replace('    "producer": ("docs/style_improvement_20261004/social_evaluation_observation_staging_v1/observation_producer.py", 17726, "2379d65876775173fbdd111d846fd847beb6b30109ecb7e04aa30f7a569064f7"),\n',
            f'    "producer": ("docs/style_improvement_20261004/social_native_evaluation_staging_v1/native_producer.py", {PRODUCER_BYTES}, "{PRODUCER_SHA}"),\n'
            f'    "parent_entry": ("docs/style_improvement_20261004/social_navigation_evaluation_staging_v1/evaluate_four_actor_social.py", {PARENT_BYTES}, "{PARENT_SHA}"),\n'
            '    "native_deployment": ("CrowdSim/humanoid/application/native_deployment.py", 5429, "af708cc18873424ed57b497473f35d17cecc2c3910b15f4c67ef4d8cd5ab49f0"),\n'
            '    "native_provider": ("CrowdSim/humanoid/runtime/dynbench_scene.py", 46157, "c49282b44d38e65b92cbf71fa1909c3e87f551e8494793aa31c5f68246ac7a73"),\n'
            '    "native_grid": ("CrowdSim/humanoid/runtime/native_grid_scene.py", 33100, "d77e9b5702d47765e045a03f6e79f770c313dc8d4d0ad9e969aead81d40024f5"),\n'
            '    "native_live": ("CrowdSim/humanoid/runtime/dynbench_live.py", 13041, "3d9db0f3ee00dacb458950cdc8551887b599510c06a0cc93d4922c7738ead692"),\n')
    replace('document["schema"] != "local.neutral_smpl.social_navigation_evaluation_profile.v1"',
            'document["schema"] != "local.neutral_smpl.social_native_evaluation_profile.v1"')
    replace('    parser.add_argument("--control-steps", type=int, choices=(4, 30, 6000), default=6000)\n'
            '    parser.add_argument("--output", type=Path, required=True)\n'
            '    parser.add_argument("--preflight-only", action="store_true")\n'
            '    args = parser.parse_args()\n'
            '    for digest in (args.amp_sha256, args.training_result_sha256, args.diffusion_sha256):\n',
            '    parser.add_argument("--actors", type=int, choices=range(3, 16), required=True)\n'
            '    parser.add_argument("--scene-config", type=Path, required=True)\n'
            '    parser.add_argument("--scene-sha256", required=True)\n'
            '    parser.add_argument("--control-steps", type=int, choices=(4, 30, 6000), default=6000)\n'
            '    parser.add_argument("--output", type=Path, required=True)\n'
            '    parser.add_argument("--preflight-only", action="store_true")\n'
            '    args = parser.parse_args()\n'
            '    for digest in (args.amp_sha256, args.training_result_sha256, args.diffusion_sha256, args.scene_sha256):\n')
    old = '''                "scene": _identity(SCENE, expected_sha=SCENE_SHA)}
    from CrowdSim.humanoid.application.deployment import driver_arguments, duration_control_steps
    from CrowdSim.humanoid.application.configuration import parse_amp_args
    from CrowdSim.humanoid.application.resources import preflight_amp_resources
    request = SimpleNamespace(config=HERE / "evaluation_profile.json", manifest=MANIFEST,
        diffusion_checkpoint=args.diffusion_checkpoint, output=output, actors=4, seed=7,
        control_steps=args.control_steps, native_dynbench=False, amp_checkpoint=args.amp_checkpoint, scene_config=SCENE)
    forwarded = driver_arguments(request)
    if duration_control_steps(str(args.control_steps / 30)) != args.control_steps:
        raise ValueError("Control clock is not representable")
    driver_args = parse_amp_args(forwarded)
    preflight = preflight_amp_resources(driver_args, profile)
    source, derivation = _derive(DRIVER.read_text())
    preparation = {"schema": "local.neutral_smpl.social_four_actor_navigation_evaluation.v1",
        "sources": sources, "selected_artifacts": selected, "evaluation_profile": profile_identity,
        "derivation": derivation, "resource_preflight": preflight.as_dict(), "driver_arguments": forwarded,
        "requested_controls": args.control_steps, "requested_actor_controls": args.control_steps * 4,
        "primary_200_seconds": args.control_steps == 6000, "physical_execution_verified": False}
'''
    new = '''                "scene": _identity(args.scene_config, expected_sha=args.scene_sha256)}
    from CrowdSim.humanoid.application.deployment import driver_arguments, duration_control_steps
    from CrowdSim.humanoid.application.configuration import parse_amp_args
    from CrowdSim.humanoid.application.resources import preflight_amp_resources
    from CrowdSim.humanoid.application.native_deployment import preflight_native_scene, make_native_scene_factory
    native_module = sys.modules[preflight_native_scene.__module__]
    if (Path(native_module.__file__).resolve() != Path(sources["native_deployment"]["path"])
            or make_native_scene_factory.__module__ != preflight_native_scene.__module__):
        raise ValueError("Actual native factory/preflight origin differs from its pinned production source")
    native_request = preflight_native_scene(args.scene_config, actors=args.actors)
    native_resources = _native_resource_records(native_request)
    native_factory = make_native_scene_factory(native_request)
    request = SimpleNamespace(config=HERE / "evaluation_profile.json", manifest=MANIFEST,
        diffusion_checkpoint=args.diffusion_checkpoint, output=output, actors=args.actors, seed=7,
        control_steps=args.control_steps, native_dynbench=True, amp_checkpoint=args.amp_checkpoint, scene_config=args.scene_config)
    forwarded = driver_arguments(request)
    if duration_control_steps(str(args.control_steps / 30)) != args.control_steps:
        raise ValueError("Control clock is not representable")
    driver_args = parse_amp_args(forwarded)
    preflight = preflight_amp_resources(driver_args, profile)
    source, derivation = _derive(DRIVER.read_text())
    preparation = {"schema": "local.neutral_smpl.social_native_navigation_evaluation.v1",
        "sources": sources, "selected_artifacts": selected, "evaluation_profile": profile_identity,
        "parent_evaluation_source": _native_parent_record, "native_entry_derivation": _native_entry_derivation,
        "native_scene_request": native_request, "native_resource_bindings": native_resources,
        "derivation": derivation, "resource_preflight": preflight.as_dict(), "driver_arguments": forwarded,
        "actors": args.actors, "requested_controls": args.control_steps,
        "requested_actor_controls": args.control_steps * args.actors,
        "primary_200_seconds": args.control_steps == 6000, "physical_execution_verified": False}
'''
    replace(old, new)
    old = '''        if (config.expert_dof_velocity_frame != "child_joint" or not config.freeze_toes
                or world.num_envs != 4 or not world.shared_world or world.scene_provider is not None):
            raise ValueError("Only the original four-person cuboid lifecycle and canonical child-frame neutral policy are supported")
        module = _load("_social_navigation_observations", sources["producer"])
        producer = module.RawSocialEvaluationObservations.from_world(world,
            actor_ids=torch.arange(4, dtype=torch.int64, device=world.device),
            scene_ids=torch.zeros(4, dtype=torch.int64, device=world.device),
            terrain_sampler=world.height_at, terrain_sampler_semantics=profile["terrain_sampler"],
            policy_description=policy.describe())
'''
    new = '''        if (config.expert_dof_velocity_frame != "child_joint" or not config.freeze_toes
                or world.num_envs != args.actors or world.shared_world is not True or world.scene_provider is None):
            raise ValueError("Require the original shared native lifecycle and canonical child-frame social policy")
        module = _load("_social_native_observations", sources["producer"])
        producer = module.RawSocialNativeEvaluationObservations.from_world(world,
            actor_ids=torch.arange(args.actors, dtype=torch.int64, device=world.device),
            scene_ids=torch.zeros(args.actors, dtype=torch.int64, device=world.device),
            native_request=native_request,
            terrain_sampler=world.height_at, terrain_sampler_semantics=profile["terrain_sampler"],
            policy_description=policy.describe())
'''
    replace(old, new)
    replace('            if _sources() != sources or _profile(HERE / "evaluation_profile.json")[1] != profile_identity:\n',
            '            if (_sources() != sources or _profile(HERE / "evaluation_profile.json")[1] != profile_identity\n'
            '                    or _native_resource_records(native_request) != native_resources):\n')
    replace('        return namespace["main"](evaluation_actor_factory=lambda *unused, **ignored: None)\n',
            '        return namespace["main"](evaluation_actor_factory=lambda *unused, **ignored: None,\n'
            '                                 native_scene_factory=native_factory)\n')
    # The parent name/metadata is replaced without modifying its execution loop.
    replace('"_social_four_actor_derived_driver"', '"_social_native_derived_driver"')
    if len(patches) != 9:
        raise ValueError("Exactly nine declared literal entry edits are required")
    reverse = source
    for old, new in reversed(patches):
        if reverse.count(new) != 1:
            raise ValueError("Native entry edit is not uniquely reversible")
        reverse = reverse.replace(new, old)
    if reverse != original:
        raise ValueError("Native entry must reverse to every c31 byte")
    before, after = ast.parse(original), ast.parse(source)
    def untouched(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name not in {"main", "_profile"}}
    if untouched(before) != untouched(after):
        raise ValueError("A parent helper, full training/USD gate or driver derivation changed")
    return source, {"literal_replacement_count": len(patches), "whole_source_reverse_equal": True,
                    "untouched_helper_AST_equal": True, "derived_sha256": hashlib.sha256(source.encode()).hexdigest()}


def main():
    parent = _record(PARENT, readonly=True)
    producer = _record(HERE / "native_producer.py", readonly=True)
    if (parent["bytes"] != PARENT_BYTES or parent["sha256"] != PARENT_SHA
            or producer["bytes"] != PRODUCER_BYTES or producer["sha256"] != PRODUCER_SHA
            or _record(Path(__file__), readonly=True) != _EXECUTION_RECORD):
        raise ValueError("Use the exact reviewed parent/producer/entry sources")
    data = PARENT.read_bytes()
    if len(data) != PARENT_BYTES or hashlib.sha256(data).hexdigest() != PARENT_SHA:
        raise ValueError("Parent changed before derivation")
    source, derivation = _derive_parent(data.decode())
    backend = ModuleType("_social_native_evaluation_entry")
    backend.__file__ = str(Path(__file__).resolve())
    backend._native_parent_record = parent
    backend._native_entry_derivation = derivation
    backend._native_resource_records = _native_resource_records
    exec(compile(source, backend.__file__, "exec"), backend.__dict__)
    return backend.main()


if __name__ == "__main__":
    raise SystemExit(main())
