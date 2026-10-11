"""Versioned dependency selection for caller-owned Sim 6.1 evaluators.

Full hashes admit reviewed source versions; they do not certify an execution.
The sealed evaluators, metrics and benchmark populations remain immutable.
"""
import hashlib
import ast
from pathlib import Path

from protomotions.repository import repository_root
from social_navigation.artifact_identity import artifact_path
from .social_evaluation_lifecycle import record

ROOT = repository_root(__file__)
ENTRY_EDITS = {
    "scripts/evaluate_shared_social_navigation.py": (
        "b556de1a9c9c1a6b5984e1484c9a67f490f549f4346649746a84dd7ab9726558",
        [('    from CrowdSim.humanoid.runtime.evaluation_source_migration import migrate_pins\n'
          '    binding["dependency_migration"] = migrate_pins(namespace)\n', ''),
         ('        (\'        owned["world"] = factory(*positional, **keywords)\\n\',\n         \'        constructor = (_migration_lifecycle._sdk61.construct_world\\n\'\n         \'                       if _migration_lifecycle._sdk61 is not None else lambda f, *a, **k: f(*a, **k))\\n\'\n         \'        owned["world"] = constructor(factory, *positional, **keywords)\\n\'),\n', '')]),
    "scripts/evaluate_shared_social_tracking.py": (
        "ff866433ce91feb2a006a137ed437ae7bab92b8b6b0c9daaf09f244521b3767c",
        [('    from CrowdSim.humanoid.runtime.evaluation_source_migration import migrate_pins\n'
          '    dependency_migration = migrate_pins(parent)\n', ''),
         ('"entry": record(Path(__file__)), "dependency_migration": dependency_migration}',
          '"entry": record(Path(__file__))}')]),
}
# relative path, historical size/hash, current size/hash, migration scope
DEPENDENCIES = {
    "native_deployment": ("src/CrowdSim/humanoid/application/native_deployment.py", 5429,
        "af708cc18873424ed57b497473f35d17cecc2c3910b15f4c67ef4d8cd5ab49f0", 5610,
        "a3a8097e5d63b03e5f96f24e6246c333b2103f3628c6209139e2aff2c564cd96", "explicit_artifact_location_admission"),
    "native_provider": ("src/CrowdSim/humanoid/runtime/dynbench_scene.py", 46157,
        "c49282b44d38e65b92cbf71fa1909c3e87f551e8494793aa31c5f68246ac7a73", 53151,
        "68bd2171ad9707c3f280c374d73538b8c532764a42e1f90335c6375090595d28", "explicit_snapshot_and_live_scene_ownership"),
    "native_grid": ("src/CrowdSim/humanoid/runtime/native_grid_scene.py", 33100,
        "d77e9b5702d47765e045a03f6e79f770c313dc8d4d0ad9e969aead81d40024f5", 41778,
        "b5563bd5a509cdb02ae33e57ad34ab7a0524825ad16e481ef631531d59308165", "sim61_collision_and_reset_runtime"),
    "native_live": ("src/CrowdSim/humanoid/runtime/dynbench_live.py", 13041,
        "3d9db0f3ee00dacb458950cdc8551887b599510c06a0cc93d4922c7738ead692", 13565,
        "4c7f5384658778623b96ceeca0c52cda938d57e644b8084f4461e21ebac42bca", "repository_location_live_identity_and_USD_point3f"),
    "deployment": ("src/CrowdSim/humanoid/application/deployment.py", 15526,
        "37f6f81538ae422bc8fa2505ce219c926b8252844cc1a0883c5d39620e42e8f0", 15569,
        "3a0f6bf99b04e8f122020a87eef70c40f213cabed2b06a9b97cfc3e3eac8795c", "repository_location"),
    "resources": ("src/CrowdSim/humanoid/application/resources.py", 15989,
        "3268a10882931cb5ad0e23506b71fd22fb21d975e9aba948e0a08fb427132a5b", 16019,
        "1a7e130ebcaff396c23238598f92e9407f5de37c0e214a63d4e235d800794bb5", "repository_location"),
    "world": ("src/CrowdSim/humanoid/amp_world.py", 24834,
        "1b63a4c8ac7d4bf2f9ed523edc97b41dca89888cdd6c7dcb2234826d240d7f83", 30450,
        "a114f1b0763b7a2307a3f7fdd9409c38b4385ed4db02a9fbf36a56854dee554a", "native_contact_and_reset_runtime"),
    "observations": ("src/CrowdSim/humanoid/amp/observations.py", 10438,
        "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba", 10586,
        "000f3980a8f4d3c1fac929c57029dd05a9ce19b1d8ce82423dad273c9315e599", "scripted_observation_origins"),
    "simulator": ("src/protomotions/simulator/isaaclab/simulator.py", 53687,
        "207ff0cbfb65fd48092e73e36c97b2e1b8e2ab55cae79bc2e7edf562528e2ad5", 55888,
        "8cd33819027fb13b457be8ee725eae6bd6bd013036add6359b3fa734634345b1", "sim61_state_actuator_and_lifecycle_API"),
}


def migrate_observation_pins(module):
    """Select the two reviewed dependencies in an owned raw-observation core."""
    selected = dict(module.PINS)
    for key in ("world", "observations"):
        relative, old_size, old_sha, size, sha, _ = DEPENDENCIES[key]
        declaration = selected[key]
        if (len(declaration) != 4 or declaration[3] is not False
                or declaration[1:3] != (old_size, old_sha)
                or artifact_path(ROOT / declaration[0]) != ROOT / relative):
            raise ValueError("Unregistered raw observation dependency: " + key)
        identity = record(ROOT / relative)
        if (identity["bytes"], identity["sha256"]) != (size, sha):
            raise ValueError("Raw observation dependency changed: " + key)
        selected[key] = (relative, size, sha, False)
    for key, (path, size, sha, readonly) in selected.items():
        selected[key] = (str(artifact_path(ROOT / path)), size, sha, readonly)
    module.PINS = selected


def migrate_pins(module):
    """Replace only exact registered historical dependencies in this namespace."""
    selected = dict(module.PINS)
    changes = {}
    for role, (relative, old_size, old_sha, size, sha, scope) in DEPENDENCIES.items():
        if role not in selected:
            continue
        path, expected_size, expected_sha = selected[role]
        absolute = Path(path) if Path(path).is_absolute() else ROOT / path
        if (artifact_path(absolute, root=ROOT) != ROOT / relative
                or (expected_size, expected_sha) != (old_size, old_sha)):
            raise ValueError("Unregistered evaluator dependency: " + role)
        actual = record(ROOT / relative)
        if (actual["bytes"], actual["sha256"]) != (size, sha):
            raise ValueError("Registered evaluator dependency changed: " + role)
        proof = {"scope": scope, "historical": {"path": str(absolute), "bytes": old_size, "sha256": old_sha},
                 "current": actual}
        if scope == "repository_location":
            restored = (ROOT / relative).read_text().replace("from protomotions.repository import repository_root\n\n", "")
            restored = restored.replace("repository_root(__file__)", "Path(__file__).resolve().parents[3]")
            if role == "resources":
                current = ('    from social_navigation.resource_bindings import resolve_path\n'
                           '    return resolve_path(value, relative_to=relative_to or Path.cwd())\n')
                original = ('    path = Path(value)\n'
                            '    if not path.is_absolute() and relative_to is not None:\n'
                            '        path = Path(relative_to) / path\n'
                            '    return path.resolve()\n')
                if restored.count(current) != 1:
                    raise ValueError("Resource resolver seam changed")
                restored = restored.replace(current, original, 1)
                proof["resource_resolver"] = record(ROOT / "src/social_navigation/resource_bindings.py")
                if proof["resource_resolver"]["sha256"] != "ee711511624c6b5c399c5b2cb11ce4908a3522feb97c9d7ccff765fc47d12461":
                    raise ValueError("Resource resolver changed")
            if len(restored.encode()) != old_size or hashlib.sha256(restored.encode()).hexdigest() != old_sha:
                raise ValueError("Repository-location inverse differs: " + role)
            proof["whole_source_location_inverse"] = True
        selected[role] = (str(ROOT / relative) if isinstance(path, str) else ROOT / relative, size, sha)
        changes[role] = proof
    module.PINS = selected
    return {"version": "evaluation_dependencies.sim61.v2", "adapter": record(__file__),
            "changes": changes, "sealed_sources_modified": False,
            "runtime_execution_verified": False, "physics_equivalence_claimed": False}


def verify_entry(path, historical_sha):
    """Prove that the maintained entry adds only explicit dependency selection."""
    path = Path(path).resolve()
    relative = str(path.relative_to(ROOT))
    if relative not in ENTRY_EDITS or historical_sha != ENTRY_EDITS[relative][0]:
        raise ValueError("Unregistered evaluation entry migration")
    original = path.read_text()
    if relative == "scripts/evaluate_shared_social_tracking.py":
        from CrowdSim.humanoid.application.historical_recovery import SNAPSHOTS
        snapshot, _, digest = SNAPSHOTS[relative]
        archived = (ROOT / snapshot).read_text()
        if (hashlib.sha256(original.encode()).hexdigest() != "9f5ddb9981aa39d67719cb3278c55de79b93ee537ac7d15b794629d061a62c58"
                or hashlib.sha256(archived.encode()).hexdigest() != digest or digest != historical_sha):
            raise ValueError("Registered tracking admission source changed")
        def unchanged_functions(source):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name not in {"load_reference_functions", "admit_training"}}
        constructor_seam = ('        constructor = getattr(lifecycle._sdk61, "construct_world", lambda factory, *a, **k: factory(*a, **k))\n'
                            '        world = constructor(AMPWorld, app, corpus, num_envs=1, shared_world=False,\n')
        if original.count(constructor_seam) != 1:
            raise ValueError("Tracking constructor ownership seam changed")
        restored = original.replace(constructor_seam, '        world = AMPWorld(app, corpus, num_envs=1, shared_world=False,\n', 1)
        if unchanged_functions(restored) != unchanged_functions(archived):
            raise ValueError("Tracking measurement functions changed")
        return {"historical_sha256": historical_sha, "current": record(path),
                "metadata_admission_version": "explicit_popart_families.v2",
                "measurement_function_AST_unchanged": True, "adapter": record(__file__)}
    restored = original
    for current, old in ENTRY_EDITS[relative][1]:
        if restored.count(current) != 1:
            raise ValueError("Evaluation dependency selector changed")
        restored = restored.replace(current, old, 1)
    if hashlib.sha256(restored.encode()).hexdigest() != historical_sha:
        raise ValueError("Evaluation source inverse differs")
    return {"historical_sha256": historical_sha, "current": record(path),
            "complete_source_inverse": True, "adapter": record(__file__)}
