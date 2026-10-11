"""Admit the terminal-only cache fix without rewriting historical registrations."""
import hashlib
from pathlib import Path

from protomotions.repository import repository_root

ROOT = repository_root(__file__)
LIFECYCLE = ROOT / "src/CrowdSim/humanoid/runtime/sdk61_lifecycle.py"
HELPER = ROOT / "src/protomotions/simulator/isaaclab/utils/sdk61_owned_close.py"
HISTORICAL_BYTES = 22539
HISTORICAL_SHA = "b4f85259dd7a13164213fb2c28d2e0dc1d359a0dba645d6555541ac257f1e676"
HELPER_SHA = "8c91e50e665fee2b8b52fa7ad7b6b4eb9972bb74fdec701cafcc4baa0181e7a5"
TERMINAL_ADDITION = (
    '            from protomotions.simulator.isaaclab.utils.sdk61_owned_close import _release_physx_replicator_cache\n'
    '            def release_cache():\n'
    '                if SimulationContext.instance() is not None:\n'
    '                    raise RuntimeError("Cannot release terminal PhysX cache before the context is closed")\n'
    '                _release_physx_replicator_cache(report)\n'
    '            attempt("SDK terminal PhysX replicator cache", release_cache)\n'
)


def admit_dependency(inventory):
    """Derive one current dependency in the caller's parsed manifest only."""
    from social_navigation.artifact_identity import artifact_path
    keys = [key for key in inventory['dependencies'] if artifact_path(key) == LIFECYCLE]
    if len(keys) != 1:
        raise ValueError('Require exactly one registered terminal lifecycle dependency')
    key = keys[0]
    row = inventory["dependencies"][key]
    if (row["path"] != str(LIFECYCLE.relative_to(ROOT))
            or (row["bytes"], row["sha256"]) != (HISTORICAL_BYTES, HISTORICAL_SHA)):
        raise ValueError("Unregistered terminal lifecycle dependency")
    source = LIFECYCLE.read_bytes()
    addition = TERMINAL_ADDITION.encode()
    if source.count(addition) != 1:
        raise ValueError("Terminal cache migration seam changed")
    restored = source.replace(addition, b"", 1)
    if len(restored) != HISTORICAL_BYTES or hashlib.sha256(restored).hexdigest() != HISTORICAL_SHA:
        raise ValueError("Terminal cache migration changed another lifecycle byte")
    helper = HELPER.read_bytes()
    if hashlib.sha256(helper).hexdigest() != HELPER_SHA:
        raise ValueError("Registered terminal cache helper changed")
    row.update(bytes=len(source), sha256=hashlib.sha256(source).hexdigest())
    inventory["terminal_cache_migration"] = {
        "version": "terminal_PhysX_cache.sim61.v1",
        "historical_sha256": HISTORICAL_SHA,
        "current_sha256": row["sha256"],
        "whole_source_inverse_verified": True,
        "helper": {"path": str(HELPER), "bytes": len(helper), "sha256": HELPER_SHA},
        "historical_manifest_modified": False,
        "physical_execution_verified": False,
    }
    return inventory
