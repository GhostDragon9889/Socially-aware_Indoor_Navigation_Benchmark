"""CPU-only extracted SDK config-copy contract, never an SDK spawn test."""
from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SDK = ROOT.parent / "IsaacLab-Protomotions/source/isaaclab/isaaclab"
OUTPUT = Path(__file__).resolve().parent


def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    # Importing this prepared provider is stdlib-only. It admits only the
    # selected metadata, not a new native simulation or physical scene.
    from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider

    actual = json.loads((ROOT / "docs/style_improvement_20261002/native_grid_scene_v1/actual_warehouse_verification.json").read_text())
    provider = DynBenchSceneProvider.from_preflight(actual["preflight"]["path"],
        expected_sha256=actual["preflight"]["sha256"],
        probe_script_snapshot=actual["verified_script_copy"]["path"])
    provider.prepare_simulation_cfg(SimpleNamespace(enable_scene_query_support=False))

    config_source = SDK / "utils/configclass.py"
    tree = ast.parse(config_source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_custom_post_init")
    namespace = {"deepcopy": deepcopy,
        # These string wrappers do not control copying/callable filtering;
        # this reproduction isolates the original helper's ownership logic.
        "_wrap_resolvable_strings": lambda value, **kwargs: value,
        "_field_module_dir": lambda obj, key: None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(config_source), "exec"), namespace)
    post_init = namespace["_custom_post_init"]

    spawn = SimpleNamespace(func=provider._spawn, usd_path=provider._data["source"]["path"])
    post_init(spawn)
    direct_callable_owner_unchanged = spawn.func.__self__ is provider
    config = SimpleNamespace(spawn=spawn, prim_path="/__PaperDynBenchSceneProvider")
    post_init(config)
    copied_owner = config.spawn.func.__self__
    owner_is_copy = copied_owner is not provider
    config_and_spawn_are_copied = config.spawn is not spawn
    copied_source_same = copied_owner._data == provider._data
    copied_precreate_flag_true = copied_owner._query_hook_applied is True
    copied_method_same_function = config.spawn.func.__func__ is provider._spawn.__func__
    assert all((direct_callable_owner_unchanged, owner_is_copy, config_and_spawn_are_copied,
                copied_source_same, copied_precreate_flag_true, copied_method_same_function))
    # A plain flag write models ownership, without invoking the spawn callback.
    copied_owner._spawned = True
    copied_flag_independent = copied_owner._spawned and not provider._spawned
    assert copied_flag_independent
    forbidden = [name for name in sys.modules if name == "pxr" or name.startswith(("omni.", "isaaclab", "isaacsim"))]
    assert not forbidden
    report = {"schema": "native_scene_config_copy_CPU_review.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_extracted_copy_contract_not_SDK_execution", "CPU_only": True,
        "GPU_used": False, "SDK_started": False, "spawn_callback_called": False,
        "string_wrappers_stubbed_for_copy_only": True,
        "helper_source": record(config_source), "helper_AST_sha256": hashlib.sha256(ast.dump(function).encode()).hexdigest(),
        "provider_source": record(ROOT / "CrowdSim/humanoid/runtime/dynbench_scene.py"),
        "preflight": actual["preflight"], "checks": {
            "direct_callable_field_is_not_copied_by_post_init": direct_callable_owner_unchanged,
            "outer_spawn_object_is_deepcopied": config_and_spawn_are_copied,
            "bound_method_provider_owner_is_deepcopied": owner_is_copy,
            "copied_source_configuration_identical": copied_source_same,
            "copied_precreate_hook_flag_true": copied_precreate_flag_true,
            "callback_function_identity_unchanged": copied_method_same_function,
            "copied_spawn_flag_does_not_update_original": copied_flag_independent},
        "conclusion": "Original world provider _spawned cannot prove callback execution. Stage marker/floor/source conflict checks still guard duplicate scene injection; a live SDK run must verify stage identity and source ownership independently.",
        "limitations": ["Official configclass post-init body was executed in isolation, with string-resolution boundaries stubbed.",
            "No AssetBaseCfg/InteractiveScene instance or USD stage was created; static SDK source review supplies their call path.",
            "The copied flag write is an ownership demonstration, not a spawn or physics measurement."]}
    (OUTPUT / "config_copy_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}))


if __name__ == "__main__":
    main()
