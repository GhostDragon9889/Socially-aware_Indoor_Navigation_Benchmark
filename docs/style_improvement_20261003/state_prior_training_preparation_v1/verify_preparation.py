"""Verify the prepared paths/recipe/old source on CPU without building a bundle."""
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("state_prior_preparation_builder", HERE / "build_isolated_training.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = builder.baseline()
    builder.fresh_paths()
    for path in ("state_prior_tracking2000_v1", "state_prior_nav2000_v1"):
        assert not (builder.REPO / "output/diffusion_amp" / path).exists()
    declared = {row["path"] for row in baseline["files"]}
    actual = {str(path.relative_to(builder.OLD)) for path in (builder.OLD / "code").rglob("*")
              if path.is_file() and not path.is_symlink()}
    assert actual == declared
    for row in baseline["files"]:
        path = builder.OLD / row["path"]
        assert path.stat().st_size == row["bytes"] and builder.digest(path) == row["sha256"]
        assert not path.stat().st_mode & 0o222
    prepared = json.loads((HERE / "commands_prepared.json").read_text())
    assert prepared == builder.commands()
    original = json.loads((builder.OLD / "commands_v2.json").read_text())
    normalized = deepcopy(prepared["commands"])
    for name, argv in normalized.items():
        argv[argv.index("--config") + 1] = original["commands"][name][original["commands"][name].index("--config") + 1]
        argv[argv.index("--output") + 1] = original["commands"][name][original["commands"][name].index("--output") + 1]
        if "--resume" in argv:
            argv[argv.index("--resume") + 1] = original["commands"][name][original["commands"][name].index("--resume") + 1]
    assert normalized == original["commands"]
    assert "--resume" not in prepared["commands"]["fullfresh2000"]
    assert prepared["CPU_env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert prepared["python"] == builder.SDK_PYTHON
    assert yaml.safe_load((HERE / "profile_prepared.yaml").read_text()) == builder.profile()
    registry = json.loads((HERE / "registration.json").read_text())
    assert registry["source_build_contract"]["modified_paths"] == [name for name, kind in builder.EXPECTED_TARGETS.items() if kind == "modified"]
    assert registry["source_build_contract"]["added_candidate_paths"] == [name for name, kind in builder.EXPECTED_TARGETS.items() if kind == "added"]
    assert registry["GPU_pilot"]["fresh_updates"] == registry["GPU_pilot"]["resume_additional_updates"] == 4
    assert registry["fresh_full"]["transitions"] == 2000 * 256 * 32
    assert registry["future_effect"] is False and registry["final_bundle_created"] is False
    for name in ("build_isolated_training.py", "check_cold_bundle_cpu.py", "verify_preparation.py"):
        ast.parse((HERE / name).read_text())
    contract_path = HERE / "candidate_delta_contract.json"
    if contract_path.exists():
        builder.validate_contract(contract_path)
    assert not any(name == "torch" or name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules)
    builder.write_json(args.output, {"schema": "public_state_prior.docs_preparation_CPU.v1", "status": "passed",
        "CPU_only": True, "GPU_used": False, "Kit_started": False, "future_effect": False,
        "old931_full_file_hashes_verified": True, "old931_code_immutable_modes_verified": True,
        "old_launcher_and_commands_identity_verified": True, "new_paths_all_absent": True,
        "all_training_CLI_fields_preserved_except_profile_output_resume_paths": True,
        "profile_technical_change": "coupled D/history-RMS cold initialization only",
        "old_full_network_adapter_retained_by_byte_copy_plan": True,
        "candidate_contract_recorded": contract_path.exists(), "candidate_source_approved": False,
        "bundle_created": False, "future_checkpoint_payload_read": False, "sealed_test_outputs_read": False,
        "registration": builder.record(HERE / "registration.json"),
        "prepared_commands": builder.record(HERE / "commands_prepared.json"),
        "prepared_profile": builder.record(HERE / "profile_prepared.yaml"),
        "builder": builder.record(HERE / "build_isolated_training.py"),
        "cold_check": builder.record(HERE / "check_cold_bundle_cpu.py")})


if __name__ == "__main__":
    main()
