"""Reuse the complete q936 payload reader with registered937 post-step metadata.

No process is launched here. A real readonly root-owner binding is required;
the pending template deliberately cannot admit a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / "state_only_matched_q_training_audit_v1"
PREP = HERE.parent / "state_only_q_poststep_training_preparation_v1"
PLAN_SHA = "acd1027740bf618400a7b4b3b9009ef37f55aafc2c7ebabad3ef2f7f6dc55a90"
MANIFEST_SHA = "e57a7a74e34d3cdf8672fd8f4fcf6fa10cb36f8c64971c19d9ae743f2f1f509e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def exact_readonly(row):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"},
            "Actual source/artifact record is required; pending null is not evidence")
    path = Path(row["path"])
    require(path.is_absolute() and not path.is_symlink() and path.is_file()
            and not path.stat().st_mode & 0o222 and record(path) == row,
            "Exact readonly registered source/artifact bytes required")
    return path


def replace_exact(source, before, after):
    require(before in source and before != after, "Declared source replacement is missing")
    return source.replace(before, after)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_reader(binding_path):
    require("torch" not in sys.modules and "state_prior_checks" not in sys.modules
            and "expert_sampling_checks" not in sys.modules, "Fresh CPU reader process required")
    binding_record = record(binding_path)
    exact_readonly(binding_record)
    binding = json.loads(Path(binding_path).read_text())
    require(binding["schema"] == "state_only_q.poststep_reader_root_owner_binding.v1"
            and binding["status"] == "registered_source_only_no_future_payload_claim",
            "Root must register real owner sources before reader execution")
    require(binding["root_plan"]["sha256"] == PLAN_SHA
            and binding["source_manifest"]["sha256"] == MANIFEST_SHA,
            "Registered937 source and exact GPU plan are required")
    for key in ("root_plan", "source_manifest", "source_archive", "background_helper"):
        exact_readonly(binding[key])
    declared_plan = json.loads(Path(binding["root_plan"]["path"]).read_text())
    require(binding["actual_training_env"] == {**declared_plan["env"], "CUDA_VISIBLE_DEVICES": "0"},
            "Exact registered plan environment plus CUDA device0 is required")
    for row in binding["old_numeric_sources"].values():
        exact_readonly(row)
    require(binding["owner_schema"] == "state_only_q.poststep_actual_training_root_owner.v1",
            "Genuine root-tool training receipt schema required")
    for name in ("pilot_seal_schema", "audit_schema"):
        require(type(binding[name]) is str and bool(binding[name]), "Actual source schema required")
    background = load_module(binding["background_helper"]["path"], "poststep_registered_background")
    helper_source = (OLD / "state_prior_checks.py").read_text()
    main_source = (OLD / "audit_state_only_training.py").read_text()
    require(record(OLD / "state_prior_checks.py") == binding["old_numeric_sources"]["state_prior_checks.py"]
            and record(OLD / "audit_state_only_training.py") == binding["old_numeric_sources"]["audit_state_only_training.py"]
            and record(OLD / "expert_sampling_checks.py") == binding["old_numeric_sources"]["expert_sampling_checks.py"],
            "Original complete numerical reader sources must match")
    helper_replacements = [
        ("pre-step h[t..]; state-only discriminator ignores applied a[t]", "post-step h[t+1..] before resets"),
        ("b5422877f513952ea3d98e613a82cd6e3dae8074a2e01ade35ce6c1be19d770b", PLAN_SHA),
        ("a05a328b3e49212277821d7cf002b037831e3ff8cd327b4d2b81ccdbde78c305", "5f2c5360ff9d84d0c2e1e65a0acf86fdf627012c8e38eb3dd11792c6e341f2b1"),
        ("state_only_matched_q_training_preparation_v1", "state_only_q_poststep_training_preparation_v1"),
        ("actual936_full_source_archive", "actual937_full_source_archive"),
        ("source_archive940_", "source_archive941_"),
        ("len(members) == len(names) == 940", "len(members) == len(names) == 941"),
        ("previous_q935_source_manifest", "previous_q936_source_manifest"),
        ("2e14550a64a5c6d091d64312e2799eb3854e88e9dacdeffacd9990297ced8359", "5c2a22098bf8df60530b5f53963c041a221cc17e62759284d7586f6ea6a05f50"),
        ("state_only_matched_q_preparation_v1/source_delta.json", "state_only_q_poststep_timing_preparation_v1/source_derivation.json"),
        ("b6fce5510148db57c23003c8aa79eeeea886303decd4c3706b473d3068e5581b", "5a735f9222af692a2745ed8a8e51fe8baa49982c9da037db7916b5a54bb4bb81"),
        ("reverse_entire_runner_bytes_exact", "inverse_fullbytes_equal"),
        ("reverse_entire_runner_AST_exact", "inverse_AST_equal"),
        ("exact_q935_parent_one_runner_delta_one_state_recipe", "exact_q936_parent_one_runner_delta_one_poststep_recipe"),
        ("len(old) == 935 and len(declared) == 936", "len(old) == 936 and len(declared) == 937"),
        ("isolated_state_only_matched_q_absolute.yaml", "isolated_state_only_q_poststep_absolute.yaml"),
        ('"regular_members": 940, "code_files": 936', '"regular_members": 941, "code_files": 937'),
        ('"unchanged_parent_files": 934', '"unchanged_parent_files": 935'),
    ]
    for before, after in helper_replacements:
        helper_source = replace_exact(helper_source, before, after)
    helper = types.ModuleType("state_prior_checks")
    helper.__file__ = str(OLD / "state_prior_checks.py")
    exec(compile(helper_source, __file__, "exec"), helper.__dict__)

    def verify_launch(bundle, phase, source_sha, ledger_path, source_record, check, *, closed):
        trial = {"pilot4": "pilot4", "milestone": "fullfresh2000", "terminal2000": "fullfresh2000"}[phase]
        owner_row = binding["actual_launch_records"][trial]
        path = exact_readonly(owner_row)
        launch = json.loads(Path(path).read_text())
        plan = json.loads(Path(binding["root_plan"]["path"]).read_text())
        check("actual_root_launch_plan_SHA_and_command_exact",
              launch["schema"] == binding["owner_schema"] and launch["name"] == trial
              and launch["launch_plan_sha256"] == PLAN_SHA and launch["command"] == plan["commands"][trial]
              and launch["cwd"] == plan["cwd"] and launch["env"] == binding["actual_training_env"]
              and launch["source_manifest"] == binding["source_manifest"]
              and launch["source_archive"] == binding["source_archive"]
              and launch["command"][launch["command"].index("--import-report") + 1] == str(ledger_path)
              and source_sha == MANIFEST_SHA and source_record(bundle / "source_manifest.json")["sha256"] == MANIFEST_SHA)
        check("actual_root_process_closed_or_live_milestone_truthful",
              type(launch["actual_tool_session_id"]) is int and launch["actual_tool_session_id"] > 0
              and type(launch["launch_chunk"]) is str and bool(launch["launch_chunk"])
              and ((launch["status"] == "closed" and type(launch["actual_exit_code"]) is int
                    and launch["actual_exit_code"] == 0 and launch["child_exit_observed"] is True
                    and type(launch["completion_chunk"]) is str and bool(launch["completion_chunk"]))
                   if closed else ((launch["status"] == "running" and launch["actual_exit_code"] is None
                                    and launch["child_exit_observed"] is False and launch["completion_chunk"] is None)
                                   or (launch["status"] == "closed" and type(launch["actual_exit_code"]) is int
                                       and launch["actual_exit_code"] == 0 and launch["child_exit_observed"] is True
                                       and type(launch["completion_chunk"]) is str and bool(launch["completion_chunk"])))))
        observation = launch["import_ledger_observation"]
        check("actual_root_import_ledger_observation_registered937",
              observation["path"] == str(ledger_path)
              and observation["source_manifest_sha256"] == MANIFEST_SHA
              and (observation["runtime_finalization_observed"] is True if closed else True))
        background.validate_observation(launch["background_GPU_evidence"])
        check("actual_root_registered_background_source_and_truthful_training_exclusivity",
              launch["background_GPU_evidence_source"] == binding["background_helper"]
              and launch["no_parallel_training_GPU_at_launch"] is True
              and launch["no_parallel_GPU_at_launch"]
              is launch["background_GPU_evidence"]["no_parallel_GPU_at_launch"])
        return launch, source_record(path)

    helper._verify_launch = verify_launch
    sys.modules["state_prior_checks"] = helper
    sys.modules["expert_sampling_checks"] = load_module(OLD / "expert_sampling_checks.py", "expert_sampling_checks")
    main_replacements = [
        ("mixed_state_terrain_state_prior_stratified_matched_v1", "mixed_state_terrain_state_prior_stratified_poststep_v1"),
        ("amp_style_state_only_matched_q_isolated", "amp_style_state_only_q_poststep_isolated"),
        ("5c2a22098bf8df60530b5f53963c041a221cc17e62759284d7586f6ea6a05f50", MANIFEST_SHA),
        ("state_only_matched_q.actual_training_checkpoint_audit.v1", binding["audit_schema"]),
        ("state_only_matched_q.actual_pilot_seal.v1", binding["pilot_seal_schema"]),
        ('seal["actual_launch"]["exit_code"]', 'seal["actual_launch"]["actual_exit_code"]'),
        ("exact936_copied_source_set", "exact937_copied_source_set"),
        ("len(declared) == 936", "len(declared) == 937"),
        ("isolated_state_only_matched_q_absolute.yaml", "isolated_state_only_q_poststep_absolute.yaml"),
        ("copied_files_hashed=936", "copied_files_hashed=937"),
        ("All936 copied files hashed", "All937 copied files hashed"),
    ]
    for before, after in main_replacements:
        main_source = replace_exact(main_source, before, after)
    core = types.ModuleType("poststep_complete_payload_reader")
    core.__file__ = __file__
    exec(compile(main_source, __file__, "exec"), core.__dict__)
    original_write = core.write

    def write_with_identity(path, document):
        if Path(path).name == "audit.json":
            document.update(actual_execution_adapter=record(__file__),
                            root_owner_source_binding=binding_record,
                            original_numeric_sources=binding["old_numeric_sources"],
                            registered_background_helper=binding["background_helper"],
                            original_full_payload_numeric_checks_unchanged=True)
        return original_write(path, document)

    core.write = write_with_identity
    return core


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--launch-binding", type=Path, required=True)
    own, remaining = parser.parse_known_args()
    require("--phase" not in remaining or remaining.index("--phase") + 1 < len(remaining), "Phase value required")
    require("--phase" in remaining and remaining[remaining.index("--phase") + 1]
            in {"pilot4", "milestone", "terminal2000"}, "Fresh poststep phases only; no old pre-step resume")
    sys.argv = [sys.argv[0], *remaining]
    return prepare_reader(own.launch_binding).run()


if __name__ == "__main__":
    main()
