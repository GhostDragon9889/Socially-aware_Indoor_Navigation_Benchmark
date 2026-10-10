"""Necessary q source/producer ABI checks, without loading a future model or capture."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location("q_prepared_executor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    registration = json.loads((HERE / "registration.json").read_text())
    executor = load(HERE / "run_registered_evaluation.py")
    executor.validate_registered_dependencies(registration)
    checks = ["Actual final q-reader source_files/status ABI and independent peer exact"]
    specs = json.loads((HERE / "metadata_adaptation_spec.json").read_text())
    for name, spec in specs.items():
        restored = (HERE / name).read_text()
        if name == "run_registered_evaluation.py":
            restored = restored.replace(executor.REGISTRATION_SHA, "UNFROZEN_REGISTRATION_SHA256")
        for operation in reversed(spec["operations"]):
            restored = restored.replace(operation["after"], operation["before"])
        original = Path(spec["original_path"]).read_text()
        assert restored == original and hashlib.sha256(restored.encode()).hexdigest() == spec["original_sha256"]
        assert ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
        checks.append("Entire inverse metadata byte/AST exact: " + name)
    path = Path(registration["future_checkpoint"]["path"])
    for protocol in ("tracking", "navigation"):
        section = registration[protocol]
        entry, argv = executor.build_command(protocol, path, "1" * 64, section["parsed"]["output"], registration)
        expected = section["argv_without_future_SHA"] if protocol == "tracking" else section["argv"]
        assert argv == expected + (["--checkpoint-sha256", "1" * 64] if protocol == "tracking" else [])
        assert entry.is_relative_to(executor.CODE)
        checks.append("Original fixed protocol argv retained: " + protocol)
    try:
        executor.build_command("tracking", path.with_name("last.pt"), "1" * 64, registration["tracking"]["parsed"]["output"], registration)
    except ValueError:
        checks.append("Unregistered mutable checkpoint alias rejected without reading a checkpoint")
    else:
        raise AssertionError("Mutable alias accepted")
    pilot = json.loads((ROOT / "docs/style_improvement_20261003/state_prior_stratified_training_audit_v1/actual_pilot4_v1/audit.json").read_text())
    try:
        executor.validate_admission(pilot, pilot["checkpoints"][0]["checkpoint"], registration)
    except ValueError:
        checks.append("Real pilot4 cannot admit future terminal2000")
    else:
        raise AssertionError("Pilot admitted as terminal")
    # Synthetic metadata exercises the replacement ABI only, never a future training admission.
    synthetic = deepcopy(pilot)
    synthetic.update(actual_terminal_fresh2000_confirmed=True, phase="terminal2000", milestone_only_admission=False,
                     terminal_training_result={}, actual_closed_metrics={})
    row = synthetic["checkpoints"][0]
    row.update(updates=2000, transitions=16384000)
    row["terrain_policy"]["post_update_calls"] = 2000
    row["state_prior"]["current_AMP_count"] = 24776000.0
    row["RMS_counts"]["amp_normalizer"] = 24776000.0
    row["actual_expert_sampling_saved_state"].update(sampled_batches=2000, sampled_windows=8192000)
    executor.validate_admission(synthetic, row["checkpoint"], registration)
    checks.append("Synthetic q metadata ABI positive only; not an actual terminal claim")
    for key, value in (("registry_sha256", "0" * 64), ("private_CPU_RNG_saved_and_actual_loader_exact", False), ("sampled_windows", 16384)):
        wrong = deepcopy(synthetic)
        wrong["checkpoints"][0]["actual_expert_sampling_saved_state"][key] = value
        try:
            executor.validate_admission(wrong, row["checkpoint"], registration)
        except ValueError:
            checks.append("Synthetic wrong q ABI rejected: " + key)
        else:
            raise AssertionError("Wrong q ABI accepted: " + key)
    proof = {"schema": "stratified_prior.evaluation_metadata_only_CPU_proof.v1", "status": "passed", "checks": checks,
             "synthetic_terminal_only": True, "future_terminal_checkpoint_read": False, "future_actual_terminal_admitted": False,
             "GPU_used": False, "SDK_started": False, "future_effect": False, "sealed34_outputs_read": False}
    with (HERE / "metadata_CPU_proof.json").open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checks": len(checks), "future_effect": False}))


if __name__ == "__main__":
    main()
