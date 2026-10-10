"""Single prepared source/config/callback boundary, never reads future payloads."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "docs/style_improvement_20261003/state_prior_training_audit_v1"
PREP = ROOT / "docs/style_improvement_20261003/state_prior_stratified_training_preparation_v1"
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1"


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    edits = json.loads((HERE / "adapter_exact_replacements.json").read_text())
    changed_main = (HERE / "audit_stratified_training.py").read_text()
    normalized = changed_main
    for before, after in reversed(edits["main_metadata_replacements"] + edits["main_explicit_q_callback_replacements"]):
        assert after in normalized
        normalized = normalized.replace(after, before)
    original_main = (OLD / "audit_state_prior_training.py").read_text()
    assert normalized == original_main
    helper = (HERE / "state_prior_checks.py").read_text()
    normalized_helper = helper.replace(edits["helper_new_archive_derivation_tail"], edits["helper_old_archive_derivation_tail"])
    for before, after in reversed(edits["helper_metadata_replacements"]):
        assert after in normalized_helper
        normalized_helper = normalized_helper.replace(after, before)
    assert normalized_helper == (OLD / "state_prior_checks.py").read_text()
    assert ast.dump(ast.parse(normalized), include_attributes=False) == ast.dump(ast.parse(original_main), include_attributes=False)
    sys.path[:0] = [str(HERE), str(BUNDLE / "code")]
    from expert_sampling_checks import validate_saved_sampling_contract
    spec = importlib.util.spec_from_file_location("prepared_stratified_auditor", HERE / "audit_stratified_training.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    checks = []

    def check(name, condition, **detail):
        checks.append({"name": name, "passed": bool(condition), **detail})
        if not condition:
            raise ValueError(name)

    source = json.loads((BUNDLE / "source_manifest.json").read_text())
    check("source935_exact_known_manifest", record(BUNDLE / "source_manifest.json")["sha256"] == auditor.MANIFEST_SHA
          == "2e14550a64a5c6d091d64312e2799eb3854e88e9dacdeffacd9990297ced8359")
    names = {str(path.relative_to(BUNDLE)) for path in (BUNDLE / "code").rglob("*")
             if path.is_file() and not path.is_symlink()}
    check("source935_closed_copied_set", len(source["files"]) == 935
          and names == {row["path"] for row in source["files"]})
    archive = auditor.verify_source_archive(BUNDLE, source, record, check)
    before_profile = yaml.safe_load((ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_v1/code/configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml").read_text())
    profile = yaml.safe_load((BUNDLE / "code/configs/diffusion_amp/isolated_terrain_state_prior_stratified_absolute.yaml").read_text())
    q = profile["training"]["expert_sampling"]
    check("only_q_mapping_changed_from_baseline_null", before_profile["training"]["expert_sampling"] is None
          and q is not None and {k: v for k, v in before_profile["training"].items() if k != "expert_sampling"}
          == {k: v for k, v in profile["training"].items() if k != "expert_sampling"})
    bank = json.loads((PREP / "actual_CPU_readiness.json").read_text())["sampler"]["sampler"]
    check("bound_existing_actual_CPU_registry_proof", record(PREP / "actual_CPU_readiness.json")["sha256"]
          == "ff1e18caf999128a252b6af928c857f037235036718c67d3e3e055fda98a2c21")
    source_ast = ast.parse((BUNDLE / "code/CrowdSim/humanoid/training/expert_sampling.py").read_text())
    saved_schema = next(ast.literal_eval(node.value) for node in source_ast.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "_STATE_SCHEMA" for target in node.targets))
    identity = {"schema": saved_schema, "config": bank["config"],
                **{key: bank[key] for key in ("config_sha256", "registry_sha256", "probabilities_sha256")},
                "manifest_sha256": bank["source"]["manifest_sha256"], "generator_state": b"synthetic metadata only",
                "sampled_windows": 0, "sampled_batches": 0}
    saved = {"schema": "navisaaclab.expert_reference_sampler.v1", "configuration": q,
             "sampler": {**deepcopy(identity), "sampled_windows": 4 * 4096, "sampled_batches": 4}}
    state = {"update_count": 4, "config": {"expert_sampling": q}, "runner_state": {"expert_sampling": saved}}
    validate_saved_sampling_contract(state, q, identity, check, "synthetic_contract:")
    negative = []
    for label in ("baseline_null_config", "missing_saved_state", "wrong_registry", "wrong_counters"):
        mutated = deepcopy(state)
        if label == "baseline_null_config":
            mutated["config"]["expert_sampling"] = None
        elif label == "missing_saved_state":
            del mutated["runner_state"]["expert_sampling"]
        elif label == "wrong_registry":
            mutated["runner_state"]["expert_sampling"]["sampler"]["registry_sha256"] = "0" * 64
        else:
            mutated["runner_state"]["expert_sampling"]["sampler"]["sampled_windows"] -= 1

        def fail_closed(name, condition, **detail):
            if not condition:
                raise ValueError(name)

        try:
            validate_saved_sampling_contract(mutated, q, identity, fail_closed)
        except ValueError as error:
            negative.append({"case": label, "rejected": True, "error": str(error)})
        else:
            raise AssertionError(label)
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr"} for name in sys.modules)
    report = {"schema": "stratified_prior.prepared_auditor_CPU_boundary.v1", "status": "passed",
              "checks": checks, "source_archive": archive, "negative_saved_boundary_cases": negative,
              "original_full_main_bytes_and_AST_exact_after_only_declared_metadata_q_normalization": True,
              "original_full_helper_bytes_exact_after_only_declared_source_metadata_normalization": True,
              "original_main": record(OLD / "audit_state_prior_training.py"), "original_helper": record(OLD / "state_prior_checks.py"),
              "new_main": record(HERE / "audit_stratified_training.py"), "new_helper": record(HERE / "state_prior_checks.py"),
              "new_sampler_callback": record(HERE / "expert_sampling_checks.py"), "executed_reader": record(Path(__file__)),
              "prepared_only": True, "future_checkpoint_read": False, "actual_training_payload_admitted": False,
              "GPU_used": False, "SDK_started": False,
              "limit": "Boundary cases use explicit synthetic metadata and no RNG tensor; real bank/private-RNG restoration is checked by the actual checkpoint callback after root pilot closure."}
    with (HERE / "prepared_boundary_CPU.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "passed", "source_checks": len(checks), "saved_contract_negative_cases": len(negative),
                      "full_original_math_normalization_exact": True, "future_payload_read": False}))


if __name__ == "__main__":
    main()
