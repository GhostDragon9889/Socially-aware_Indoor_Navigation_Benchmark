"""One independent reader delta/source/boundary check; no future checkpoint."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
READER = ROOT / "docs/style_improvement_20261003/state_prior_stratified_training_audit_v1"
OLD = ROOT / "docs/style_improvement_20261003/state_prior_training_audit_v1"
PREP = ROOT / "docs/style_improvement_20261003/state_prior_stratified_training_preparation_v1"


def record(path):
    b = path.read_bytes()
    return {"path": str(path), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def main():
    freeze = json.loads((READER / "source_freeze_v1.json").read_text())
    assert record(READER / "source_freeze_v1.json")["sha256"] == "ef0a920159f98223a60c3dc6207548130445972757cd6dcacdbd33414cd82180"
    for row in freeze["source_files"]:
        p = Path(row["path"])
        assert record(p) == row and stat.S_IMODE(p.stat().st_mode) == 0o444 and not p.is_symlink()
    assert len(freeze["source_files"]) == 11
    assert record(Path(freeze["archive"]["path"])) == freeze["archive"]
    with tarfile.open(freeze["archive"]["path"], "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == 11 and {m.name for m in members} == {Path(r["path"]).name for r in freeze["source_files"]}
        for m in members:
            assert m.isfile() and archive.extractfile(m).read() == (READER / m.name).read_bytes()
    edits = json.loads((READER / "adapter_exact_replacements.json").read_text())
    original = (OLD / "audit_state_prior_training.py").read_text()
    restored = (READER / "audit_stratified_training.py").read_text()
    for before, after in reversed(edits["main_metadata_replacements"] + edits["main_explicit_q_callback_replacements"]):
        assert after in restored
        restored = restored.replace(after, before)
    assert restored == original and ast.dump(ast.parse(restored)) == ast.dump(ast.parse(original))
    helper = (READER / "state_prior_checks.py").read_text()
    assert helper.count(edits["helper_new_archive_derivation_tail"]) == 1
    helper = helper.replace(edits["helper_new_archive_derivation_tail"], edits["helper_old_archive_derivation_tail"])
    for before, after in reversed(edits["helper_metadata_replacements"]):
        assert after in helper
        helper = helper.replace(after, before)
    assert helper == (OLD / "state_prior_checks.py").read_text()
    spec = importlib.util.spec_from_file_location("independent_saved_q_contract", READER / "expert_sampling_checks.py")
    callback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(callback)
    q = yaml.safe_load((PREP / "profile_prepared.yaml").read_text())["training"]["expert_sampling"]
    bank = json.loads((PREP / "actual_CPU_readiness.json").read_text())["sampler"]["sampler"]
    identity = {"schema": "navisaaclab.expert_window_sampler_state.v2", "config": bank["config"],
                **{k: bank[k] for k in ("config_sha256", "registry_sha256", "probabilities_sha256")},
                "manifest_sha256": bank["source"]["manifest_sha256"], "generator_state": b"synthetic metadata only",
                "sampled_windows": 0, "sampled_batches": 0}
    saved = {"schema": "navisaaclab.expert_reference_sampler.v1", "configuration": q,
             "sampler": {**deepcopy(identity), "sampled_windows": 16384, "sampled_batches": 4}}
    state = {"update_count": 4, "config": {"expert_sampling": q}, "runner_state": {"expert_sampling": saved}}

    def check(name, condition):
        if not condition:
            raise ValueError(name)

    assert callback.validate_saved_sampling_contract(state, q, identity, check) == saved
    outcomes = []
    for name in ("baseline_null", "missing_state", "wrong_probability_hash", "wrong_sampled_windows"):
        bad = deepcopy(state)
        if name == "baseline_null":
            bad["config"]["expert_sampling"] = None
        elif name == "missing_state":
            del bad["runner_state"]["expert_sampling"]
        elif name == "wrong_probability_hash":
            bad["runner_state"]["expert_sampling"]["sampler"]["probabilities_sha256"] = "0" * 64
        else:
            bad["runner_state"]["expert_sampling"]["sampler"]["sampled_windows"] = 16383
        try:
            callback.validate_saved_sampling_contract(bad, q, identity, check)
        except ValueError:
            outcomes.append({"case": name, "rejected": True})
        else:
            raise AssertionError(name)
    assert not any(n.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for n in sys.modules)
    report = {"schema": "stratified_prior.independent_reader_delta_CPU_check.v1", "status": "passed_prepared_source_and_synthetic_boundary_only",
              "reader_freeze": record(READER / "source_freeze_v1.json"), "eleven_sources_readonly_hash_and_archive_payload_exact": True,
              "inverse_metadata_seven_hooks_restore_original_full_main_bytes_AST": True,
              "inverse_metadata_source_derivation_restore_original_helper_full_bytes": True,
              "actual_synthetic_callback_positive_passed": True, "actual_synthetic_callback_negatives": outcomes,
              "future_payload_read": False, "real_RNG_tensor_tested": False, "SDK_started": False, "GPU_used": False}
    with (HERE / "actual_peer_checks.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "frozen_sources": 11, "synthetic_negative_rejections": len(outcomes), "future_payload_read": False}))


if __name__ == "__main__":
    main()
