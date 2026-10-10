"""One source-only peer; no executor, test, model, capture or SDK is run."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs/style_improvement_20261003"
HERE = Path(__file__).resolve().parent


def record(path):
    path = Path(path).absolute()
    assert path.is_file() and not path.is_symlink(), path
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def exact(row):
    assert record(row["path"]) == row, row


def read(path):
    return json.loads(Path(path).read_text())


def verify_freeze(directory, sha):
    path = directory / "source_freeze_v1.json"
    assert record(path)["sha256"] == sha
    assert not path.stat().st_mode & 0o222
    freeze = read(path)
    assert freeze["status"] == "frozen_prepared_only"
    rows = freeze["source_files"]
    names = {str(Path(row["path"]).relative_to(ROOT)): row for row in rows}
    assert len(names) == len(rows)
    for row in rows:
        exact(row)
        assert not Path(row["path"]).stat().st_mode & 0o222
    exact(freeze["archive"])
    with tarfile.open(freeze["archive"]["path"], "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(rows) == freeze["archive_regular_members"]
        assert {member.name for member in members} == set(names)
        for member in members:
            assert member.isfile() and not member.mode & 0o222
            raw = archive.extractfile(member).read()
            assert len(raw) == names[member.name]["bytes"]
            assert hashlib.sha256(raw).hexdigest() == names[member.name]["sha256"]
    return freeze


def inverse(raw, operations):
    for operation in reversed(operations):
        raw = raw.replace(operation["after"], operation["before"])
    return raw


def exact_original(restored, original):
    assert restored.encode() == Path(original).read_bytes()
    assert ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(Path(original).read_text()), include_attributes=False)


def main():
    evaluation = DOCS / "state_prior_stratified_evaluation_v1"
    metrics = DOCS / "state_prior_stratified_CPU_metrics_preparation_v1"
    ef = verify_freeze(evaluation, "f8be06861c782d1a298b0728cf80f120689bf1efcbdc1d66314f02e1eaf70ea6")
    mf = verify_freeze(metrics, "7c4ad7426ed9c9de20ca4b905abe9c3e972b690a52eab1e152873bcf62096635")
    registration = read(evaluation / "registration.json")
    spec = read(evaluation / "metadata_adaptation_spec.json")
    for name, row in spec.items():
        assert record(row["original_path"])["sha256"] == row["original_sha256"]
        raw = (evaluation / name).read_text()
        if name == "run_registered_evaluation.py":
            raw = raw.replace(record(evaluation / "registration.json")["sha256"], "UNFROZEN_REGISTRATION_SHA256")
        exact_original(inverse(raw, row["operations"]), row["original_path"])
    old = read(registration["sources"]["prior_registration"]["path"])
    for protocol in ("tracking", "navigation"):
        allowed = {"checkpoint", "checkpoint_sha256", "output"} if protocol == "tracking" else {"config", "resume", "output", "scene_config"}
        assert {k: v for k, v in registration[protocol]["parsed"].items() if k not in allowed} == {k: v for k, v in old[protocol]["parsed"].items() if k not in allowed}
    assert registration["acceptance"] == old["acceptance"]
    future = registration["future_checkpoint"]
    assert future["sha256"] is None and future["bytes"] is None
    assert future["actual_payload_preflight_performed"] is False and future["actual_model_effect_evaluated"] is False
    assert (future["required_updates"], future["required_transitions"]) == (2000, 16384000)
    sources = registration["sources"]
    for key in ("isolated_source_manifest", "registered_profile", "state_prior_auditor", "state_prior_auditor_source_freeze", "state_prior_auditor_peer"):
        exact(sources[key])
    assert sources["isolated_source_manifest"]["sha256"] == "2e14550a64a5c6d091d64312e2799eb3854e88e9dacdeffacd9990297ced8359"
    reader = read(sources["state_prior_auditor_source_freeze"]["path"])
    peer = read(sources["state_prior_auditor_peer"]["path"])
    assert reader["schema"] == "stratified_prior.independent_auditor_source_freeze.v1"
    assert reader["status"] == "prepared_only_frozen_pending_independent_peer"
    assert reader["source_manifest"] == sources["isolated_source_manifest"]
    assert sources["state_prior_auditor"] in reader["source_files"]
    assert peer["status"] == "passed_prepared_only" and peer["required_findings"] == []
    assert peer["reader_freeze"] == sources["state_prior_auditor_source_freeze"] and peer["main"] == sources["state_prior_auditor"]
    callback = next(row for row in reader["source_files"] if Path(row["path"]).name == "expert_sampling_checks.py")
    exact(callback)
    assert callback["sha256"] == "737e2dbba86221e5350f6b867c0a84c4089de18a96eb669009fbe0b1fa59bac6"
    pilot = read(DOCS / "state_prior_stratified_training_audit_v1/actual_pilot4_v1/audit.json")
    actual = pilot["checkpoints"][0]["actual_expert_sampling_saved_state"]
    assert future["required_expert_sampling_contract"] == {k: actual[k] for k in future["required_expert_sampling_contract"]}
    assert actual["sampled_windows"] == 16384 and actual["sampled_batches"] == 4
    expected_reward = {"amp_reward_scale": 2.0, "amp_reward_weight": 2.0, "task_reward_weight": 0.5, "direction_mode": "backwards_penalty", "amp_probability_epsilon": 0.0001}
    assert future["required_reward_contract"] == expected_reward
    profile = Path(sources["registered_profile"]["path"]).read_text()
    for k, v in expected_reward.items():
        assert f"  {k}: {v}" in profile
    assert "  amp_weight: 2.0" in profile
    auditor = Path(sources["state_prior_auditor"]["path"]).read_text()
    assert 'json.loads(json.dumps(cfg)) == expected_config' in auditor
    assert 'TrainingConfig(**profile["training"])' in auditor
    executor = (evaluation / "run_registered_evaluation.py").read_text()
    assert 'sampling["sampled_batches"] != 2000' in executor and 'sampling["sampled_windows"] != 8192000' in executor
    proof = read(evaluation / "metadata_CPU_proof.json")
    assert proof["status"] == "passed" and len(proof["checks"]) == 12 and proof["synthetic_terminal_only"] is True
    receipts = [read(evaluation / name) for name in ("draft_actual_CPU_receipt.json", "finalize_actual_CPU_receipt.json")]
    assert all(r["status"] == "closed" and r["exit_code"] == 0 and r["child_exit_observed"] is True and r["driver_completed"] is True for r in receipts)
    ms = read(metrics / "metadata_adaptation_spec.json")
    exact(ms["original_runner"])
    exact(ms["original_catalog"])
    exact_original(inverse((metrics / "run_CPU_metrics.py").read_text(), ms["operations"]), ms["original_runner"]["path"])
    catalog = read(metrics / "commands_CPU_metrics_v1.json")
    old_catalog = read(ms["original_catalog"]["path"])
    math_keys = ["style90_entry", "style90_math", "style90_reference", "pose45_basis", "pose45_reference_evaluation", "pose45_evaluator", "raw_ground_tool", "idle_tool", "tracking_decomposition_tool", "tracking_comparison_tool"]
    for key in math_keys:
        new, old_row = catalog["sources"][key], old_catalog["sources"][key]
        exact(new)
        assert {k: new[k] for k in ("bytes", "sha256")} == {k: old_row[k] for k in ("bytes", "sha256")}
    for key, row in catalog["sources"].items():
        exact(row)
    assert catalog["sources"]["evaluation_source_freeze"] == record(evaluation / "source_freeze_v1.json")
    assert catalog["sources"]["evaluation_executor"] == record(evaluation / "run_registered_evaluation.py")
    assert catalog["sources"]["evaluation_registration"] == record(evaluation / "registration.json")
    assert catalog["sources"]["evaluation_evidence"] == record(evaluation / "evaluation_evidence.py")
    assert set(catalog["metrics"]) == set(old_catalog["metrics"])
    for name, row in catalog["metrics"].items():
        old_row = old_catalog["metrics"][name]
        assert row["protocol"] == old_row["protocol"]
        old_argv = old_row["metric_argv"]
        remapped = [v.replace(old_row["run_name"] + "=" + old_row["input"], row["run_name"] + "=" + row["input"]).replace(old_row["output"], row["output"]) for v in old_argv]
        assert row["metric_argv"] == remapped
        assert row["input"] == registration[row["protocol"]]["parsed"]["output"]
        assert row["checkpoint_admission"] == future["terminal_admission_path"]
        assert row["command"][1] == str(metrics / "run_CPU_metrics.py") and row["command"][2:] == ["--metric", name]
    metric_receipt = read(metrics / "actual_CPU_preparation.json")
    assert metric_receipt["status"] == "closed" and metric_receipt["exit_code"] == 0 and metric_receipt["child_exit_observed"] is True and metric_receipt["driver_completed"] is True
    clarification = read(metrics / "sidecar_filename_clarification.json")
    assert clarification["actual_navigation_raw_artifacts"] == {"raw_report": "audit.json", "source_sidecar": "actual_source_admission.json"}
    metric_source = (metrics / "run_CPU_metrics.py").read_text()
    assert 'write(output / "actual_source_admission.json"' in metric_source
    common = {"status": "passed_prepared_only", "required_findings": [], "CPU_only": True, "GPU_used": False, "SDK_started": False, "models_loaded": False, "old_suites_rerun": False, "builder_12_checks_rerun": False, "future_checkpoint_read": False, "future_capture_read": False, "sealed34_outputs_read": False, "actual_future_terminal_admitted": False, "actual_quality_pass": False, "independent_source_checker": record(__file__)}
    evaluation_peer = {"schema": "stratified_prior.independent_prepared_evaluation_peer.v1", **common, "source_freeze": record(evaluation / "source_freeze_v1.json"), "registration": record(evaluation / "registration.json"), "source_files_verified": len(ef["source_files"]), "archive_members_verified": ef["archive_regular_members"], "inverse_whole_bytes_and_AST_exact_files": list(spec), "original_full_tracking22_and_navigation6000_protocols_retained": True, "original_acceptance_and_failure_inclusion_unchanged": True, "terminal_expert_sampling_windows": 8192000, "terminal_expert_sampling_batches": 2000, "reward2_enforcement": "Exact independently frozen actual auditor compares the complete saved TrainingConfig against the hash-bound weight2 profile. The executor uses that actual audited report and q saved-state fields; it does not claim a standalone reward_contract report field.", "producer_callback": callback, "reused_builder_CPU_proof": record(evaluation / "metadata_CPU_proof.json"), "reused_actual_preparation_receipts": [record(evaluation / name) for name in ("draft_actual_CPU_receipt.json", "finalize_actual_CPU_receipt.json")], "future_terminal_SHA_and_bytes_unknown": True, "no_MMD_quality_filter_added": True, "limits": ["Preparation and source identity are verified; actual terminal payload and motion quality remain pending.", "q tags are kinematic proxies, window ESS is reduced, and importance correction is absent; unchanged from the registered q experiment."]}
    metrics_peer = {"schema": "stratified_prior.independent_prepared_CPU_metrics_peer.v1", **common, "source_freeze": record(metrics / "source_freeze_v1.json"), "source_files_verified": len(mf["source_files"]), "archive_members_verified": mf["archive_regular_members"], "inverse_whole_runner_bytes_and_AST_exact": True, "all_five_metric_parameters_exact_except_registered_identity_paths": True, "all_ten_math_reference_basis_sources_bytes_exact": True, "math_reference_sources": {key: catalog["sources"][key] for key in math_keys}, "evaluation_source_contract_bound": True, "actual_source_admission_precedes_measurement": True, "navigation_raw_artifacts": clarification["actual_navigation_raw_artifacts"], "frozen_README_typo_clarification": record(metrics / "sidecar_filename_clarification.json"), "reused_actual_preparation_receipt": record(metrics / "actual_CPU_preparation.json"), "no_refit_resampling_failure_filter_or_metric_gate_added": True, "limits": ["No future captures or metrics were consumed. The frozen README filename typo is documented separately; the actual producer writes actual_source_admission.json."]}
    for name, report in (("evaluation_peer_review.json", evaluation_peer), ("CPU_metrics_peer_review.json", metrics_peer)):
        with (HERE / name).open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        (HERE / name).chmod(0o444)
    Path(__file__).chmod(0o444)
    print(json.dumps({"status": "passed_prepared_only", "evaluation_peer": record(HERE / "evaluation_peer_review.json"), "CPU_metrics_peer": record(HERE / "CPU_metrics_peer_review.json"), "no_models_tests_GPU_or_future_capture": True}))


if __name__ == "__main__":
    main()
