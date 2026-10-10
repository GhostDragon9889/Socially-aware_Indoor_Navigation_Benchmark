"""Synthetic CPU checkpoint lineage contracts; no real actor or SDK invented."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).with_name("audit_amass_root_tracking_lineage.py")
spec = importlib.util.spec_from_file_location("root_tracking_lineage_test", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def payload(step):
    return {"format": "navisaaclab_method_reconstruction_training", "physical_runner_state_included": True,
            "update_count": step, "physical_transition_count": step * 4 * 32,
            "runner_state": {"control_step": step * 32}, "config": {"expert_sampling": None},
            "model": {}, "optimizer": {}, "discriminator_optimizer": {}, "normalizers": {},
            "rng": dict.fromkeys(("python", "numpy", "torch_cpu", "torch_cuda")), "policy_replay": None}


def training(directory, *, step=1000, resume=None):
    return {"arguments": {"num_envs": 4, "rollout_steps": 32, "output": str(directory),
                          "updates": 250 if resume else 1000, "resume": str(resume) if resume else None},
            "training_config": {"expert_sampling": None}, "update_count": step,
            "actual_control_steps": (250 if resume else 1000) * 32}


def seal(directory, parent, root):
    item = audit.record(parent)
    item.update(update_count=500, physical_transition_count=500 * 4 * 32, control_step=500 * 32,
                complete_resume_payload_checked=True, active_plans_match_actual_bank=True,
                physical_bitwise_equivalence_claim=False)
    path = directory / "snapshots.json"
    path.write_text(json.dumps({"status": "passed", "snapshots": [item], "training_output_read_only": str(root)}))
    return path


def test_saved_budget_and_clock_detect_counter_tampering():
    cfg = training(Path("synthetic_only"))
    assert audit.check_saved_payload(payload(500), cfg, 500)["physical_transition_count"] == 64000
    bad = payload(500)
    bad["update_count"] = 1000
    with pytest.raises(ValueError, match="update counter"):
        audit.check_saved_payload(bad, cfg, 500)
    bad = payload(500)
    bad["runner_state"]["control_step"] += 1
    with pytest.raises(ValueError, match="control clock"):
        audit.check_saved_payload(bad, cfg, 500)


def test_resume_initialization_is_explicit_preserved_provenance_not_assumed():
    value = {"mode": "synthetic_public_transfer_contract", "sha256": "synthetic_not_a_real_weight"}
    assert audit.initialization_record({"initialization": value})[0] == value
    actual, source = audit.initialization_record({"provenance": {"initialization": value}})
    assert actual == value and "preserved on resume" in source
    with pytest.raises(ValueError, match="lacks"):
        audit.initialization_record({"provenance": {}})


def test_mask_tuple_vs_json_array_is_accepted_but_value_change_is_rejected():
    cfg = training(Path("synthetic_only"))
    cfg["training_config"]["action_mask"] = [1, 0, 1]
    saved = payload(500)
    saved["config"]["action_mask"] = (1, 0, 1)
    assert audit.check_saved_payload(saved, cfg, 500)["saved_config_matches_training"]
    cfg["training_config"]["action_mask"][1] = 1
    with pytest.raises(ValueError, match="training config differs"):
        audit.check_saved_payload(saved, cfg, 500)


def test_immutable500_half_budget_requires_matching_independent_seal(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    parent = tmp_path / "step_500.pt"
    torch.save(payload(500), parent)
    ledger = seal(tmp_path, parent, tmp_path / "original")
    result = audit.verify_lineage(audit.record(parent), training(tmp_path / "original"), 500, "original_snapshot", ledger)
    assert result["selected_budget_updates"] == 500
    assert not result["equal_budget_to_baseline1000"]
    bad = json.loads(ledger.read_text())
    bad["snapshots"][0]["sha256"] = "wrong"
    ledger.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="sealed ledger"):
        audit.verify_lineage(audit.record(parent), training(tmp_path / "original"), 500, "original_snapshot", ledger)


def test_750_is_actual500_plus250_new_stage_not_original_750(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    parent = tmp_path / "step_500.pt"
    torch.save(payload(500), parent)
    ledger = seal(tmp_path, parent, tmp_path / "original")
    stage = tmp_path / "new_stage"
    stage.mkdir()
    terminal = stage / "last.pt"
    torch.save(payload(750), terminal)
    metrics = [{"update_count": i, "physical_transitions": i * 4 * 32} for i in range(501, 751)]
    (stage / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    result = audit.verify_lineage(audit.record(terminal), training(stage, step=750, resume=parent), 750, "continued_from500", ledger)
    assert result["additional_updates"] == 250 and result["selected_budget_updates"] == 750
    assert result["additional_physical_transitions"] == 32000
    assert not result["is_original_trajectory750"] and not result["physical_resume_bitwise_guarantee"]
    metrics[0]["update_count"] = 500
    (stage / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    with pytest.raises(ValueError, match="contiguous"):
        audit.verify_lineage(audit.record(terminal), training(stage, step=750, resume=parent), 750, "continued_from500", ledger)
