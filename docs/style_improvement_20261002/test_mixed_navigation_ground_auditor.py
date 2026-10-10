"""CPU contracts for candidate admission and independent actual-plan replay."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

DOCS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mixed_navigation_auditor_under_test", DOCS/"audit_mixed_navigation_ground.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_parameters_reject_new_or_changed_navigation_even_equal_number_bool():
    baseline = {"config": "old", "resume": "old.pt", "output": "oldout", "seed": 7, "guidance_weight": 20.0}
    candidate = dict(baseline, config="new", resume="new.pt", output="newout", near_horizon_waypoint_mass=0.0)
    assert set(gate.argument_changes(candidate, baseline)) == {"config", "resume", "output"}
    for change in ({"seed": 8}, {"guidance_weight": 40.0}, {"seed": True}, {"unknown_flag": False}):
        with pytest.raises(ValueError, match="Unregistered"):
            gate.argument_changes(dict(candidate, **change), baseline)
    assert gate.strict_equal({"flag": False}, {"flag": 0}) is False


@pytest.mark.parametrize("defect", ["failed", "count", "gpu", "sdk", "checks", "sha", "budget", "duplicate"])
def test_actual_terminal_admission_rejects_invalid_selected_checkpoint(defect):
    admission = json.loads(gate.ADMISSION.read_text())
    assert gate.select_checkpoint(admission, 500)["physical_transitions"] == 4096000
    value = deepcopy(admission)
    if defect == "failed":
        value["status"] = "failed"
    elif defect == "count":
        value["check_count"] = 219
    elif defect == "gpu":
        value["GPU_used"] = True
    elif defect == "sdk":
        value["SDK_started"] = True
    elif defect == "checks":
        value["checks"][-1]["passed"] = False
    elif defect == "sha":
        value["checkpoints"][0]["checkpoint"]["sha256"] = "0"*64
    elif defect == "budget":
        value["checkpoints"][0]["group_statistics"]["procedural_actor_transitions"] -= 1
    elif defect == "duplicate":
        value["checkpoints"].append(deepcopy(value["checkpoints"][0]))
    with pytest.raises(ValueError):
        gate.select_checkpoint(value, 500)


@pytest.mark.parametrize("defect", ["traversal", "absolute", "backslash", "extra", "link", "duplicate"])
def test_source_archive_rejects_unsafe_or_unregistered_members(defect):
    record = {"path": "CrowdSim/humanoid/amp_world.py", "bytes": 3, "sha256": "unused"}
    member = tarfile.TarInfo(record["path"])
    member.size = 3
    assert gate.validate_member_paths([record], [member]) == {record["path"]: record}
    if defect in {"traversal", "absolute", "backslash"}:
        record["path"] = {"traversal": "../escape", "absolute": "/tmp/escape", "backslash": "a\\escape"}[defect]
        member.name = record["path"]
    elif defect == "extra":
        with pytest.raises(ValueError):
            gate.validate_member_paths([record], [member, tarfile.TarInfo("extra")])
        return
    elif defect == "link":
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp/elsewhere"
    elif defect == "duplicate":
        with pytest.raises(ValueError):
            gate.validate_member_paths([record, record], [member])
        return
    with pytest.raises(ValueError):
        gate.validate_member_paths([record], [member])


@pytest.fixture(scope="module")
def actual_baseline():
    folder = gate.ROOT/"output/diffusion_amp/shared_amass2848_backward_penalty_ground_support_200s_v1"
    if not (folder/"physical_states.npz").exists():
        pytest.skip("Requires explicitly recorded200s native baseline")
    result = json.loads((folder/"result.json").read_text())
    with np.load(folder/"physical_states.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    with np.load(folder/"reference_plans.npz", allow_pickle=False) as archive:
        plans = {k: archive[k] for k in archive.files}
    return arrays, plans, result


def test_real6000_frame_reference_phase_and_route_clock_replay(actual_baseline):
    arrays, plans, result = actual_baseline
    audit = gate.load_helper().Audit()
    proof = gate.verify_navigation_arrays(audit, arrays, plans, result)
    assert proof["plan_records"] == 1680
    assert proof["joint_forward_calls"] == 420
    assert proof["failure_events"] == []
    assert len(proof["reset_events"]) == 4
    assert proof["full_cycles"] == 4
    assert proof["all_failed_and_fallen_frames_retained"] is True
    assert all(c["passed"] is True for c in audit.checks)


@pytest.mark.parametrize("defect", ["missing_plan", "target_phase", "clock", "fake_arrival"])
def test_real_plan_capture_corruption_fails_closed(actual_baseline, defect):
    original, source_plans, source_result = actual_baseline
    arrays, plans, result = dict(original), dict(source_plans), deepcopy(source_result)
    if defect == "missing_plan":
        plans = {k: v[1:] for k, v in plans.items()}
    elif defect == "target_phase":
        arrays["target_xy"] = original["target_xy"].copy()
        arrays["target_xy"][100, 0, 0] += .1
    elif defect == "clock":
        result["planning_clock"]["forward_calls"] += 1
    elif defect == "fake_arrival":
        arrays["route_vertex_reached"] = original["route_vertex_reached"].copy()
        arrays["route_vertex_reached"][0, 0] = True
    with pytest.raises(ValueError):
        gate.verify_navigation_arrays(gate.load_helper().Audit(), arrays, plans, result)


def test_real_fallen_frames_and_reset_offgrid_plans_remain_in_temporal_proof():
    folder = gate.ROOT/"output/diffusion_amp/shared_amass_root_curriculum_ground_support_200s_v1"
    if not (folder/"physical_states.npz").exists():
        pytest.skip("Requires the explicit older13-fall200s native capture")
    result = json.loads((folder/"result.json").read_text())
    with np.load(folder/"physical_states.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    with np.load(folder/"reference_plans.npz", allow_pickle=False) as archive:
        plans = {k: archive[k] for k in archive.files}
    proof = gate.verify_navigation_arrays(gate.load_helper().Audit(), arrays, plans, result)
    assert len(proof["failure_events"]) == 13
    assert all(row["fall"] is True for row in proof["failure_events"])
    assert len(proof["reset_events"]) == 17
    assert sum(row["vertices"] for row in proof["per_episode_routes"]) == 5


def test_actual_capture_report_reproduction_uses_registered138_wrapper():
    from types import SimpleNamespace

    module = SimpleNamespace(markdown=lambda report: "Verified measurements\n\n## 可复现 CPU 命令\nold helper command")
    args = SimpleNamespace(candidate_dir="/tmp/source states", candidate_step=2000,
        registered_preparation=gate.DOCS/"mixed_curriculum_step2000_ground_support_200s_preparation_v1.json",
        source_manifest=gate.SOURCE, checkpoint_admission=gate.ADMISSION)
    text = gate.capture_markdown(module, {}, args)
    assert "Verified measurements" in text and "old helper command" not in text
    assert "audit_mixed_navigation_ground.py" in text
    assert "--registered-preparation" in text and "--checkpoint-admission" in text
    assert "'/tmp/source states'" in text
