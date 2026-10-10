"""Exercise the actual root launcher with mock child processes; never start GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "state_prior_pacer_reward_actual_runs_v1/launch_registered_run.py"


def identity(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("mocked_actual_launch_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "HERE", tmp_path)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text('{}\n')
    source = identity(manifest)
    plan = {"schema": "public_state_prior.pacer_reward_root_launch_commands.v1",
            "source_manifest": source, "single_Kit_GPU_at_a_time": True,
            "cwd": str(tmp_path), "env": {}, "CPU_watch_env": {"CUDA_VISIBLE_DEVICES": ""},
            "watch_command": ["mock_cpu_watcher"],
            "commands": {"pilot4": ["mock_training", "--output", str(tmp_path / "pilot")],
                         "fullfresh2000": ["mock_training", "--output", str(tmp_path / "full")]}}
    gate = {"schema": "pacer_reward.actual_preparation_admission.v1", "status": "passed",
            "source_manifest": source, "CPU_only": True, "GPU_used": False,
            "SDK_started": False, "sealed34_outputs_read": False, "reward_contract_validated": True}
    state = {"popen": [], "kill": [], "killpg": []}
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "")
    monkeypatch.setattr(module.os, "kill", lambda *a: state["kill"].append(a))
    monkeypatch.setattr(module.os, "killpg", lambda *a: state["killpg"].append(a))

    def invoke(run="pilot4", child=None):
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan))
        gate_path = tmp_path / "gate.json"
        gate_path.write_text(json.dumps(gate))
        argv = [str(DRIVER), "--run", run, "--plan", str(plan_path),
                "--plan-sha256", identity(plan_path)["sha256"]]
        if run != "watch":
            argv += ["--admission", str(gate_path), "--admission-sha256", identity(gate_path)["sha256"]]
        monkeypatch.setattr(sys, "argv", argv)

        def popen(*args, **kwargs):
            state["popen"].append((args, kwargs))
            if child is None:
                raise AssertionError("Rejected run must not spawn a process")
            return child
        monkeypatch.setattr(module.subprocess, "Popen", popen)
        return module.main()
    return module, plan, gate, state, invoke


class Child:
    pid = 12345

    def __init__(self, results=(0,)):
        self.results = list(results)
        self.returncode = None
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        self.returncode = result
        return result


def full_gate(source):
    return {"schema": "pacer_reward.actual_training_checkpoint_audit.v1", "status": "passed",
            "source_manifest": source, "CPU_only": True, "GPU_used": False,
            "SDK_started": False, "sealed34_outputs_read": False,
            "phase": "pilot4", "copied_files_hashed": 935,
            "checks": [{"passed": True}], "check_count": 1,
            "checkpoints": [{"updates": 4, "transitions": 32768,
                             "full_payload_resume_validated": True}],
            "state_prior_initialization_validated": True, "public_D_or_AMP_RMS_transferred": True,
            "reward_contract": {"amp_reward_scale": 2.0, "amp_reward_weight": 0.5,
                                "task_reward_weight": 0.5, "direction_mode": "backwards_penalty",
                                "amp_probability_epsilon": 0.0001}}


def test_full_positive_actual_contract_accepts(setup):
    module, plan, _, _, _ = setup
    module.validate_gate(full_gate(plan["source_manifest"]), "fullfresh2000", plan["source_manifest"])


def test_full_actual_contract_rejects_old_reward_weight(setup):
    module, plan, _, _, _ = setup
    gate = full_gate(plan["source_manifest"])
    gate["reward_contract"]["amp_reward_weight"] = 2.0
    with pytest.raises(ValueError, match="PACER reward contract differs"):
        module.validate_gate(gate, "fullfresh2000", plan["source_manifest"])


@pytest.mark.parametrize("field,value", [
    ("checks", []), ("check_count", True), ("check_count", 1.0),
    ("checks", [{"passed": 1}]), ("copied_files_hashed", 935.0),
    ("state_prior_initialization_validated", 1),
])
def test_full_gate_rejects_false_or_ambiguous_admission(setup, field, value):
    module, plan, _, _, _ = setup
    gate = full_gate(plan["source_manifest"])
    gate[field] = value
    with pytest.raises(ValueError):
        module.validate_gate(gate, "fullfresh2000", plan["source_manifest"])


@pytest.mark.parametrize("token", ["--resume", "--resume=old.pt"])
def test_actual_fresh_entry_rejects_both_argparse_resume_forms(setup, token):
    _, plan, _, state, invoke = setup
    plan["commands"]["pilot4"].append(token)
    with pytest.raises(ValueError, match="fresh training"):
        invoke()
    assert state["popen"] == []


def test_actual_normal_exit_record_observes_child_and_session(setup):
    module, _, _, state, invoke = setup
    child = Child()
    with pytest.raises(SystemExit) as error:
        invoke(child=child)
    assert error.value.code == 0
    record = json.loads((module.HERE / "pilot4_actual_launch.json").read_text())
    assert record["status"] == "closed" and record["exit_code"] == 0
    assert record["child_exit_observed"] is True and record["driver_completed"] is True
    assert state["popen"][0][1]["start_new_session"] is True


def test_actual_interrupt_waits_after_process_lookup_race(setup, monkeypatch):
    module, _, _, _, invoke = setup
    child = Child((InterruptedError("parent interrupt"), -15))
    monkeypatch.setattr(module.os, "killpg", lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    with pytest.raises(SystemExit) as error:
        invoke(child=child)
    assert error.value.code == 1
    record = json.loads((module.HERE / "pilot4_actual_launch.json").read_text())
    assert record["exit_code"] == -15 and record["child_exit_observed"] is True
    assert record["driver_completed"] is False and child.wait_calls == [None, 10]


def test_actual_interrupt_kills_and_waits_when_term_times_out(setup):
    module, _, _, state, invoke = setup
    child = Child((InterruptedError("interrupt"), subprocess.TimeoutExpired("mock", 10), -9))
    with pytest.raises(SystemExit) as error:
        invoke(child=child)
    assert error.value.code == 1
    record = json.loads((module.HERE / "pilot4_actual_launch.json").read_text())
    assert record["status"] == "closed" and record["exit_code"] == -9
    assert record["child_exit_observed"] is True
    assert state["killpg"] == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]


def test_failed_cleanup_preserves_unobserved_exit_without_closed_zero(setup):
    module, _, _, _, invoke = setup
    child = Child((InterruptedError("interrupt"), subprocess.TimeoutExpired("mock", 10),
                   subprocess.TimeoutExpired("mock", 10)))
    with pytest.raises(SystemExit) as error:
        invoke(child=child)
    assert error.value.code == 1
    record = json.loads((module.HERE / "pilot4_actual_launch.json").read_text())
    assert record["status"] == "child_exit_not_observed" and record["exit_code"] is None
    assert record["child_exit_observed"] is False and record["driver_completed"] is False
    assert record["cleanup_error"]["type"] == "TimeoutExpired"


def test_actual_spawn_failure_is_recorded_as_start_failed(setup, monkeypatch):
    module, _, _, _, invoke = setup
    # invoke() mock returns a child; simulate Popen's own failure at its call.
    original_main = module.main
    def main_with_failing_popen():
        monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(OSError("spawn failed")))
        return original_main()
    monkeypatch.setattr(module, "main", main_with_failing_popen)
    with pytest.raises(SystemExit) as error:
        invoke(child=Child())
    assert error.value.code == 1
    record = json.loads((module.HERE / "pilot4_actual_launch.json").read_text())
    assert record["status"] == "start_failed" and record["exit_code"] == 1
    assert record["child_exit_observed"] is False and "child_pid" not in record


@pytest.mark.parametrize("tamper", [False, True])
def test_full_actual_entry_requires_unchanged_readonly_pilot(setup, tamper):
    module, plan, gate, state, invoke = setup
    pilot_output = Path(plan["commands"]["pilot4"][-1])
    pilot_output.mkdir()
    checkpoint = pilot_output / "step_4.pt"
    checkpoint.write_bytes(b"mock checkpoint: no tensor payload fabricated")
    admitted = identity(checkpoint)
    checkpoint.chmod(0o444)
    gate.clear()
    gate.update(full_gate(plan["source_manifest"]))
    gate["checkpoints"][0]["checkpoint"] = admitted
    if tamper:
        checkpoint.chmod(0o644)
        checkpoint.write_bytes(b"changed checkpoint")
        checkpoint.chmod(0o444)
        with pytest.raises(ValueError, match="Admitted pilot checkpoint changed"):
            invoke(run="fullfresh2000")
        assert state["popen"] == []
    else:
        with pytest.raises(SystemExit) as error:
            invoke(run="fullfresh2000", child=Child())
        assert error.value.code == 0
        record = json.loads((module.HERE / "fullfresh2000_actual_launch.json").read_text())
        assert record["child_exit_observed"] is True and record["exit_code"] == 0


def test_actual_entry_rejects_changed_source_manifest(setup):
    _, plan, _, state, invoke = setup
    Path(plan["source_manifest"]["path"]).write_text("changed source\n")
    with pytest.raises(ValueError, match="Isolated source manifest changed"):
        invoke()
    assert state["popen"] == []


def test_actual_entry_rejects_existing_GPU_compute_pid(setup, monkeypatch):
    module, _, _, state, invoke = setup
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "123\n")
    with pytest.raises(ValueError, match="Another GPU compute process"):
        invoke()
    assert state["popen"] == []


@pytest.mark.parametrize("changed", ["launch_plan_sha256", "source_manifest", "child_pid"])
def test_watch_rejects_unbound_owner_before_process_checks(setup, changed):
    module, plan, _, state, invoke = setup
    owner = {"schema": "state_prior.actual_launch.v1", "name": "fullfresh2000",
             "status": "running", "child_pid": 12345,
             "source_manifest": copy.deepcopy(plan["source_manifest"]),
             "launch_plan_sha256": "not_this_registered_plan"}
    # Compute the same plan serialization digest as invoke() below.
    owner["launch_plan_sha256"] = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
    owner[changed] = {"path": "other_manifest"} if changed == "source_manifest" else True if changed == "child_pid" else "old_plan"
    (module.HERE / "fullfresh2000_actual_launch.json").write_text(json.dumps(owner))
    with pytest.raises(ValueError):
        invoke(run="watch")
    assert state["kill"] == [] and state["popen"] == []


def test_watch_actual_valid_owner_is_CPU_only(setup):
    module, plan, _, state, invoke = setup
    owner = {"schema": "state_prior.actual_launch.v1", "name": "fullfresh2000",
             "status": "running", "child_pid": 12345,
             "source_manifest": copy.deepcopy(plan["source_manifest"]),
             "launch_plan_sha256": hashlib.sha256(json.dumps(plan).encode()).hexdigest()}
    (module.HERE / "fullfresh2000_actual_launch.json").write_text(json.dumps(owner))
    with pytest.raises(SystemExit) as error:
        invoke(run="watch", child=Child())
    assert error.value.code == 0
    record = json.loads((module.HERE / "watch_actual_launch.json").read_text())
    assert record["CPU_only"] is True and record["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert state["kill"] == [(12345, 0)]
