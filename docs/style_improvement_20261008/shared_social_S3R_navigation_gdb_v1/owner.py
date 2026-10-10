"""Run the unchanged full navigation under GDB; retain real debugger exit/trace."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts import evaluate_shared_social_recovery as entry  # noqa: E402
from scripts import run_shared_social_post_training as original  # noqa: E402

HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/style_improvement_20261008/shared_social_S3R_evaluation_v2"
OUTPUT = ROOT / "output/evaluation/shared_social_S3R_navigation_gdb6000_v1"


def save(name, data):
    temporary = HERE / (name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(HERE / name)


if "--detach" in sys.argv:
    if (HERE / "owner.log").exists():
        raise FileExistsError("Debug owner already launched")
    with (HERE / "owner.log").open("xb") as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    print(json.dumps({"owner_pid": process.pid, "actual_debugger_exit_observed": False}))
    raise SystemExit(0)

if OUTPUT.exists() or (HERE / "owner_state.json").exists():
    raise FileExistsError("Use a new debug output; retain prior runs")
bindings = {stage: entry.prepare(stage)[1] for stage in entry.PARENTS}
inputs = [record(PREVIOUS / name) for name in ("completion.json", "training_execution.json", "owner_plan.json")]
receipt = json.loads((PREVIOUS / "completion.json").read_text())
if receipt["status"] != "failed_preserved" or receipt["steps"][0]["actual_exit_code"] != 1:
    raise ValueError("Require the retained actual failed evaluation owner")
command = ["/usr/bin/gdb", "-q", "-batch", "-return-child-result", "-ex", "set pagination off",
           "-ex", "set confirm off", "-ex", "run", "-ex", "thread apply all bt 12", "--args",
           sys.executable, str(ROOT / "scripts/evaluate_shared_social_recovery.py"), "--stage", "navigation",
           "--execution-receipt", str(PREVIOUS / "training_execution.json"),
           "--diffusion-checkpoint", str(original.DIFFUSION), "--diffusion-sha256", original.DIFFUSION_SHA,
           "--control-steps", "6000", "--output", str(OUTPUT)]
state = {"schema": "shared_social.navigation_native_debug_execution.v1", "status": "running",
         "owner_pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
         "command": command, "bindings": bindings, "inputs": inputs, "owner_source": record(__file__),
         "controls": 6000, "debugger_attached": True, "physics_or_metric_code_changed": False,
         "quality_or_success_certified": False}
save("owner_state.json", state)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1", PYTHONFAULTHANDLER="1",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", LC_ALL="C")
with (HERE / "gdb.log").open("xb") as stream:
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT)
    state["debugger_pid"] = process.pid
    save("owner_state.json", state)
    code = process.wait()
state.update(status="debugger_exited", actual_debugger_exit_code=code,
             exit_observation="subprocess.Popen.wait with gdb -return-child-result",
             finished_at_utc=datetime.now(timezone.utc).isoformat(), log=record(HERE / "gdb.log"),
             actual_pipeline_success_claimed=False)
if bindings != {stage: entry.prepare(stage)[1] for stage in entry.PARENTS}:
    state["source_changed"] = True
save("owner_state.json", state)
save("completion.json", state)
(HERE / "completion.json").chmod(0o444)
print(json.dumps({"status": state["status"], "actual_debugger_exit_code": code}), flush=True)
raise SystemExit(0 if code == 0 else 1)
