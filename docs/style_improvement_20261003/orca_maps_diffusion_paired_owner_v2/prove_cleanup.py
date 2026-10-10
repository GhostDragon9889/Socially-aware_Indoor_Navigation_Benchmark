"""Execute the actual cleanup AST against bounded fake child outcomes, no models."""
import ast
import hashlib
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ENTRY = HERE / "launch_paired_CPU.py"
source = ast.parse(ENTRY.read_text())
main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
handler = next(node for node in ast.walk(main) if isinstance(node, ast.ExceptHandler)
               and node.name == "error")
body = ast.Module(body=handler.body[:-1], type_ignores=[])
compiled = compile(body, str(ENTRY), "exec")


class Receipt(dict):
    def update(self, **values):
        self.updates.append(values)
        super().update(**values)


class Child:
    pid = 100
    returncode = None

    def poll(self):
        return None

    def wait(self, timeout):
        self.waits += 1
        if self.waits == 1 or self.never_closes:
            raise subprocess.TimeoutExpired("fake child", timeout)
        self.returncode = -9
        return self.returncode


cases = []
for never_closes in (True, False):
    receipt = Receipt(exit_code=None, child_exit_observed=False)
    receipt.updates = []
    child = Child()
    child.waits, child.never_closes = 0, never_closes
    kills = []
    exec(compiled, {"receipt": receipt, "child": child, "error": RuntimeError("original owner error"),
         "os": SimpleNamespace(killpg=lambda pid, sig: kills.append((pid, sig))),
         "signal": signal, "subprocess": subprocess})
    assert receipt["owner_error"] == "RuntimeError('original owner error')"
    assert receipt["driver_completed"] is False and child.waits == 2 and len(kills) == 2
    assert {"exit_code", "child_exit_observed"}.issubset(receipt.updates[-1])
    if never_closes:
        assert "cleanup_error" in receipt and receipt["exit_code"] is None and receipt["child_exit_observed"] is False
    else:
        assert "cleanup_error" not in receipt and receipt["exit_code"] == -9 and receipt["child_exit_observed"] is True
    cases.append({"child_still_unclosed": never_closes, "actual_cleanup_AST_result": dict(receipt),
                  "explicit_final_returncode_and_observed_update": True})
proof = {"schema": "local_orca_maps.paired_owner_cleanup_CPU_proof.v2", "status": "passed",
         "owner": {"path": str(ENTRY), "bytes": ENTRY.stat().st_size,
                   "sha256": hashlib.sha256(ENTRY.read_bytes()).hexdigest()},
         "source": {"path": str(Path(__file__)), "bytes": Path(__file__).stat().st_size,
                    "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
         "cases": cases, "synthetic_child_only": True, "real_child_killed": False,
         "models_executed": False, "GPU_used": False, "SDK_started": False}
with (HERE / "cleanup_CPU_proof.json").open("x") as stream:
    json.dump(proof, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"status": "passed", "cases": 2, "models_executed": False}))
