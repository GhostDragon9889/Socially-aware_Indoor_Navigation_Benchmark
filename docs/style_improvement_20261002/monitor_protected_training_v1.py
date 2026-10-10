"""Read-only progress of registered isolated training; no Torch or SDK."""
from pathlib import Path
import json
import time
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_v1"
WATCH = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_snapshots_v1/ledger.json"
started = time.monotonic()
while time.monotonic() - started < 12000:
    records = []
    metrics = OUT / "metrics.jsonl"
    if metrics.exists():
        with metrics.open("rb") as stream:
            stream.seek(max(0, metrics.stat().st_size - 16384))
            for line in stream.read().splitlines():
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(value, dict) and "update_count" in value:
                    records.append(value)
    latest = records[-1] if records else {}
    report = {key: latest.get(key) for key in ("update_count", "physical_transitions", "elapsed_s", "fall_resets", "time_limit_resets")}
    if WATCH.exists():
        watcher = json.loads(WATCH.read_text())
        report["snapshot_status"] = watcher["status"]
        report["sealed_updates"] = [row["update_count"] for row in watcher["snapshots"]]
        report["missing_updates"] = watcher["missing"]
    result = OUT / "result.json"
    if result.exists():
        final = json.loads(result.read_text())
        report["terminal_status"] = final["status"]
        report["terminal_update_count"] = final["update_count"]
    print(json.dumps(report), flush=True)
    if result.exists():
        break
    time.sleep(45)
