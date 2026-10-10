"""Seal byte-identical checkpoints; this does not audit physical resume semantics.

The independent mixed-curriculum auditor must check the sealed payload before
it is used for evaluation. Missing milestones remain missing in the ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def counters(bundle, actors, rollout):
    if bundle.get("format") != "navisaaclab_method_reconstruction_training" or bundle.get("version") != 1:
        raise ValueError("Unsupported checkpoint format")
    required = {"config", "provenance", "model", "optimizer", "discriminator_optimizer",
                "normalizers", "rng", "policy_replay", "runner_state"}
    if not required <= bundle.keys() or bundle.get("physical_runner_state_included") is not True:
        raise ValueError("Complete training payload missing")
    update, transitions = bundle["update_count"], bundle["physical_transition_count"]
    if type(update) is not int or update < 1 or type(transitions) is not int:
        raise ValueError("Invalid trainer counters")
    runner = bundle["runner_state"]
    if transitions != update * actors * rollout or runner["control_step"] != update * rollout:
        raise ValueError("Actual budget mismatch")
    if runner["runner_contract"]["trajectory_source"] != "mixed_amass_procedural":
        raise ValueError("Wrong training recipe")
    if runner["physical"]["root_pos"].shape != (actors, 3) or runner["plans"].shape != (actors, 151, 2):
        raise ValueError("Wrong physical/plan dimensions")
    return update, transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--final-update", type=int, default=2000)
    parser.add_argument("--targets", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--max-seconds", type=float, default=6000)
    args = parser.parse_args()
    if (args.num_envs < 2 or args.num_envs % 2 or args.rollout_steps < 1
            or args.final_update < 1 or not 0 < args.max_seconds < 86400
            or any(type(t) is not int or not 1 <= t <= args.final_update for t in args.targets)
            or len(args.targets) != len(set(args.targets)) or args.final_update not in args.targets):
        parser.error("Invalid fixed budget, target list or timeout")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(2)
    args.snapshot_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = args.snapshot_dir / "ledger.json"
    ledger = {"status": "watching", "watcher_sha256": digest(Path(__file__)),
              "training_output": str(args.training_output), "targets": args.targets,
              "budget": {"actors": args.num_envs, "rollout": args.rollout_steps,
                         "final_update": args.final_update}, "snapshots": [], "missing": [],
              "semantic_resume_validation_performed": False,
              "gate": "serialization, full payload presence, counters, recipe and dimensions only",
              "independent_full_payload_audit_required_before_evaluation": True}

    def save():
        temporary = ledger_path.with_suffix(".pending")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        os.replace(temporary, ledger_path)

    save()
    started, observed, copy_path = time.monotonic(), None, None
    source = args.training_output / "last.pt"
    try:
        while time.monotonic() - started < args.max_seconds:
            if source.exists() and signature(source) != observed:
                before = signature(source)
                try:
                    bundle = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
                    update, transitions = counters(bundle, args.num_envs, args.rollout_steps)
                    del bundle
                    accounted = {r["update_count"] for r in ledger["snapshots"] + ledger["missing"]}
                    for target in args.targets:
                        if target < update and target not in accounted:
                            ledger["missing"].append({"update_count": target,
                                                      "first_later_observed_update": update})
                    if update in args.targets and update not in accounted:
                        destination = args.snapshot_dir / f"step_{update}.pt"
                        copy_path = args.snapshot_dir / f".copy_{update}_{os.getpid()}.pt"
                        with source.open("rb") as incoming, copy_path.open("xb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing, 16 * 1024 * 1024)
                            outgoing.flush()
                            os.fsync(outgoing.fileno())
                        copied = torch.load(copy_path, map_location="cpu", weights_only=True, mmap=True)
                        copied_counters = counters(copied, args.num_envs, args.rollout_steps)
                        del copied
                        checksum = digest(copy_path)
                        if (copied_counters != (update, transitions) or signature(source) != before
                                or digest(source) != checksum or signature(source) != before):
                            raise OSError("Mutable source changed during snapshot; retry")
                        os.link(copy_path, destination)  # Exclusive publication, never replace an existing file.
                        copy_path.unlink()
                        copy_path = None
                        row = {"update_count": update, "physical_transition_count": transitions,
                               "path": str(destination), "bytes": destination.stat().st_size,
                               "sha256": checksum, "identical_source_bytes_verified": True}
                        ledger["snapshots"].append(row)
                        print(json.dumps(row), flush=True)
                    observed = before
                    save()
                except (OSError, EOFError, RuntimeError) as error:
                    if copy_path is not None and copy_path.exists():
                        copy_path.unlink()
                    copy_path = None
                    print(f"Mutable checkpoint read retry: {type(error).__name__}", flush=True)
            result_path = args.training_output / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result.get("status") != "passed":
                    ledger.update(status="training_failed", terminal_status=result.get("status"))
                    break
                if (result.get("update_count") != args.final_update
                        or result.get("physical_transition_count") != args.final_update * args.num_envs * args.rollout_steps):
                    raise ValueError("Terminal result budget mismatch")
                if any(r["update_count"] == args.final_update for r in ledger["snapshots"]):
                    ledger.update(status="sealed_with_missing" if ledger["missing"] else "sealed",
                                  terminal_result_sha256=digest(result_path))
                    break
            time.sleep(.25)
        else:
            ledger["status"] = "timed_out"
    except Exception as error:
        ledger.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if copy_path is not None and copy_path.exists():
            copy_path.unlink()
        ledger["elapsed_s"] = time.monotonic() - started
        save()
    return 0 if ledger["status"] == "sealed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
