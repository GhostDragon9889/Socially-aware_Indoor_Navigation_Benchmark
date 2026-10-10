"""Stdlib-only source/protocol admission for future actual isolated evaluations.

Never imports the isolated launcher, Torch, SDK or simulation entrypoints.
Hashes copied source and the actual executed-import ledger. Large shared banks
are checked by symlink/stat, not falsely described as fully rehashed here.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_v1"
CODE = BUNDLE / "code"
MANIFEST_SHA = "f2af12c16bb7f5a0808657e4f2589bc3c4db91370c4201086c5b4416bdafb8fa"
LAUNCHER_SHA = "2f2a8137599f83a259073e210bac4c283303bae03c765682aefb4885bc24d3a5"
CONTROL_AST_SHA = "2273a011000d5579b38d76fab5ac0e14d526ecfa905f8bf111445444e1902f8b"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_protocol(result, protocol, registration):
    """Require actual complete captures with every registered CLI value intact."""
    require(protocol in {"tracking", "navigation"}, "Unknown registered protocol")
    require(result.get("status") == "passed" and result.get("error") is None,
            "Actual evaluation must finish successfully before audit")
    args = result.get("arguments", {})
    expected = registration[protocol]["parsed"]
    require(set(args) == set(expected), "Actual argument schema differs from old146 registration")
    for key, value in expected.items():
        if key == "output" or protocol == "tracking" and key == "checkpoint_sha256":
            continue
        require(type(args[key]) is type(value) and args[key] == value,
                "Actual registered evaluation argument differs:" + key)
    if protocol == "tracking":
        digest = args.get("checkpoint_sha256")
        require(isinstance(digest, str) and len(digest) == 64 and
                all(char in "0123456789abcdef" for char in digest), "Actual checkpoint SHA required")
        rows, aggregate = result.get("sequences", []), result.get("aggregate", {})
        require(len(rows) == 22 and len({row["motion_id"] for row in rows}) == 22 and
                type(aggregate.get("observed_metric_steps")) is int and
                aggregate["observed_metric_steps"] == 11982 and aggregate.get("complete_success_denominator") == 22 and
                all(type(row.get("observed_metric_steps")) is int and row["observed_metric_steps"] > 0 and
                    row.get("expected_control_steps") == row["observed_metric_steps"] and
                    row.get("complete_clip") is True and row.get("eligible") is True and
                    row.get("in_full_sequence_success_denominator") is True and
                    row.get("app_or_simulation_stopped") is False for row in rows) and
                sum(row["observed_metric_steps"] for row in rows) == 11982,
                "Full failed-inclusive22/11982 capture required")
    else:
        require(type(result.get("actual_control_steps")) is int and result["actual_control_steps"] == 6000,
                "Full4x6000 navigation controls required")
        require(result.get("physics", {}).get("fall_definition") == {
            "pelvis_ground_height_less_than_m": .45, "root_up_z_less_than": .1},
            "Existing isolated physical fall definition changed")


def validate_observed_ledger(ledger, events, declared, protocol, registration, admission_record,
                             checkpoint_record, result_arguments):
    """Validate recorded executions without inventing finalization/imports."""
    require(ledger.get("schema") == "paper_amp.actual_owned_import_ledger.v3" and
            ledger.get("source_manifest_sha256") == MANIFEST_SHA and
            ledger.get("launcher_sha256") == LAUNCHER_SHA and ledger.get("code") == str(CODE) and
            ledger.get("all_observed_owned_modules_isolated") is True,
            "Actual isolatedv3 evaluation ledger header required")
    count = ledger.get("event_count")
    require(type(count) is int and 1 <= count <= 4096 and len(events) == count and
            [row.get("sequence") for row in events] == list(range(1, count + 1)),
            "Exact committed evaluation event sequence required")
    modules = ledger.get("owned_modules", [])
    observed = {row["module"]: row for row in events if row.get("event") == "owned_execution"}
    require(modules and len(modules) == len(observed) == len({row["module"] for row in modules}),
            "Actual owned executions and atomic ledger disagree")
    for row in modules:
        path = Path(row["path"])
        require(path.is_absolute() and path.resolve().is_relative_to(CODE), "Owned execution escaped isolated code")
        relative = "code/" + str(path.relative_to(CODE))
        expected = declared.get(relative)
        event = observed[row["module"]]
        require(expected is not None and row.get("execution_observed") is True and
                row["sha256"] == expected["sha256"] and row["bytes"] == expected["bytes"] and
                all(event.get(key) == row[key] for key in ("module", "path", "bytes", "sha256", "execution_observed")),
                "Actual owned module/hash/event differs:" + row["module"])
    admitted = [row for row in events if row.get("event") == "registered_evaluation_admitted"]
    entries = [row for row in events if row.get("event") == "registered_entry_selected"]
    require(len(admitted) == len(entries) == 1, "Registered actual evaluation entry/admission event required")
    admitted, entry = admitted[0], entries[0]
    executor_path = HERE / "run_isolated_state_action_evaluation_v1.py"
    registration_record = record(HERE / "state_action_evaluation_registration_v1/registration.json")
    entry_name = "scripts/" + ("benchmark_paper_motion_tracking.py" if protocol == "tracking" else "train_paper_amp.py")
    expected_entry = {"path": str(CODE / entry_name), "bytes": declared["code/" + entry_name]["bytes"],
                      "sha256": declared["code/" + entry_name]["sha256"]}
    executor = load_module("future_actual_registered_evaluation_gate", executor_path)
    _, argv = executor.build_command(protocol, checkpoint_record["path"], checkpoint_record["sha256"],
                                     result_arguments["output"], registration)
    require(admitted.get("protocol") == protocol and admitted.get("adapter") == record(executor_path) and
            admitted.get("registration") == registration_record and admitted.get("payload_admission") == admission_record and
            admitted.get("checkpoint") == checkpoint_record and admitted.get("entry") == expected_entry and
            admitted.get("runner_arguments") == argv and entry.get("path") == expected_entry["path"] and
            entry.get("sha256") == expected_entry["sha256"] and entry.get("argv") == argv,
            "Actual registered entry/argv/checkpoint/admission binding differs")
    require(any(row["path"] == expected_entry["path"] for row in modules),
            "Benchmark/driver execution not actually observed")
    require(not any(row.get("event") == "owned_import_failed" for row in events), "Actual owned import failed")
    return {"owned_executions": len(modules), "committed_events": count,
            "runtime_finalization_observed": ledger.get("runtime_finalization_observed"),
            "unobserved_finalization_or_imports_inferred": False}


def admit_actual_evaluation(result, protocol, admission_path, import_report, *, require_check=None):
    """Read current immutable copy and actual completed-capture execution evidence."""
    check = require_check or require
    registration_path = HERE / "state_action_evaluation_registration_v1/registration.json"
    registration = json.loads(registration_path.read_text())
    validate_protocol(result, protocol, registration)
    source_path, launcher_path = BUNDLE / "source_manifest_v3.json", BUNDLE / "launch_isolated_v3.py"
    source_record, launcher_record = record(source_path), record(launcher_path)
    check(source_record["sha256"] == MANIFEST_SHA and launcher_record["sha256"] == LAUNCHER_SHA,
          "exact isolatedv3 source manifest and launcher")
    source = json.loads(source_path.read_text())
    declared = {row["path"]: row for row in source["files"]}
    actual_names = {str(path.relative_to(BUNDLE)) for path in CODE.rglob("*") if path.is_file() and not path.is_symlink()}
    check(len(declared) == len(source["files"]) == 926 and set(declared) == actual_names,
          "exact926 immutable copied evaluation files")
    for name, expected in declared.items():
        path = BUNDLE / name
        actual = record(path)
        check(not path.is_symlink() and path.resolve().is_relative_to(CODE) and
              actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"] and
              not path.stat().st_mode & 0o222, "actual isolated source bytes:" + name)
    for link in source["shared_resource_links"]:
        path = BUNDLE / link["path"]
        check(path.is_symlink() and str(path.resolve()) == link["target"], "shared resource link:" + link["path"])
    for item in source["resources"]:
        stat = Path(item["path"]).stat()
        check(stat.st_size == item["bytes"] and stat.st_mtime_ns == item["mtime_ns"],
              "shared resource stat without repeated whole-bank SHA:" + item["path"])
    executor = load_module("future_actual_evaluation_executor_admission", HERE / "run_isolated_state_action_evaluation_v1.py")
    admission = json.loads(Path(admission_path).read_text())
    checkpoint_key = "checkpoint" if protocol == "tracking" else "resume"
    checkpoint_record = record(result["arguments"][checkpoint_key])
    executor.validate_admission(admission, checkpoint_record)
    check(admission.get("actual_terminal_fresh2000_confirmed") is True,
          "actual complete fresh2000 terminal payload admitted")
    if protocol == "tracking":
        check(result["arguments"]["checkpoint_sha256"] == checkpoint_record["sha256"], "actual captured controller SHA")
    ledger_bytes = Path(import_report).read_bytes()
    ledger = json.loads(ledger_bytes)
    lines = Path(ledger["event_ledger"]).read_text().splitlines()
    check(len(lines) >= ledger["event_count"], "evaluation event ledger committed prefix exists")
    event_lines = lines[:ledger["event_count"]]
    observed = validate_observed_ledger(ledger, [json.loads(line) for line in event_lines], declared,
        protocol, registration, record(admission_path), checkpoint_record, result["arguments"])
    check(True, "actual observed isolated evaluation entry/argv/source ledger")
    runner = ast.parse((CODE / "scripts/train_paper_amp.py").read_text())
    loops = [node for node in ast.walk(runner) if isinstance(node, ast.For) and
             isinstance(node.target, ast.Name) and node.target.id == "step"]
    check(len(loops) == 1, "one old146 sole physical control loop")
    loop_sha = hashlib.sha256(ast.dump(loops[0], include_attributes=False).encode()).hexdigest()
    check(loop_sha == CONTROL_AST_SHA, "old146 sole control-loop AST frozen exactly")
    resources = {"controller_checkpoint": checkpoint_record}
    for name, key in (("diffusion_checkpoint", "diffusion_checkpoint"), ("profile", "config"),
                      ("scene", "scene_config"), ("manifest", "manifest")):
        if result["arguments"].get(key):
            resources[name] = record(result["arguments"][key])
    check(resources["manifest"] == registration["sources"]["fixed_manifest"], "fixed v7 manifest exact SHA")
    if protocol == "navigation":
        check(resources["profile"] == registration["sources"]["registered_profile"] and
              resources["scene"] == registration["sources"]["scene"], "registered actual profile/scene exact bytes")
        diffusion_sha = result.get("evaluation_planner_change", {}).get("evaluation_sha256", result["provenance"]["diffusion_sha256"])
        check(resources["diffusion_checkpoint"]["sha256"] == diffusion_sha, "runtime deployed Diffusion checksum")
    return {"isolated_source_manifest": source_record, "isolated_launcher": launcher_record,
            "registered_protocol": record(registration_path), "actual_terminal_admission": record(admission_path),
            "actual_evaluation_import_report": record(import_report), "actual_evaluation_imports": observed,
            "control_loop_ast_sha256": loop_sha, "copied_source_files_full_SHA_recomputed": 926,
            "large_shared_banks_full_SHA_recomputed": False, "resources_full_sha256_recomputed": resources,
            "controller_provenance": "actual immutable resume/checkpoint full SHA, bound to terminal admission and execution ledger",
            "source_binding_limit": "Actual synchronous executed-owned-module ledger; unobserved imports/finalization not invented. SDK/native external package bytes are outside the926 owned-source boundary.",
            "ledger_bytes": ledger_bytes, "event_prefix_bytes": ("\n".join(event_lines) + "\n").encode()}
