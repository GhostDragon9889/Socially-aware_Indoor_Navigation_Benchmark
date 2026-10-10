"""Narrow CPU source review; never import trainer, SDK or checkpoint readers."""
import ast
import hashlib
import json
from pathlib import Path
import stat
import sys
import tarfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = HERE.parent


def record(path):
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    started = time.perf_counter()
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append({"name": name, "passed": True})

    preparation = DOC / "state_only_matched_q_training_preparation_v1"
    reader = DOC / "state_only_matched_q_training_audit_v1"
    freezes = [(preparation / "source_freeze_v1.json", "350c27770016983947024936f83385112104e8cfd4de7ecbb3831fe7ffe55564"),
               (reader / "source_freeze_v1.json", "661ce76cdb7aa5c05cc73bbc3eb39ea6e7c4ea2851281987b1a7ab05cd11dcfa")]
    for path, digest in freezes:
        check("freeze_identity:" + path.parent.name, record(path)["sha256"] == digest)
        freeze = json.loads(path.read_text())
        check("prepared_scope:" + path.parent.name, freeze["status"] == "frozen_prepared_only")
        for row in freeze["source_files"]:
            source = Path(row["path"])
            check("frozen_source:" + str(source.relative_to(DOC)), not source.is_symlink()
                  and stat.S_ISREG(source.stat().st_mode) and source.stat().st_nlink == 1
                  and source.stat().st_mode & 0o222 == 0 and record(source) == row)
    rf = json.loads(freezes[1][0].read_text())
    check("reader_archive_SHA", record(rf["archive"]["path"]) == rf["archive"])
    with tarfile.open(rf["archive"]["path"], "r:gz") as tar:
        members = tar.getmembers()
        check("reader_archive_nine_regular_unique", len(members) == 9
              and len({member.name for member in members}) == 9 and all(member.isfile() for member in members))
        for member in members:
            check("reader_archive_bytes:" + member.name, tar.extractfile(member).read() == (reader / member.name).read_bytes())

    manifest_path = Path(rf["source_manifest"]["path"])
    source = json.loads(manifest_path.read_text())
    parent = json.loads(Path(source["previous_source_manifest"]["path"]).read_text())
    new_rows = {row["path"]: row for row in source["files"]}
    old_rows = {row["path"]: row for row in parent["files"]}
    changed = {name for name in old_rows if (new_rows[name]["bytes"], new_rows[name]["sha256"])
               != (old_rows[name]["bytes"], old_rows[name]["sha256"])}
    added = set(new_rows) - set(old_rows)
    check("exact_manifest935_to936_delta", len(old_rows) == len(parent["files"]) == 935
          and len(new_rows) == len(source["files"]) == 936 and set(old_rows) <= set(new_rows)
          and changed == {"code/scripts/train_paper_amp.py"}
          and added == {"code/configs/diffusion_amp/isolated_state_only_matched_q_absolute.yaml"})
    delta = json.loads((DOC / "state_only_matched_q_preparation_v1/source_delta.json").read_text())
    parent_text = Path(delta["parent_runner"]["path"]).read_text()
    derived = parent_text
    for operation in delta["runner_operations"]:
        check("exact_runner_fragment:" + operation["purpose"], derived.count(operation["before"]) == 1)
        derived = derived.replace(operation["before"], operation["after"])
    actual_runner = manifest_path.parent / "code/scripts/train_paper_amp.py"
    check("entire_runner_exact_three_fragments", derived == actual_runner.read_text()
          and ast.dump(ast.parse(derived)) == ast.dump(ast.parse(actual_runner.read_text())))
    old_profile = yaml.safe_load(Path(delta["parent_profile"]["path"]).read_text())
    new_profile = yaml.safe_load((manifest_path.parent / next(iter(added))).read_text())
    training_delta = {key: (old_profile["training"].get(key), new_profile["training"].get(key))
                      for key in old_profile["training"].keys() | new_profile["training"].keys()
                      if old_profile["training"].get(key) != new_profile["training"].get(key)}
    check("only_three_training_flags", training_delta == {
        "discriminator_input": ("state_action", "state"), "warm_start_discriminator": (False, True),
        "warm_start_action_discriminator_from_state_prior": (True, False)})
    execution_fields = (old_profile.keys() | new_profile.keys()) - {"training", "profile", "assumptions", "initialization"}
    check("all_other_execution_profile_fields_exact", all(old_profile[key] == new_profile[key] for key in execution_fields))
    check("initialization_only_description_changes", all(old_profile["initialization"][key] == new_profile["initialization"][key]
          for key in old_profile["initialization"].keys() | new_profile["initialization"].keys() if key != "discriminator"))

    edits = json.loads((reader / "adapter_exact_replacements.json").read_text())
    main_text = (reader / "audit_state_only_training.py").read_text()
    for before, after in reversed(edits["main_replacements"]):
        main_text = main_text.replace(after, before)
    helper_text = (reader / "state_prior_checks.py").read_text()
    helper_text = helper_text.replace(edits["helper_archive_derivation_after"], edits["helper_archive_derivation_before"])
    helper_text = helper_text.replace(edits["helper_state_prior_after"], edits["helper_state_prior_before"])
    for before, after in reversed(edits["helper_replacements"]):
        helper_text = helper_text.replace(after, before)
    for name, text, row in (("main", main_text, edits["parent_main"]), ("helper", helper_text, edits["parent_helper"])):
        old_text = Path(row["path"]).read_text()
        check("reader_inverse_whole_bytes_AST:" + name, text == old_text
              and ast.dump(ast.parse(text)) == ast.dump(ast.parse(old_text)))
    check("q_callback_whole_bytes_exact", (reader / "expert_sampling_checks.py").read_bytes()
          == Path(edits["parent_q_callback"]["path"]).read_bytes())
    for directory, receipt_name in ((preparation, "actual_CPU_process_closed.json"),
                                    (reader, "prepared_boundary_process_closed.json")):
        receipt = json.loads((directory / receipt_name).read_text())
        check("recorded_actual_CPU_closed:" + directory.name, receipt["status"] == "closed"
              and type(receipt["exit_code"]) is int and receipt["exit_code"] == 0
              and receipt["child_exit_observed"] is True and receipt["driver_completed"] is True
              and record(receipt["artifact"]["path"]) == receipt["artifact"])
    boundary = json.loads((reader / "prepared_boundary_CPU.json").read_text())
    check("recorded_boundary_not_reexecuted", boundary["status"] == "passed"
          and all(row["passed"] is True for row in boundary["checks"])
          and len(boundary["negative_cases"]) == 4 and all(row["rejected"] is True for row in boundary["negative_cases"])
          and boundary["synthetic_tensor_metadata_only"] is True and boundary["actual_training_payload_admitted"] is False)
    plan = json.loads((preparation / "root_GPU_launch_commands.json").read_text())
    check("only_fresh4_and2000_registered", set(plan["commands"]) == {"pilot4", "fullfresh2000"}
          and all(not any(value == "--resume" or value.startswith("--resume=") for value in command)
                  for command in plan["commands"].values()))
    check("no_torch_or_SDK_import", not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr"} for name in sys.modules))
    report = {"schema": "state_only_matched_q.independent_source_peer.v1", "status": "passed_prepared_source_only",
              "checks": checks, "freeze_records": [record(path) for path, _ in freezes],
              "source_manifest": record(manifest_path), "training_flags_delta": training_delta,
              "required_findings": [], "runtime_weight_transfer_executed_by_peer": False,
              "full_training_payload_admitted_by_peer": False, "boundary_suite_reexecuted": False,
              "GPU_used": False, "SDK_started": False, "models_or_weights_read": False,
              "sealed34_outputs_read": False, "future_physical_quality_confirmed": False,
              "CPU_elapsed_s": time.perf_counter() - started}
    with (HERE / "source_peer.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "report": record(HERE / "source_peer.json"),
                      "CPU_elapsed_s": report["CPU_elapsed_s"]}))


if __name__ == "__main__":
    main()
