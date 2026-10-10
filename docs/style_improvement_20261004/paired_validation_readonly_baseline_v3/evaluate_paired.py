"""Paired v3: load a readonly independent byte-identical OLD baseline copy.

Frozen original/v2 sources, selected windows and model/loss/metric APIs remain
bound. No permission change or hardlink replacement is made by this reader.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/style_improvement_20261003/orca_all_temporal_paired_validation_v1/evaluate_paired.py"
ORIGINAL_SHA = "9f7732c99c33984a5ffaf24d505dc8261668b84a4ccc3703c1e5c5bc07d7e113"
V2 = HERE.parent / "paired_validation_EMA_normalization_v2/evaluate_paired.py"
V2_SHA = "1cca02c04efe00d6cb35177b878792c3f617faac215c618d956e347c485847f5"
HELPER = HERE.parent / "full_validation_loss_diagnostic_v1/normalization_contract.py"
HELPER_SHA = "2bfb3f9766a854636029960a0794a2ca79f83e3b0700109264b367c7e2b7bf15"
REGISTRATION = HERE / "source_registration.json"
OLD_ORIGINAL = {
    "path": str(ROOT / "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt"),
    "bytes": 279052920,
    "sha256": "63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac",
}
OLD_COPY = {
    **OLD_ORIGINAL,
    "path": str(ROOT / "output/diffusion_amp/formal_EMA40000_vs_63e_baseline_snapshot_v1/OLD_step5000.ckpt"),
}
SNAPSHOT_RECEIPT = Path(OLD_COPY["path"]).parent / "snapshot_receipt.json"
REPLACEMENTS = (
    ('    verify(registration["OLD"])\n',
     '    _old_snapshot_admission_v3(registration["OLD"])\n'),
    ('for name, row in (("OLD", registration["OLD"]), ("NEW", audit["checkpoint"]))}',
     'for name, row in (("OLD", _old_copy_v3), ("NEW", audit["checkpoint"]))}'),
    ('              "terminal_audit": audit_record, "NEW": audit["checkpoint"], "OLD": registration["OLD"],\n',
     '              "terminal_audit": audit_record, "NEW": audit["checkpoint"], "OLD": _old_copy_v3,\n'
     '              "OLD_declared_original": registration["OLD"],\n'
     '              "OLD_snapshot_provenance": _old_snapshot_provenance_v3(),\n'),
    ('default=ROOT / "output/diffusion_amp/formal_EMA40000_vs_63e_validation_EMA_normalization_v2")',
     'default=ROOT / "output/diffusion_amp/formal_EMA40000_vs_63e_validation_readonly_baseline_v3")'),
    ('"schema": "official_orca.paired_validation_EMA_normalization_actual_comparison.v2"',
     '"schema": "official_orca.paired_validation_readonly_baseline_actual_comparison.v3"'),
)


def metadata(info):
    return {key: getattr(info, "st_" + key) for key in
            ("dev", "ino", "size", "mtime_ns", "ctime_ns", "mode", "nlink")}


def source_record(path, *, readonly=True):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Ordinary single-link source required: " + str(path))
    if readonly and before.st_mode & 0o222:
        raise ValueError("Readonly source required: " + str(path))
    data = path.read_bytes()
    if metadata(before) != metadata(path.lstat()):
        raise ValueError("Source changed while read: " + str(path))
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, data


_EXECUTION_RECORD = source_record(Path(__file__).resolve(), readonly=False)[0]


def source_binding():
    registration_record, registration_bytes = source_record(REGISTRATION)
    registered = json.loads(registration_bytes)
    if registered.get("schema") != "official_orca.paired_readonly_baseline_source_registration.v3":
        raise ValueError("Unexpected v3 source registration")
    records, sources = {}, {}
    for role, path in (("entry", Path(__file__).resolve()), ("original", ORIGINAL),
                       ("normalization_v2", V2), ("normalization_helper", HELPER),
                       ("actual_snapshot_receipt", SNAPSHOT_RECEIPT)):
        row, data = source_record(path)
        if row != registered[role]:
            raise ValueError("Registered source changed: " + role)
        records[role], sources[role] = row, data.decode("utf-8")
    if (records["entry"] != _EXECUTION_RECORD
            or records["original"]["sha256"] != ORIGINAL_SHA
            or records["normalization_v2"]["sha256"] != V2_SHA
            or records["normalization_helper"]["sha256"] != HELPER_SHA
            or registered["OLD_declared_original"] != OLD_ORIGINAL
            or registered["OLD_actually_loaded_copy"] != OLD_COPY):
        raise ValueError("Execution source or fixed OLD snapshot contract differs")
    return {"registration": registration_record, **records}, sources


def derive(original, v2_source):
    # Parse frozen v2's four literal edits without importing/executing its entry.
    assignments = [node for node in ast.parse(v2_source).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "REPLACEMENTS" for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Unique frozen v2 literal replacement table required")
    v2_edits = ast.literal_eval(assignments[0].value)
    if len(v2_edits) != 4:
        raise ValueError("Frozen v2 must retain exactly four edits")
    edits = (*v2_edits, *REPLACEMENTS)
    derived = original
    for before, after in edits:
        if derived.count(before) != 1:
            raise ValueError("Missing or ambiguous exact replacement")
        derived = derived.replace(before, after, 1)
    reverse = derived
    for before, after in reversed(edits):
        if reverse.count(after) != 1:
            raise ValueError("Missing or ambiguous exact reverse replacement")
        reverse = reverse.replace(after, before, 1)
    if reverse != original:
        raise ValueError("Whole-byte original inverse differs")
    before_functions = {node.name: node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef)}
    after_functions = {node.name: node for node in ast.parse(derived).body if isinstance(node, ast.FunctionDef)}
    if before_functions.keys() != after_functions.keys():
        raise ValueError("Original function inventory changed")
    unchanged = {}
    for name, node in before_functions.items():
        if name in ("admit", "evaluate", "main"):
            continue
        old_text = ast.get_source_segment(original, node)
        new_text = ast.get_source_segment(derived, after_functions[name])
        if old_text != new_text or ast.dump(node, include_attributes=False) != ast.dump(after_functions[name], include_attributes=False):
            raise ValueError("Original function changed: " + name)
        unchanged[name] = hashlib.sha256(old_text.encode()).hexdigest()
    old_admit = ast.get_source_segment(original, before_functions["admit"])
    new_admit = ast.get_source_segment(derived, after_functions["admit"])
    if new_admit.replace(REPLACEMENTS[0][1].rstrip("\n"), REPLACEMENTS[0][0].rstrip("\n"), 1) != old_admit:
        raise ValueError("Terminal admission changed beyond OLD snapshot I/O")
    compile(derived, str(__file__) + ":exact_reader_delta", "exec")
    return derived, {"v2_literal_replacements_retained": 4, "v3_IO_identity_replacements": len(REPLACEMENTS),
                     "whole_byte_reverse_exact": True,
                     "derived_source_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "unchanged_function_source_sha256": unchanged,
                     "admit_only_change": "strict independent OLD snapshot instead of readonly original",
                     "terminal_and_closed_owner_admission_retained": True,
                     "model_loss_plan_RNG_metrics_and_fixed_population_retained": True}


def artifact_observation(row, *, readonly, single_link):
    path = Path(row["path"])
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1 or (single_link and before.st_nlink != 1):
        raise ValueError("Required regular artifact/link count differs: " + str(path))
    if readonly and before.st_mode & 0o222:
        raise ValueError("Actually loaded OLD copy must be readonly")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        import os
        opened_before = os.fstat(stream.fileno())
        if metadata(opened_before) != metadata(before):
            raise ValueError("Artifact changed before opened: " + str(path))
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if metadata(os.fstat(stream.fileno())) != metadata(before):
            raise ValueError("Opened artifact changed while hashed: " + str(path))
    if metadata(path.lstat()) != metadata(before):
        raise ValueError("Artifact path changed while hashed: " + str(path))
    observed = {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}
    if observed != row:
        raise ValueError("Fixed OLD artifact identity differs: " + str(path))
    return {"artifact": observed, "file_metadata": metadata(before)}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    binding, sources = source_binding()  # Sources checked before derivation/imports.
    derived, derivation = derive(sources["original"], sources["normalization_v2"])
    if argv == ["--check-source-only"]:
        print(json.dumps({"status": "source_only_ready", "sources": binding, "derivation": derivation,
                          "snapshot_or_model_read": False, "selection_read": False,
                          "NN_executed": False, "GPU_used": False, "SDK_started": False}, allow_nan=False))
        return 0
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("evaluate",), required=True)
    parser.parse_known_args(argv)  # This revision cannot reselect.
    receipt = json.loads(sources["actual_snapshot_receipt"])
    if (receipt.get("schema") != "official_orca.actual_OLD63e_independent_baseline_snapshot.v1"
            or receipt.get("status") != "copied_exact_bytes"
            or {key: receipt["source"][key] for key in OLD_ORIGINAL} != OLD_ORIGINAL
            or receipt.get("snapshot") != OLD_COPY
            or receipt.get("snapshot_mode") != "0o444" or receipt.get("snapshot_nlink") != 1
            or receipt.get("independent_inode") is not True
            or receipt.get("opened_source_before_after_and_current_path_stat_equal") is not True
            or receipt.get("source_or_hardlink_alias_modified") is not False
            or receipt.get("model_or_tensors_loaded") is not False or receipt.get("NN_SDK_GPU_used") is not False):
        raise ValueError("Bound actual root OLD snapshot receipt differs")
    helper = {"__file__": str(HELPER), "__name__": "_paired_normalization_contract_v3"}
    exec(compile(sources["normalization_helper"], str(HELPER), "exec"), helper)
    snapshot_before = None

    def snapshot_admission(declared):
        nonlocal snapshot_before
        if declared != OLD_ORIGINAL:
            raise ValueError("Frozen declaration of OLD original differs")
        current = {"declared_original": artifact_observation(OLD_ORIGINAL, readonly=False, single_link=False),
                   "actually_loaded_copy": artifact_observation(OLD_COPY, readonly=True, single_link=True)}
        original_meta = current["declared_original"]["file_metadata"]
        copy_meta = current["actually_loaded_copy"]["file_metadata"]
        if (oct(stat.S_IMODE(original_meta["mode"])) != receipt["source"]["mode"]
                or any(type(receipt["source"][key]) is not int or original_meta[meta] != receipt["source"][key]
                       for key, meta in (("nlink", "nlink"), ("device", "dev"), ("inode", "ino")))
                or oct(stat.S_IMODE(copy_meta["mode"])) != receipt["snapshot_mode"]):
            raise ValueError("Original permissions/links or independent snapshot differ from root observation")
        if (original_meta["dev"], original_meta["ino"]) == (copy_meta["dev"], copy_meta["ino"]):
            raise ValueError("OLD copy must be independent, not the original inode")
        if snapshot_before is not None and current != snapshot_before:
            raise ValueError("Declared original or actual OLD snapshot changed during evaluation")
        snapshot_before = current

    def snapshot_provenance():
        snapshot_admission(OLD_ORIGINAL)
        return {"schema": "official_orca.independent_OLD_baseline_snapshot_observation.v3",
                **snapshot_before, "same_full_bytes_and_SHA256": True,
                "loaded_copy_readonly_single_link": True, "original_permissions_or_links_modified": False,
                "actual_root_snapshot_receipt": binding["actual_snapshot_receipt"],
                "root_tool_exit_authenticated_by_this_reader": False, "tensor_validation_by_snapshot_gate": False}

    def normalization_contract(old, new, audit):
        reports = {}
        for name, planner, weight_source in (("OLD", old, "raw"), ("NEW", new, "EMA")):
            actual = {}
            for key in ("state_mean", "state_std"):
                buffer = getattr(planner.model, key)
                if str(buffer.dtype) != "torch.float32" or tuple(buffer.shape) != (5,):
                    raise ValueError("Original FP32 five-state normalization buffers required")
                actual[key] = buffer.tolist()
            reports[name] = helper["validate_normalization"](actual, asdict(planner.config), planner.trained_steps,
                weight_source=weight_source, ema_recipe=audit["training_recipe"]["EMA"])
        return {"models": reports, "normalization_buffers_rewritten": False,
                "legacy_exact_equality_registration_flag_overridden_by_v2_source": True,
                "same_stored_normalization_between_models_claimed": False}

    def reader_revision():
        after, _ = source_binding()
        if after != binding:
            raise ValueError("Reader/helper source changed during actual evaluation")
        return {"schema": "official_orca.paired_reader_revision.v3", "entry": binding["entry"],
                "source_binding": binding, "derivation": derivation,
                "original_frozen_evaluator_executed_bitexact": False,
                "original_selection_loss_plan_RNG_metrics_and_terminal_admission_retained": True,
                "normalization_gate_only_accepts_shared_recipe_supported_values": True,
                "OLD_original_declared_path_preserved_actual_load_is_independent_copy": True}

    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "_paired_validation_readonly_baseline_v3",
                 "_normalization_contract_v2": normalization_contract, "_reader_revision_v2": reader_revision,
                 "_old_snapshot_admission_v3": snapshot_admission, "_old_snapshot_provenance_v3": snapshot_provenance,
                 "_old_copy_v3": OLD_COPY.copy()}
    exec(compile(derived, str(__file__) + ":exact_reader_delta", "exec"), namespace)
    return namespace["main"](argv)


if __name__ == "__main__":
    raise SystemExit(main())
