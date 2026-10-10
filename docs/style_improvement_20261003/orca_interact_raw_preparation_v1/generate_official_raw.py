"""Seeded CPU delegation to the unchanged pinned TRACE raw generator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_sources():
    plan = json.loads((HERE / "registration.json").read_text())
    trace = Path(plan["trace_repository"])
    head = subprocess.check_output(["git", "-C", str(trace), "rev-parse", "HEAD"], text=True).strip()
    if head != plan["trace_commit"] or subprocess.check_output(["git", "-C", str(trace), "status", "--porcelain"], text=True).strip():
        raise ValueError("The registered clean TRACE commit is required")
    for row in plan["trace_files"]:
        if record(row["path"]) != row:
            raise ValueError("Registered TRACE source bytes differ")
        relative = str(Path(row["path"]).relative_to(trace))
        original = subprocess.check_output(["git", "-C", str(trace), "show", f"{head}:{relative}"])
        if original != Path(row["path"]).read_bytes():
            raise ValueError("TRACE working source differs from the fixed commit")
    for row in plan["local_comparison_sources"]:
        if record(row["path"]) != row:
            raise ValueError("Frozen local comparison source differs")
    if record(plan["scene_splits"]["path"]) != plan["scene_splits"]:
        raise ValueError("Preassigned scene split registration differs")
    rvo = Path(plan["rvo2_repository"])
    if subprocess.check_output(["git", "-C", str(rvo), "rev-parse", "HEAD"], text=True).strip() != plan["rvo2_commit"]:
        raise ValueError("The registered RVO2 source commit is required")
    output = Path(plan["output"])
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError("The new raw output must be absent or empty; never overwrite data")
    versions = {name: importlib.metadata.version(name) for name in plan["CPU_packages"]}
    return plan, {"status": "prepared_source_bound_only", "trace_commit": head,
                  "trace_clean_and_selected_files_equal_commit": True,
                  "CPU_package_versions": versions, "python": sys.version,
                  "output": str(output), "generation_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan, prepared = prepare_sources()
    if args.prepare_only:
        print(json.dumps(prepared))
        return 0
    import numpy as np
    generator_path = Path(plan["official_raw_argv"][0])
    specification = importlib.util.spec_from_file_location("registered_official_TRACE_ORCA_raw", generator_path)
    generator = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(generator)
    np.random.seed(plan["generation_seed"])
    sys.argv = plan["official_raw_argv"]
    generator.args = generator.parse_args()
    generator.gen_dataset(generator.args)
    print(json.dumps({"status": "official_raw_generator_returned", "output": plan["output"],
                      "seed": plan["generation_seed"], "ORCACorpus_connected": False,
                      "author_dataset": False, "source_commit": plan["trace_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
