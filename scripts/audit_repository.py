"""Inventory source and compare it with a NavIsaacLab reference without executing it."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


SOURCE_DIRS = ("src", "tests", "configs", "data/scripts", "data/navigation_maps", "examples", "tools", "packages", "scripts", ".github")
REFERENCE_SOURCE_DIRS = ("protomotions", "CrowdSim", "src", "tests", "configs", "data/scripts", "examples", "deployment", "pyroki", "usd_convert", "scripts", ".github")


def reference_relative(relative):
    """Compare canonical source paths with the original NavIsaacLab layout."""
    prefixes = (("src/protomotions/", "protomotions/"), ("src/CrowdSim/", "CrowdSim/"),
                ("src/deployment/", "deployment/"), ("configs/crowdsim/", "CrowdSim/config/"),
                ("tools/retargeting/", "pyroki/"), ("tools/asset_conversion/", "usd_convert/"))
    for current, original in prefixes:
        if relative.startswith(current):
            return original + relative[len(current):]
    return relative
EXTENSIONS = {".py", ".sh", ".yaml", ".yml", ".toml", ".json"}


def inventory(root: Path, reference: Path) -> list[dict]:
    paths = {path for folder in SOURCE_DIRS for path in (root / folder).rglob("*")
             if path.is_file() and path.suffix in EXTENSIONS
             and not any(part in {"__pycache__", ".venv", "build", "dist"} for part in path.parts)}
    paths.update(path for path in root.iterdir() if path.is_file() and path.suffix in EXTENSIONS)
    rows = []
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        previous = reference / reference_relative(relative)
        previous_hash = hashlib.sha256(previous.read_bytes()).hexdigest() if previous.is_file() else None
        row = {"path": relative, "bytes": len(content), "lines": len(content.splitlines()),
               "sha256": hashlib.sha256(content).hexdigest(), "reference_sha256": previous_hash,
               "reference_status": "only_current" if previous_hash is None else "identical" if hashlib.sha256(content).hexdigest() == previous_hash else "different",
               "syntax_status": "not_checked", "syntax_error": None, "ruff_findings": []}
        try:
            if path.suffix == ".py":
                tree = ast.parse(content, filename=relative)
                compile(tree, relative, "exec")
                row["definitions"] = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
                row["imports"] = sorted({node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {item.name for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names})
                row["syntax_status"] = "passed"
            elif path.suffix == ".sh":
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                row["syntax_status"] = "passed" if result.returncode == 0 else "failed"
                row["syntax_error"] = result.stderr or None
            elif path.suffix in {".yaml", ".yml"}:
                import yaml
                yaml.safe_load(content)
                row["syntax_status"] = "passed"
            elif path.suffix == ".toml":
                import tomllib
                tomllib.loads(content.decode())
                row["syntax_status"] = "passed"
            elif path.suffix == ".json":
                json.loads(content)
                row["syntax_status"] = "passed"
        except Exception as exc:
            row["syntax_status"] = "failed"
            row["syntax_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/audit"))
    args = parser.parse_args()
    root, reference = args.root.resolve(), args.reference.resolve()
    if not (reference / "protomotions").is_dir():
        parser.error(f"Reference core is unavailable: {reference / 'protomotions'}")
    rows = inventory(root, reference)
    current_paths = {reference_relative(row["path"]) for row in rows}
    reference_paths = {path.relative_to(reference).as_posix()
                             for folder in REFERENCE_SOURCE_DIRS for path in (reference / folder).rglob("*")
                             if path.is_file() and path.suffix in EXTENSIONS
                             and not any(part in {"__pycache__", ".venv", "build", "dist"} for part in path.parts)}
    reference_paths.update(path.name for path in reference.iterdir()
                           if path.is_file() and path.suffix in EXTENSIONS)
    reference_only = sorted(reference_paths - current_paths)
    by_path = {row["path"]: row for row in rows}
    lint_status = "unavailable"
    if shutil.which("ruff"):
        lint = subprocess.run(["ruff", "check", "--target-version", "py312", "--output-format", "json",
                               "--select", "F821,F822,F823,F811,E9,F63,F7,F82",
                               *(str(root / row["path"]) for row in rows if row["path"].endswith(".py"))],
                              cwd=root, capture_output=True, text=True)
        if lint.returncode not in (0, 1):
            raise RuntimeError(f"Ruff did not complete: {lint.stderr}")
        for finding in json.loads(lint.stdout):
            relative = Path(finding["filename"]).relative_to(root).as_posix()
            by_path[relative]["ruff_findings"].append({"code": finding["code"], "line": finding["location"]["row"], "message": finding["message"]})
        lint_status = "completed"
    summary = {"files": len(rows), "by_directory": dict(Counter(row["path"].split('/')[0] for row in rows)),
               "syntax": dict(Counter(row["syntax_status"] for row in rows)),
               "reference": dict(Counter(row["reference_status"] for row in rows)),
               "reference_only_files": len(reference_only),
               "ruff_status": lint_status, "ruff_findings": sum(len(row["ruff_findings"]) for row in rows)}
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "reference": str(reference), "target_python": "3.12", "scope": list(SOURCE_DIRS),
               "limitations": ["Static inventory and syntax checks are not proof of line-by-line semantic correctness.",
                               "Reference equality is not a requirement: the current ProtoMotions core has newer APIs.",
                               "FLUX/NavDP internals, model/data binaries and generated output are outside this migration audit."],
               "summary": summary, "files": rows, "reference_only_files": reference_only}
    (output / "source_inventory.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "source_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "lines", "sha256", "reference_status", "syntax_status", "ruff_findings"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: len(row[key]) if key == "ruff_findings" else row[key] for key in writer.fieldnames})
    print(json.dumps(summary, indent=2))
    return int(any(row["syntax_status"] == "failed" or row["ruff_findings"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
