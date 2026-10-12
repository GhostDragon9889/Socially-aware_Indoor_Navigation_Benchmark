"""Inventory every owned Python statement, including never-executed modules.

Coverage percentages describe execution, not semantic correctness. Generated
code with a virtual filename is listed separately and receives no guessed
credit against its original source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def inventory(root: Path, *, ignore_isaacgym: bool = False) -> dict:
    """Use Git's NUL-delimited inventory and include pending owned source files."""
    from coverage.parser import PythonParser

    root = root.resolve()
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    records, aliases = {}, []
    for raw in sorted(set(output.split(b"\0")) - {b""}):
        relative = Path(raw.decode("utf-8", errors="surrogateescape"))
        if relative.suffix != ".py" or "tests" in relative.parts:
            continue
        path = root / relative
        canonical = path.resolve()
        if not canonical.is_relative_to(root):
            raise ValueError("Owned source link escaped checkout: " + str(relative))
        if path.is_symlink():
            aliases.append(
                {
                    "path": relative.as_posix(),
                    "target": canonical.relative_to(root).as_posix(),
                }
            )
        if not canonical.is_file():
            raise ValueError("Owned Python source is missing: " + str(relative))
        name = canonical.relative_to(root).as_posix()
        if name in records:
            continue
        parser = PythonParser(filename=str(canonical), exclude=None)
        parser.parse_source()
        content = canonical.read_bytes()
        records[name] = {
            "path": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "source_lines": len(content.splitlines()),
            "statements": sorted(parser.statements),
        }
    excluded = []
    if ignore_isaacgym:
        for name in list(records):
            if Path(name).parts[:4] == ("src", "protomotions", "simulator", "isaacgym"):
                excluded.append(
                    {**records.pop(name), "reason": "explicit_isaacgym_waiver"}
                )
    return {
        "schema": "social_navigation.statement_inventory.v1",
        "files": list(records.values()),
        "aliases": aliases,
        "source_files": len(records),
        "statement_count": sum(len(row["statements"]) for row in records.values()),
        "ignored_backends": ["isaacgym"] if ignore_isaacgym else [],
        "excluded_files": excluded,
        "scope": "All tracked and pending non-test Python files; no pragma or main-guard exclusions. "
        + (
            "Dedicated IsaacGym backend explicitly waived."
            if ignore_isaacgym
            else "No backend exclusions."
        ),
    }


def report(
    root: Path,
    data_file: Path,
    *,
    baseline: dict | None = None,
    ignore_isaacgym: bool = False,
) -> dict:
    """Report real traced statements and retain all zero-hit files in the total."""
    from coverage import CoverageData
    from coverage.parser import PythonParser

    root = root.resolve()
    if not data_file.is_file():
        raise FileNotFoundError("Combined coverage data is missing: " + str(data_file))
    data = CoverageData(basename=str(data_file))
    data.read()
    current = inventory(root, ignore_isaacgym=ignore_isaacgym)
    records = {row["path"]: row for row in current["files"]}
    measured, virtual = {}, []
    for filename in sorted(data.measured_files()):
        path = Path(filename)
        resolved = (path if path.is_absolute() else root / path).resolve()
        if resolved.is_relative_to(root):
            relative = resolved.relative_to(root).as_posix()
            if relative in records:
                measured.setdefault(relative, set()).update(data.lines(filename) or [])
            elif ".py:" in relative:
                virtual.append(
                    {
                        "filename": relative,
                        "executed_line_count": len(data.lines(filename) or []),
                        "credited_to_source": False,
                    }
                )
    rows = []
    for source in current["files"]:
        statements = set(source["statements"])
        parser = PythonParser(filename=str(root / source["path"]), exclude=None)
        parser.parse_source()
        # CPython traces continuation lines of multiline statements. Use the
        # same parser normalization as coverage's own analysis, retaining all
        # original statements and all zero-hit modules in the denominator.
        traced = parser.translate_lines(measured.get(source["path"], set()))
        covered = sorted(statements & traced)
        missing = sorted(statements - set(covered))
        rows.append(
            {
                **source,
                "covered_statements": covered,
                "missing_statements": missing,
                "statement_percent": 100 * len(covered) / len(statements)
                if statements
                else 100.0,
            }
        )
    changed = []
    if baseline is not None:
        original = {row["path"]: row["sha256"] for row in baseline["files"]}
        actual = {row["path"]: row["sha256"] for row in current["files"]}
        changed = sorted(
            path
            for path in original.keys() | actual.keys()
            if original.get(path) != actual.get(path)
        )
    total = current["statement_count"]
    covered = sum(len(row["covered_statements"]) for row in rows)
    same_scope = (
        baseline is not None
        and baseline.get("ignored_backends", []) == current["ignored_backends"]
    )
    # Branch data remains separate so it cannot mask missing statements.
    return {
        "schema": "social_navigation.statement_coverage.v1",
        "files": rows,
        "source_files": current["source_files"],
        "statement_count": total,
        "covered_statements": covered,
        "missing_statements": total - covered,
        "statement_percent": 100 * covered / total if total else 100.0,
        "zero_hit_files": [
            row["path"]
            for row in rows
            if row["statements"] and not row["covered_statements"]
        ],
        "virtual_code": virtual,
        "branch_data_collected": data.has_arcs(),
        "source_stability_verified": same_scope and not changed,
        "source_changes": changed,
        "scope": current["scope"],
        "scope_stability_verified": same_scope,
        "ignored_backends": current["ignored_backends"],
        "excluded_files": current["excluded_files"],
        "gate_passed": bool(total)
        and total == covered
        and baseline is not None
        and same_scope
        and not changed,
    }


def main(argv=None) -> int:
    import argparse
    from protomotions.repository import repository_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    parser.add_argument("--data", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ignore-isaacgym",
        action="store_true",
        help="Explicitly waive only the dedicated IsaacGym backend",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every statement and source identity is verified",
    )
    args = parser.parse_args(argv)
    if args.require_complete and args.data is None:
        parser.error("--require-complete requires --data")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    result = (
        report(
            args.root,
            args.data,
            baseline=baseline,
            ignore_isaacgym=args.ignore_isaacgym,
        )
        if args.data
        else inventory(args.root, ignore_isaacgym=args.ignore_isaacgym)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"files", "virtual_code", "zero_hit_files", "aliases"}
            },
            indent=2,
        )
    )
    return int(args.require_complete and not result["gate_passed"])
