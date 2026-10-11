"""CPU v7 evidence reader after the registered checkout relocation.

The archived implementation and archive bytes stay immutable. This explicit
adapter changes only artifact location comparisons and tar member lookup.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from protomotions.repository import repository_root

ROOT = repository_root(__file__)
DOC = ROOT / "docs/style_improvement_20261002/native_map_admission_v7"
PARENT = DOC / "audit_actual_four_scenes.py"
PARENT_SHA = "bab805d1389f8d1d60e126506f6686ba4a8cf67db2da5db1912e131f1a2619b0"


def derive_reader(source):
    if hashlib.sha256(source.encode()).hexdigest() != PARENT_SHA:
        raise ValueError("Frozen v7 audit implementation changed")
    replacements = (
        ('ROOT = Path(__file__).resolve().parents[3]\nDOC = Path(__file__).resolve().parent\n',
         'from social_navigation.artifact_identity import artifact_path, same_artifact\n'),
        ('if file_record(record["path"]) != record:',
         'if not same_artifact(file_record(artifact_path(record["path"])), record):'),
        ('tarfile.open(freeze["source_archive"]["path"], "r:gz")',
         'tarfile.open(artifact_path(freeze["source_archive"]["path"]), "r:gz")'),
        ('member = str(Path(record["path"]).relative_to(ROOT))',
         'member = str(artifact_path(record["path"]).relative_to(ROOT))\n'
         '            if member.startswith("src/"):\n'
         '                member = member.removeprefix("src/")'),
        ('[file_record(record["path"]) for record in freeze["source_files"]]',
         '[file_record(artifact_path(record["path"])) for record in freeze["source_files"]]'),
        ('np.load(record["path"], allow_pickle=False)',
         'np.load(artifact_path(record["path"]), allow_pickle=False)'),
        ('"reader": file_record(__file__)}',
         '"reader": file_record(__file__), "historical_reader": file_record(PARENT),\n'
         '            "reader_migration": dict(DERIVATION)}'),
    )
    derived = source
    counts = []
    for before, after in replacements:
        count = derived.count(before)
        expected = 2 if before.startswith('if file_record') else 1
        if count != expected:
            raise ValueError("Ambiguous v7 artifact relocation")
        counts.append(count)
        derived = derived.replace(before, after)
    inverse = derived
    for (before, after), count in zip(reversed(replacements), reversed(counts)):
        if inverse.count(after) != count:
            raise ValueError("Ambiguous v7 artifact relocation inverse")
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise ValueError("v7 artifact relocation does not recover the whole source")
    return derived, {"version": "native_v7_CPU_reader.relocated.v2",
        "historical_sha256": PARENT_SHA,
        "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
        "exact_inverse": True, "SDK_executed": False}


_source, DERIVATION = derive_reader(PARENT.read_text())
exec(compile(_source, str(PARENT) + ":relocated", "exec"), globals())
