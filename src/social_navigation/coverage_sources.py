"""Attribute virtual Python frames only when their original code is identical.

Filename spelling alone is insufficient. Function bytecode, constants, names,
flags, exception/line tables and original line numbers must all match a code
object compiled from the SHA256-bound source inventory. Changed functions stay
under their virtual filename and receive no original-source execution credit.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import marshal
import os
from pathlib import Path
from types import CodeType

from coverage import CoveragePlugin, FileTracer


def code_identity(code: CodeType) -> bytes:
    constants = tuple(
        normalize_code(value) if isinstance(value, CodeType) else value
        for value in code.co_consts
    )
    return hashlib.sha256(
        marshal.dumps(code.replace(co_filename="", co_consts=constants))
    ).digest()


def normalize_code(code: CodeType) -> CodeType:
    return code.replace(
        co_filename="",
        co_consts=tuple(
            normalize_code(value) if isinstance(value, CodeType) else value
            for value in code.co_consts
        ),
    )


def compiled_identities(code: CodeType) -> set[bytes]:
    result = {code_identity(code)}
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            result.update(compiled_identities(constant))
    return result


class SourceTracer(FileTracer):
    def __init__(self, plugin, filename, original, *, virtual):
        self.plugin, self.filename, self.original = plugin, filename, original
        self.virtual = virtual
        self.decisions = {}

    def source_filename(self):
        return str(self.original)

    def has_dynamic_source_filename(self):
        return self.virtual

    def dynamic_source_filename(self, filename, frame):
        code = frame.f_code
        if code not in self.decisions:
            identity = code_identity(code)
            matched = identity in self.plugin.identities(self.original)
            self.decisions[code] = str(self.original) if matched else self.filename
            if matched:
                self.plugin.proofs.add(
                    (
                        self.filename,
                        str(self.original),
                        code.co_qualname,
                        code.co_firstlineno,
                        identity.hex(),
                    )
                )
        return self.decisions[code]

    def line_number_range(self, frame):
        return frame.f_lineno, frame.f_lineno


class VerifiedSources(CoveragePlugin):
    def __init__(self, root: Path, baseline: Path, proof_dir: Path):
        self.root = root.resolve()
        inventory = json.loads(baseline.read_text())
        self.sources = {self.root / row["path"]: row for row in inventory["files"]}
        if any(not path.resolve().is_relative_to(self.root) for path in self.sources):
            raise ValueError("Coverage source inventory escapes the checkout")
        self.reference = {}
        self.proofs = set()
        self.proof_dir = proof_dir
        atexit.register(self.save_proofs)

    def identities(self, path):
        if path not in self.reference:
            content = path.read_bytes()
            expected = self.sources[path]
            if (
                len(content) != expected["bytes"]
                or hashlib.sha256(content).hexdigest() != expected["sha256"]
            ):
                self.reference[path] = set()
            else:
                original = compile(content, str(path), "exec", dont_inherit=True)
                self.reference[path] = compiled_identities(original)
        return self.reference[path]

    def file_tracer(self, filename):
        path = Path(filename).resolve()
        if path in self.sources:
            # Use the same file-tracer identity for direct and proven virtual
            # execution, avoiding conflicting tracer metadata when combining.
            return SourceTracer(self, str(path), path, virtual=False)
        if ".py:" in filename:
            original = Path(filename.split(".py:", 1)[0] + ".py").resolve()
            if original in self.sources:
                return SourceTracer(self, str(path), original, virtual=True)
        return None

    def file_reporter(self, filename):
        return "python"

    def save_proofs(self):
        self.proof_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "virtual_filename": virtual,
                "source": source,
                "source_sha256": self.sources[Path(source)]["sha256"],
                "qualified_name": name,
                "first_line": line,
                "code_sha256": identity,
            }
            for virtual, source, name, line, identity in sorted(self.proofs)
        ]
        (self.proof_dir / (str(os.getpid()) + ".json")).write_text(
            json.dumps(
                {
                    "schema": "social_navigation.verified_virtual_frames.v1",
                    "frames": rows,
                },
                indent=2,
            )
            + "\n"
        )


def coverage_init(registry, options):
    registry.add_file_tracer(
        VerifiedSources(
            Path(options["root"]), Path(options["baseline"]), Path(options["proof_dir"])
        )
    )
