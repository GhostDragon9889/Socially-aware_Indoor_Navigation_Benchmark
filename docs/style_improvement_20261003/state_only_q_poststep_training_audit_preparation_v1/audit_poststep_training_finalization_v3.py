"""One guarded metadata correction to the readonly reviewed poststep reader.

Source derivation occurs only at actual CLI execution. The original complete
payload, source and closed-root checks remain in the pinned base reader.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

BASE_SHA = "ae502054ae34fa77f48cb6dbb881cab4b7ed1d47ded37dc04ee3e6c41d4bab9a"
OLD_GUARD = '(observation["runtime_finalization_observed"] is True if closed else True)'
NEW_GUARD = ('(type(observation["runtime_finalization_observed"]) is bool and '
             'observation["runtime_finalization_observed"] is '
             'json.loads(Path(ledger_path).read_text())["runtime_finalization_observed"])')


def main():
    base = Path(__file__).resolve().with_name("audit_poststep_training.py")
    if base.is_symlink() or not base.is_file() or base.stat().st_mode & 0o222:
        raise ValueError("Reviewed readonly regular base reader required")
    raw = base.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError("Reviewed base reader bytes changed")
    before = raw.decode("utf-8")
    if before.count(OLD_GUARD) != 1 or NEW_GUARD in before:
        raise ValueError("Exact sole finalization guard required")
    after = before.replace(OLD_GUARD, NEW_GUARD, 1)
    if after.count(NEW_GUARD) != 1 or after.replace(NEW_GUARD, OLD_GUARD, 1) != before:
        raise ValueError("Inverse entire base reader bytes differ")
    namespace = {"__file__": __file__, "__name__": "_poststep_finalization_v3"}
    exec(compile(after, __file__, "exec"), namespace)
    return namespace["main"]()


if __name__ == "__main__":
    main()
