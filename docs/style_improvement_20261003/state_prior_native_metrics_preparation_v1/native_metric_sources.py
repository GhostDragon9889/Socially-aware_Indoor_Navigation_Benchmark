"""Source admission for CPU mathematics; native capture admission is separate."""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
PREFIXES = {"CrowdSim", "protomotions", "scripts", "crowdsim_ppo"}
AUDITED = {"raw_sensor_source_contracts_passed_bounded_zero_fall_observed",
           "raw_sensor_source_contracts_passed_physical_stability_failed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular artifact required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(row):
    require(record(row["path"]) == row, "Source identity changed: " + row["path"])


def validate_engine(catalog):
    for row in catalog["sources"].values():
        exact(row)
    source = json.loads(Path(catalog["sources"]["production_V11"]["path"]).read_text())
    require(source["source_count"] == len(source["files"]) == 183, "Exact production V11 required")
    allowed = {}
    for row in source["files"] + catalog["engine_extra_sources"]:
        exact(row)
        allowed[row["path"]] = row
    exact(source["archive"])
    freeze = json.loads(Path(catalog["sources"]["native_reader_freeze"]["path"]).read_text())
    for row in freeze["files"]:
        exact(row)
        require(not Path(row["path"]).stat().st_mode & 0o222, "Frozen native reader must be readonly")
    exact(freeze["archive"])
    return allowed


def validate_raw_contract(audit, actors, expected_count, schema):
    require(audit.get("schema") == schema and audit.get("status") in AUDITED,
            "A complete independent native raw/source audit is required")
    require(audit.get("native_200s_artifacts_independently_audited") is True,
            "Actual native artifact audit required")
    counts = audit.get("counts", {})
    require(counts.get("controls") == 6000 and counts.get("actors") == actors
            and counts.get("duration_s") == 200.0
            and counts.get("shared_physics_substeps") == 24000,
            "Actual complete N-person 200s native capture required")
    epoch = audit.get("source_epoch", {})
    require(epoch.get("source_count") == expected_count
            and epoch.get("all_archived_source_bytes_verified") is True,
            "Each native run must retain its own source epoch")
    require(type(audit.get("checks")) is list and bool(audit["checks"]), "Raw audit checks required")


def verify_archived_epoch(epoch):
    exact(epoch["manifest"])
    exact(epoch["archive"])
    source = json.loads(Path(epoch["manifest"]["path"]).read_text())
    require(source["source_count"] == len(source["files"]) == epoch["source_count"],
            "Historical source count changed")
    with tarfile.open(epoch["archive"]["path"], "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        require(set(members) == {row["path"] for row in source["files"]},
                "Historical source archive file set differs")
        for row in source["files"]:
            stream = archive.extractfile(members[row["path"]])
            value = stream.read()
            require(len(value) == row["bytes"] and hashlib.sha256(value).hexdigest() == row["sha256"],
                    "Historical archived source bytes differ: " + row["path"])


def admit_run(catalog, row):
    """Read completed evidence before any physical arrays, reference or metric."""
    audit_path = Path(row["raw_audit"])
    if row["kind"] == "new":
        bridge = load("state_prior_native_evidence", catalog["sources"]["native_evidence"]["path"])
        registration = json.loads(Path(row["registration"]).read_text())
        source = bridge.admit_closed_native(row["actors"], registration)
        require(not audit_path.stat().st_mode & 0o222, "Completed new native raw audit must be readonly")
        audit = json.loads(audit_path.read_text())
        validate_raw_contract(audit, row["actors"], 183, "public_state_prior.native200s_independent_CPU_audit.v1")
        require(audit.get("actual_source_admission") == source
                and audit.get("registration") == registration
                and audit.get("execution_reader") == catalog["sources"]["native_reader"],
                "Completed raw audit must bind the same actual native source and frozen reader")
        require(audit["source_epoch"]["manifest"] == source["production_source"]["manifest"]
                and audit["source_epoch"]["archive"] == source["production_source"]["archive"],
                "Actual raw source epoch must equal native production evidence")
    else:
        exact(row["raw_audit_record"])
        audit = json.loads(audit_path.read_text())
        validate_raw_contract(audit, row["actors"], row["source_count"],
                              f"native_Warehouse{row['actors']}_200s_independent_CPU_audit.v1")
        verify_archived_epoch(audit["source_epoch"])
        source = {"historical_native_raw_proof": row["raw_audit_record"],
                  "historical_source_epoch": audit["source_epoch"],
                  "current_production_used_as_historical_runtime_proof": False}
    require(Path(audit["run"]).resolve() == Path(row["input"]).resolve(), "Raw audit identifies another run")
    require(set(audit["artifacts"]) == {"result.json", "physical_states.npz", "ground_support.npz",
                                      "reference_plans.npz", "deployment_entry.json"},
            "Five complete native artifact records required")
    for name, item in audit["artifacts"].items():
        require(Path(item["path"]).absolute() == Path(row["input"]) / name,
                "Native artifact alias differs")
        exact(item)
    return {"raw_audit": record(audit_path), "raw_status": audit["status"],
            "capture_artifacts": audit["artifacts"], "native_source_admission": source,
            "runtime_source_count": audit["source_epoch"]["source_count"],
            "style_success_is_quality_success": False, "default_promotion": False}


class OwnedLoader(importlib.abc.Loader):
    def __init__(self, delegate, ledger):
        self.delegate, self.ledger = delegate, ledger

    def create_module(self, spec):
        method = getattr(self.delegate, "create_module", None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        self.ledger.observe(module.__name__, module.__file__)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class OwnedGuard(importlib.abc.MetaPathFinder):
    def __init__(self, ledger):
        self.ledger = ledger

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in PREFIXES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(ROOT), str(ROOT / "src")] if path is None else path)
        require(spec is not None, "No unregistered owned module fallback: " + fullname)
        if spec.origin not in (None, "built-in", "frozen"):
            self.ledger.verify_path(spec.origin)
        require(all(Path(p).resolve().is_relative_to(ROOT) for p in spec.submodule_search_locations or []),
                "Owned namespace escaped production")
        if spec.loader is not None:
            spec.loader = OwnedLoader(spec.loader, self.ledger)
        return spec


class CPULedger:
    def __init__(self, path, allowed, catalog_record):
        self.path, self.allowed = Path(path), allowed
        require(not self.path.exists(), "Fresh CPU source ledger required")
        self.rows = {}
        self.catalog_record = catalog_record

    def verify_path(self, path):
        path = str(Path(path).absolute())
        require(path in self.allowed, "Owned CPU mathematics source is not registered: " + path)
        exact(self.allowed[path])

    def observe(self, name, path):
        self.verify_path(path)
        self.rows[name] = {"module": name, **record(path), "execution_observed": True}

    def activate(self):
        sys.path[:0] = [str(ROOT), str(ROOT / "src")]
        sys.dont_write_bytecode = True
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] in PREFIXES and getattr(module, "__file__", None):
                self.observe(name, module.__file__)
        sys.meta_path.insert(0, OwnedGuard(self))

    def finish(self, status):
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] in PREFIXES and getattr(module, "__file__", None):
                self.observe(name, module.__file__)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x") as stream:
            json.dump({"schema": "public_state_prior.native_metric_CPU_owned_sources.v1",
                       "status": status, "runtime_finalization_observed": True,
                       "CPU_only": True, "GPU_used": False, "math_source_epoch": 183,
                       "catalog": self.catalog_record, "owned_modules": sorted(self.rows.values(), key=lambda r: r["module"]),
                       "native_GPU_import_ledger_claimed": False}, stream, indent=2)
            stream.write("\n")
