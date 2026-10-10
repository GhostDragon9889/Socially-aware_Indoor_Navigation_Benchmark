"""Representation-only fix around the immutable v1 native15 CPU auditor."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CORE = ROOT / "docs/style_improvement_20261003/state_prior_native_reader_preparation_v1"
CORE_FREEZE_SHA = "154c4eb9debc41a83a487e4549c73009dcc3c4f15ab3c36e85e0990129a2ac50"
CORE_REGISTRATION_SHA = "8c972122b3a498ae6a5b81c1086d42f9a425f7963c0bd9e5a1b23d0b00e84195"
FIRST_FAILURE_SHA = "9a796592fb8ab86e12b7bda9b3b1ea8cfed698dca2b20434d767a4f6b1ea1228"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def load_core():
    freeze_path = CORE / "source_freeze_v1.json"
    require(record(freeze_path)["sha256"] == CORE_FREEZE_SHA, "Exact immutable v1 core required")
    freeze = read(freeze_path)
    for row in freeze["files"] + [freeze["archive"]]:
        require(record(row["path"]) == row and Path(row["path"]).stat().st_mode & 0o222 == 0, "Immutable v1 source changed")
    registration_path = CORE / "native15_registration.json"
    require(record(registration_path)["sha256"] == CORE_REGISTRATION_SHA, "Exact actual15 v1 registration required")
    failure_path = HERE / "native15_first_actual_failure.log"
    require(record(failure_path)["sha256"] == FIRST_FAILURE_SHA, "Preserve first actual tuple/list failure")
    sys.path.insert(0, str(CORE))
    spec = importlib.util.spec_from_file_location("native15_immutable_v1_math_core", CORE / "audit_state_prior_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(sys.modules["native_evidence"].__file__).resolve() == CORE / "native_evidence.py", "No other native evidence helper fallback")
    return module


def canonical_proof(proof):
    """JSON representation only, preserving all labels, integers and values."""
    return json.loads(json.dumps(proof, allow_nan=False))


def registration():
    core = load_core()
    _, actual_proof = core.readers(15)
    registered = read(CORE / "native15_registration.json")
    require(actual_proof != registered["adapter_proof"], "Real tuple/list representation mismatch must be demonstrated")
    require(canonical_proof(actual_proof) == registered["adapter_proof"], "Every original proof value must remain exact")
    return {"schema": "public_state_prior.native15_JSON_proof_registration.v2", "status": "prepared_actual_native15_raw_audit_pending", "created_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "GPU_used": False, "SDK_started": False, "native_execution_audited": False, "representation_adapter": record(__file__), "immutable_core_freeze": record(CORE / "source_freeze_v1.json"), "immutable_core_registration": record(CORE / "native15_registration.json"), "first_actual_failure": record(HERE / "native15_first_actual_failure.log"), "actual_original_proof": canonical_proof(actual_proof), "representation_delta": "Only JSON list-of-tuples to list-of-lists in four origin-shape proof rows; values, compiled v1/v2/routes AST and raw math unchanged.", "source_count": 183}


def audit(registered):
    require(registered["schema"] == "public_state_prior.native15_JSON_proof_registration.v2" and registered["native_execution_audited"] is False and registered["representation_adapter"] == record(__file__) and registered["immutable_core_registration"] == record(CORE / "native15_registration.json") and registered["immutable_core_freeze"] == record(CORE / "source_freeze_v1.json") and registered["first_actual_failure"] == record(HERE / "native15_first_actual_failure.log"), "Exact frozen v2 representation adapter and immutable v1 sources required")
    core = load_core()
    original_readers = core.readers

    def readers_with_JSON_proof(actors):
        require(actors == 15, "This revision is bounded to15")
        native, original_proof = original_readers(actors)
        normalized = canonical_proof(original_proof)
        require(normalized == registered["actual_original_proof"], "Representation normalization cannot change original proof values")
        return native, normalized

    core.readers = readers_with_JSON_proof
    report = core.audit(15, read(CORE / "native15_registration.json"))
    report["representation_adapter"] = record(__file__)
    report["representation_revision_registration"] = registered
    report["first_actual_CPU_failure"] = registered["first_actual_failure"]
    report["immutable_math_core_freeze"] = registered["immutable_core_freeze"]
    report["compiled_v1_v2_routes_math_changed"] = False
    report["proof_tuple_list_difference_only"] = True
    require(report["execution_reader"] == record(CORE / "audit_state_prior_native.py"), "Original mathematics executor identity must remain explicit")
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "CPU revision imported model/SDK")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous evidence")
    require(args.register_only or args.registration is not None, "Actual stage requires frozen v2 registration")
    report = registration() if args.register_only else audit(read(args.registration))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "counts": report.get("counts"), "actual_native_audited": report.get("native_200s_artifacts_independently_audited", False)}))


if __name__ == "__main__":
    main()
