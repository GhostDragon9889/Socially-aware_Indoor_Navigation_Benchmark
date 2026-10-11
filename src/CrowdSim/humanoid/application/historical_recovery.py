"""Read completed S3-R evidence using its original, byte-verified sources.

These snapshots establish historical provenance only. Execution plans must
still admit the live files. No archived source is imported or executed here.
"""
import json

from protomotions.repository import repository_root
from social_navigation.artifact_identity import artifact_path, same_artifact
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record

ROOT = repository_root(__file__)
SNAPSHOTS = {
    "src/CrowdSim/humanoid/runtime/social_observation_producer.py": (
        "docs/style_improvement_20261006/shared_social_native_seeded_preparation_v1/source_snapshot_v1/CrowdSim/humanoid/runtime/social_observation_producer.py",
        3733, "4ffaa5a4e9c2d1e9e717ccae33f1978cbeb4a991d91193461af87593750ad40c"),
    "scripts/audit_shared_social_popart.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/audit_shared_social_popart.py.txt",
        4276, "fa8719013f3dd6bd0c01049c3050d1830bc954e87ad8d627008be7628c9d2c78"),
    "scripts/audit_shared_social_recovery.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/audit_shared_social_recovery.py.txt",
        5020, "da6f4cdbb32c07e431a0b3f6fa87bb083cbb1548bcb611418c25eaa2f6fe5c10"),
    "src/CrowdSim/humanoid/amp_world.py": (
        "docs/style_improvement_20261006/shared_social_native_seeded_preparation_v1/source_snapshot_v1/CrowdSim/humanoid/amp_world.py",
        24834, "1b63a4c8ac7d4bf2f9ed523edc97b41dca89888cdd6c7dcb2234826d240d7f83"),
    "scripts/train_shared_social.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/train_shared_social.py.txt",
        4318, "a1247911b78cc9adce2ee67da42e2afefb43c785fcdbd8afe02e97b1b572cacc"),
    "scripts/evaluate_shared_social_recovery.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/evaluate_shared_social_recovery.py.txt",
        6934, "a579bc85c88dc2ac24be431905f38d9b011c5689c098362b995aa9639afff3a3"),
    "scripts/evaluate_shared_social_navigation.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/evaluate_shared_social_navigation.py.txt",
        9119, "b556de1a9c9c1a6b5984e1484c9a67f490f549f4346649746a84dd7ab9726558"),
    "scripts/evaluate_shared_social_tracking.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/evaluate_shared_social_tracking.py.txt",
        16532, "ff866433ce91feb2a006a137ed437ae7bab92b8b6b0c9daaf09f244521b3767c"),
    "tests/test_shared_social_recovery_evaluation_resume.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/test_shared_social_recovery_evaluation_resume.py.txt",
        3717, "9e97c7e6529cb9b39542ceec49899429a38801ee7ae0b839298675c83e20c423"),
    "src/CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py": (
        "docs/style_improvement_20261005/shared_social_native_raw_audit_preparation_v2/source_snapshot_v1/shared_social_native_raw_audit_preparation_v1/CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py",
        10059, "be868487365352f092128b7a1054ba814a1912e37a1bb211db7219abe2e3654a"),
    "src/CrowdSim/humanoid/runtime/social_navigation_lifecycle.py": (
        "docs/style_improvement_20261006/shared_social_native_seeded_preparation_v1/source_snapshot_v1/CrowdSim/humanoid/runtime/social_navigation_lifecycle.py",
        8956, "e7c283fbdef14d03deeea8748013fb38f413a75a9d78b039114a61991527db96"),
    "src/CrowdSim/humanoid/runtime/social_popart_inference.py": (
        "docs/style_improvement_20261006/shared_social_native_seeded_preparation_v1/source_snapshot_v1/CrowdSim/humanoid/runtime/social_popart_inference.py",
        3150, "d3b5724567acea05dcca7557133b73361553085ef4ebb61c3dfdda72c5ec1344"),
    "src/CrowdSim/humanoid/runtime/social_popart_training.py": (
        "docs/style_improvement_20261006/shared_social_native_seeded_preparation_v1/source_snapshot_v1/CrowdSim/humanoid/runtime/social_popart_training.py",
        5386, "66f8d37bd6e515f577c306f009d268111c860d847268626fdf01db0826152493"),
    "src/CrowdSim/humanoid/application/recovery_evaluation_admission.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/recovery_evaluation_admission.py.txt",
        11594, "e8d3a85385972f9e97aa7310525ef819942890ffab41f6ef3cdbc3dbac1209c5"),
    "tests/test_shared_social_recovery_evaluation.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/test_shared_social_recovery_evaluation.py.txt",
        6574, "14f5350bbccae36f928c2cc6757abbd87fae2427fc017507792995e97cf06e4c"),
    "scripts/resume_shared_social_recovery_evaluation.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/resume_shared_social_recovery_evaluation.py.txt",
        19446, "868372c988ff4c7267d3544219bfd3c98ee4e55c2b9c9c3fe95b7653a2551369"),
    "scripts/own_shared_social_recovery_evaluation.py": (
        "src/CrowdSim/humanoid/application/historical_recovery_sources/own_shared_social_recovery_evaluation.py.txt",
        9500, "e017f58e605ccd3be6c66ba4552bed2009d207596d426cadbc311e27b55ec818"),
}


def source_record(path):
    """Return an original source identity after verifying every snapshot byte."""
    path = artifact_path(path, root=ROOT)
    relative = str(path.relative_to(ROOT))
    if relative not in SNAPSHOTS:
        return record(path)
    snapshot, size, digest = SNAPSHOTS[relative]
    actual = record(ROOT / snapshot)
    if (actual["bytes"], actual["sha256"]) != (size, digest):
        raise ValueError("Historical recovery source snapshot changed: " + relative)
    return {"path": str(path), "bytes": size, "sha256": digest}


def verify_sources(items, *, historical=False, required_paths=None):
    if required_paths is not None and (
            type(items) is not list
            or [artifact_path(item["path"], root=ROOT) for item in items] != list(required_paths)):
        raise ValueError("Recovery source changed: require the complete ordered source/input inventory")
    for item in items:
        actual = source_record(item["path"]) if historical else record(artifact_path(item["path"], root=ROOT))
        if not same_artifact(actual, item, root=ROOT):
            raise ValueError("Recovery source changed: " + item["path"])


def plan_inventory(kind, *, field="sources"):
    """The original readonly registrations define mandatory inventory roles."""
    registrations = {
        "owner": ("shared_social_S3R_evaluation_v2", "e216ccd1124b515861a1ce9f2309eae32a4883579f0abe111b7b1c52c417b9fc"),
        "retry": ("shared_social_S3R_navigation_retry_v1", "7a1d979633b7522cbb45916ddbba174cbec56abec4f85e083a491e21b5e94f05"),
    }
    family, digest = registrations[kind]
    path = ROOT / "docs/style_improvement_20261008" / family / "owner_plan.json"
    if record(path)["sha256"] != digest or path.stat().st_mode & 0o222:
        raise ValueError("Recovery inventory registration changed")
    return [artifact_path(row["path"], root=ROOT) for row in json.loads(path.read_text())[field]]


def training_admission(receipt_path, *, stage=None):
    """Rebuild original metadata after full admission of the completed training."""
    from . import recovery_evaluation_admission as admission

    checkpoint, digest, accepted = admission.admit_training(receipt_path)
    accepted["optimizer_recovery"]["admission_source"] = source_record(admission.__file__)
    if stage is not None:
        from scripts import evaluate_shared_social_recovery as evaluation

        relative, sha = evaluation.PARENTS[stage]
        parent = source_record(ROOT / relative)
        if parent["sha256"] != sha:
            raise ValueError("Historical evaluation parent changed: " + stage)
        source_path = ROOT / (SNAPSHOTS[relative][0] if relative in SNAPSHOTS else relative)
        _, proof = evaluation.derive(source_path.read_text(), stage)
        binding = {"stage": stage, "parent": parent, "entry": source_record(evaluation.__file__),
                   "admission_source": source_record(admission.__file__), "derivation": proof}
        accepted["recovery_evaluation_entry"] = binding
    return checkpoint, digest, accepted
