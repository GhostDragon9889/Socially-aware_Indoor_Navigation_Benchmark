"""Explicit current CPU reader provenance, separate from sealed SDK evidence.

Only two dependencies are migrated. The N4 collector is byte-identical after
undoing the repository lookup. The lifecycle file has an exact current version
and its only consumed CPU function, record(), retains every historical byte.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from protomotions.repository import repository_root


LIFECYCLE_HISTORICAL_SHA = "991b6107f113d961e9b2cccca1efdb8bbc9d1d484188c1b3b716e4cb11e3e15e"
LIFECYCLE_CURRENT_SHA = "a2c6ed27c3485ebf10e8ab1a7faf2f92be0de8a8a113990ac24260532b5544a7"
RECORD_SHA = "3f3c33da7eadf98b5af61c4a7545a0aa8c6967cc903e2be12922bc72202ad244"
INITIAL_SHA = "3395ebc1e4de385788fec738f49994af2887bb1b0af998363608c8625e17022d"
ROOT = repository_root(__file__)
SNAPSHOT = ROOT / "resources/runtime_sources/style_improvement_20261006/shared_social_native_seeded_admission_preparation_v1/source_snapshot_v1"
READER_VERSIONS = {
    "native_navigation_audit": (
        "f80f005bb2796bf37168281f243ab462ee207223609b735656a41b7febe9fe8b",
        "0290676ce86f64e0725d3316af59fd5572259e53aa47e6c29e33adbf19dbe186"),
    "native_navigation_style": (
        "f0ec149c97e10bc7511fdd7775281d8ee1f28d2b35aaf6e9f73bb7386fd7ff54",
        "9cd1e3f15f325f0d8885398f433e42c993a648fc6928701db7d3e66805a6f47f"),
    "native_navigation_seeded_audit": (
        "b405e4926915abc4f8165d2ff996d15227e218e12d9209555aca349d1e5d7bbd",
        "c7676f1576672f1b80f178f8e4dff7f37b08fc813e578035755b65319b190605"),
}


def verify_reader(path, historical_sha):
    """Admit reviewed reader versions and prove their measurement code unchanged."""
    path = Path(path).resolve()
    if path == ROOT / "scripts/evaluate_shared_social_navigation_style.py":
        expected = "394ce99e1eae7d89c886adc5cbcd6cb292b58bee3758609522a7b72270e6c10a"
        if historical_sha != expected:
            raise ValueError("Unregistered CPU reader migration: " + str(path))
        source = path.read_text()
        replacements = (
            ('    from social_navigation.artifact_identity import artifact_path, same_artifact\n'
             '    path = artifact_path(identity["path"], root=ROOT)\n'
             '    require(same_artifact(record(path), identity, root=ROOT), "Recorded input changed: " + identity["path"])\n'
             '    return json.loads(path.read_text())\n',
             '    require(record(identity["path"]) == identity, "Recorded input changed: " + identity["path"])\n'
             '    return json.loads(Path(identity["path"]).read_text())\n'),
            ('    from social_navigation.artifact_identity import artifact_path\n'
             '    manifest = artifact_path(artifact["manifest"]["path"], root=ROOT)\n'
             '    identities.append(record(manifest))\n', ''),
            ('    body, rotation, hashes, provenance = tool.load_reference(manifest, artifact["selection"])\n',
             '    body, rotation, hashes, provenance = tool.load_reference(Path(artifact["manifest"]["path"]), artifact["selection"])\n'),
        )
        restored = source
        for current, historical in replacements:
            if restored.count(current) != 1:
                raise ValueError("Frozen style location adapter changed")
            restored = restored.replace(current, historical, 1)
        if hashlib.sha256(restored.encode()).hexdigest() != expected:
            raise ValueError("Frozen style migration changed historical metric or admission bytes")
        return {"version": "CPU_frozen_style_location.sim61.v1", "historical_sha256": expected,
                "current_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "whole_source_location_inverse": True, "all_metric_function_bytes_unchanged": True,
                "reference_manifest_rechecked_after_measurement": True, "SDK_execution_verified": False}
    if path == ROOT / "scripts/evaluate_shared_social_style.py":
        expected = "d3296011b0e08286ae109daea6a518c95f5f9fa9f24a474a20555cd002baa47c"
        if historical_sha != expected:
            raise ValueError("Unregistered CPU reader migration: " + str(path))
        source = path.read_text()
        restored = source.replace("resources/runtime_sources/", "docs/")
        replacements = (
            ("6e9ed2fa9949336acf2c7d01f0ede77f952832bbf770d1c1800c4864500c91c2", "c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564"),
            ("ba3fa07097bd983090b7bea1fd59baa05292b9a80fe6e02cef6f7f3897d4a37d", "208ff775d765b7d046e7e7e8c331dbb2b8d3bfdc4c1567c1df499a584af30bfa"),
            ("875f72ec99cb2ab8c3c5fd5f796495fe634b8837f25cee19f7ee104dd4cbf512", "9478afe6b16540f7624867e1d736f30f78399c84177c12d6c9b616bc496c3780"),
            ('    from social_navigation.artifact_identity import artifact_path\n'
             '    body, rotation, hashes, _ = tool.load_reference(artifact_path(artifact["manifest"]["path"]), artifact["selection"])\n',
             '    body, rotation, hashes, _ = tool.load_reference(Path(artifact["manifest"]["path"]), artifact["selection"])\n'),
        )
        for current, historical in replacements:
            if restored.count(current) != 1:
                raise ValueError("Frozen style source relocation seam changed")
            restored = restored.replace(current, historical, 1)
        if hashlib.sha256(restored.encode()).hexdigest() != expected:
            raise ValueError("Frozen style source relocation changed historical metric bytes")
        return {"version": "CPU_style_resources.sim61.v1", "historical_sha256": expected,
                "current_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "whole_source_location_inverse": True, "all_metric_function_bytes_unchanged": True,
                "SDK_execution_verified": False}
    versions = READER_VERSIONS.get(path.stem)
    relative = Path("CrowdSim/humanoid/analysis") / path.name
    if versions is None or path != ROOT / "src" / relative or historical_sha != versions[0]:
        raise ValueError("Unregistered CPU reader migration: " + str(path))
    current, historical = path.read_text(), (SNAPSHOT / relative).read_text()
    if (hashlib.sha256(current.encode()).hexdigest() != versions[1]
            or hashlib.sha256(historical.encode()).hexdigest() != versions[0]):
        raise ValueError("Registered CPU reader source changed: " + str(path))

    def functions(source, *, migrated):
        tree = ast.parse(source)
        # This selector only chooses the proven current dependency; it does
        # not alter the raw kernel's duration/count transformations or math.
        if migrated and path.stem == "native_navigation_audit":
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "CURRENT_HIGH_LEVEL_SHA":
                    node.id = "HIGH_LEVEL_SHA"
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name != "source_bindings"}

    before, after = functions(historical, migrated=False), functions(current, migrated=True)
    if before != after:
        raise ValueError("CPU reader migration changed a measurement/admission function")
    return {"version": "CPU_native_reader_sources.sim61.v2", "historical_sha256": versions[0],
            "current_sha256": versions[1], "unchanged_functions": sorted(before),
            "measurement_AST_unchanged": True, "SDK_execution_verified": False}


def verify_dependency(path, historical_sha):
    path = Path(path)
    source = path.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    proof = {"version": "CPU_reader_dependencies.sim61.v2", "historical_sha256": historical_sha,
             "current_sha256": digest, "SDK_execution_verified": False}
    if (path.resolve() == ROOT / "scripts/verify_shared_social_evidence_admission.py"
            and historical_sha == "f2cf03483f9bab988d9fb293093c644ef97b7cfeda32458e9ba0b2a93c40600a"):
        if digest != "7c173e34e2baf48527658a88c8276d041b542cb63c1efe45783ee1a726f90101":
            raise ValueError("Unreviewed portable metric source")
        oracle = ROOT / "resources/runtime_sources/oracles/evidence_admission.py.txt"
        baseline = oracle.read_text()
        if hashlib.sha256(baseline.encode()).hexdigest() != historical_sha:
            raise ValueError("Metric source oracle changed")
        def normalized_functions(text):
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    node.value = node.value.replace("resources/runtime_sources/", "docs/")
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name != "_artifact"}
        restored = source.replace(
            '    from social_navigation.artifact_identity import artifact_path\n'
            '    original.require(artifact_path(value["path"], root=ROOT).resolve() == (ROOT / relative).resolve()\n',
            '    original.require(Path(value["path"]).resolve() == ROOT / relative\n', 1)
        if normalized_functions(restored) != normalized_functions(baseline):
            raise ValueError("Portable path migration changed a metric/admission function")
        return {**proof, "artifact_location_adapter_exact_inverse": True,
                "all_metric_function_bytes_unchanged": True,
                "location_normalized_function_AST_verified": True}
    if (path.resolve() == ROOT / "scripts/compare_shared_social_evidence.py"
            and historical_sha == "76e589b05101480ac21ae229e7c8ecdefde049073670deaddb88a4306ed77a35"):
        current = ('    from social_navigation.artifact_identity import artifact_path, same_artifact\n'
                   '    path = artifact_path(identity["path"])\n'
                   '    if not same_artifact(record(path), identity):\n')
        original = '    if record(identity["path"]) != identity:\n'
        if source.count(current) == 1 and source.count('    return read(path)\n') == 1:
            restored = source.replace(current, original, 1).replace(
                '    return read(path)\n', '    return read(identity["path"])\n', 1)
            if hashlib.sha256(restored.encode()).hexdigest() == historical_sha:
                return {**proof, "artifact_location_adapter_exact_inverse": True,
                        "all_metric_function_bytes_unchanged": True}
    if path.name == "paired_initial.py" and historical_sha == INITIAL_SHA:
        from .native_paired_initial import restore_core_location
        restore_core_location(source)
        return {**proof, "whole_source_location_inverse": True}
    if (path.name == "social_evaluation_lifecycle.py"
            and historical_sha == LIFECYCLE_HISTORICAL_SHA and digest == LIFECYCLE_CURRENT_SHA):
        functions = [node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == "record"]
        if len(functions) == 1:
            selected = ast.get_source_segment(source, functions[0])
            if hashlib.sha256(selected.encode()).hexdigest() == RECORD_SHA:
                return {**proof, "consumed_CPU_function": "record", "exact_function_bytes_unchanged": True}
    raise ValueError("Pinned transitive CPU reader dependency changed: " + str(path))
