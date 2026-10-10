"""Run staged review tests using in-process source aliases; no business merge."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import pytest
import CrowdSim
import scripts

DOC = Path(__file__).resolve().parent
ROOT = Path(CrowdSim.__file__).resolve().parents[1]
PROPOSAL = DOC / "proposed"


def source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PathsForDocsOnlyCopies:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            module = item.module
            if module.__name__ == "test_paper_scene_maps_probe":
                # The copied test must retain its original real-artifact root.
                module.__file__ = str(ROOT / "tests/test_paper_scene_maps_probe.py")
            if module.__name__ == "test_native_map_v7":
                module.ROOT = ROOT
            if module.__name__ == "test_native_map_audit_reader":
                module.ROOT = ROOT
                module.reader.ROOT = ROOT
                module.reader.DOC = ROOT / "docs/style_improvement_20261002/native_map_admission_v7"


def main():
    for relative in ("CrowdSim/humanoid/runtime/native_map_v8.py", "CrowdSim/humanoid/runtime/dynbench_scene.py", "CrowdSim/humanoid/runtime/native_grid_scene.py"):
        source(relative.removesuffix(".py").replace("/", "."), PROPOSAL / relative)
    scripts.probe_paper_scene_maps = source("scripts.probe_paper_scene_maps", PROPOSAL / "scripts/probe_paper_scene_maps.py")
    files = [
        PROPOSAL / "tests/test_paper_scene_maps_probe.py",
        PROPOSAL / "tests/test_paper_scene_maps_clock_v8.py",
        ROOT / "tests/test_dynbench_scene_provider.py",
        ROOT / "tests/test_native_grid_scene.py",
        PROPOSAL / "tests/test_native_map_v7.py",
        PROPOSAL / "tests/test_native_map_v8.py",
        PROPOSAL / "tests/test_native_map_audit_reader.py",
    ]
    return pytest.main([*map(str, files), "-q", "--tb=short", "--junitxml=" + str(DOC / "full_artifact_tests.xml")], plugins=[PathsForDocsOnlyCopies()])


if __name__ == "__main__":
    raise SystemExit(main())
