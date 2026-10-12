"""SDK lookup must reject ambiguous providers and preserve portable identities."""

import ctypes
import json
from types import SimpleNamespace

import pytest

from social_navigation import sdk_paths as paths


def lab(tmp_path, name="SDK"):
    root = tmp_path / name
    package = root / "source/isaaclab/isaaclab"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    return root, package


def test_binding_resolves_sdk_when_module_is_not_importable(tmp_path, monkeypatch):
    root, _ = lab(tmp_path)
    project = tmp_path / "project"
    (project / ".runtime").mkdir(parents=True)
    (project / ".runtime/external_sdk.json").write_text(
        json.dumps({"lab_root": str(root)})
    )
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    monkeypatch.setattr(paths, "repository_root", lambda *a: project)
    monkeypatch.setattr(paths.util, "find_spec", lambda name: None)
    assert paths.lab_root() == root
    (root / "source/isaaclab/isaaclab/__init__.py").unlink()
    with pytest.raises(RuntimeError, match="not found"):
        paths.lab_root()


@pytest.mark.parametrize(
    "locations", [[], ["unknown/path"], ["/shallow"], ["duplicated", "foreign"]]
)
def test_invalid_import_locations_cannot_select_a_different_sdk(
    tmp_path, monkeypatch, locations
):
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    monkeypatch.setattr(paths, "repository_root", lambda *a: tmp_path)
    monkeypatch.setattr(
        paths.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=locations),
    )
    with pytest.raises(RuntimeError, match="not found|does not identify"):
        paths.lab_root()


def test_multiple_real_sdk_providers_are_ambiguous(tmp_path, monkeypatch):
    _, first = lab(tmp_path, "one")
    _, second = lab(tmp_path, "two")
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    monkeypatch.setattr(
        paths.util,
        "find_spec",
        lambda name: SimpleNamespace(
            submodule_search_locations=[str(first), str(second)]
        ),
    )
    with pytest.raises(RuntimeError, match="one source checkout"):
        paths.lab_root()


def test_all_namespaces_reject_parent_traversal_before_lookup(tmp_path):
    for value in ["project:../secret", "lab:source/../../secret", "sim:../secret"]:
        with pytest.raises(ValueError, match="parent traversal"):
            paths.relocate_sdk_path(value, root=tmp_path)


def test_legacy_sdk_paths_resolve_only_by_complete_directory_components(
    tmp_path, monkeypatch
):
    root, _ = lab(tmp_path)
    sim = tmp_path / "installed_sim"
    sim.mkdir()
    monkeypatch.setattr(paths, "lab_root", lambda: root)
    monkeypatch.setattr(paths, "sim_root", lambda: sim)
    assert paths.relocate_sdk_path(
        "/old/IsaacLab/.venv/lib/python3.12/site-packages/isaacsim/config",
        root=tmp_path,
    ) == str(sim / "config")
    assert paths.relocate_sdk_path(
        "/old/IsaacLab-Protomotions/source/api.py", root=tmp_path
    ) == str(root / "source/api.py")
    assert paths.relocate_sdk_path(
        "/old/ProtoMotions/src/api.py", root=tmp_path
    ) == str(tmp_path / "src/api.py")
    assert (
        paths.relocate_sdk_path("/old/notIsaacLab/source/api.py", root=tmp_path)
        == "/old/notIsaacLab/source/api.py"
    )
    assert paths.relocate_sdk_path(str(tmp_path / "existing"), root=tmp_path) == str(
        tmp_path / "existing"
    )


@pytest.mark.parametrize("result,count", [(0, 1), (0, 0), (17, 1)])
def test_cuda_driver_count_and_error_codes_are_admitted_before_kit(
    monkeypatch, result, count
):
    class Driver:
        def cuInit(self, flags):
            return 0

        def cuDeviceGetCount(self, pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = count
            return result

    monkeypatch.setattr(ctypes, "CDLL", lambda name: Driver())
    if result or not count:
        with pytest.raises(RuntimeError, match="Kit was not started"):
            paths.require_cuda_driver()
    else:
        paths.require_cuda_driver()


def test_missing_driver_library_retains_original_cause(monkeypatch):
    error = OSError("missing native library")

    def missing(name):
        raise error

    monkeypatch.setattr(ctypes, "CDLL", missing)
    with pytest.raises(RuntimeError, match="library unavailable") as caught:
        paths.require_cuda_driver()
    assert caught.value.__cause__ is error
