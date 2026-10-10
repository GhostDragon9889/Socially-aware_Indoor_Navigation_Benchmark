"""CPU checks for exact authored dependencies, scope, and local composition."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pxr import Sdf, Usd, UsdGeom


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_dynbench_missing.py"
SPEC = importlib.util.spec_from_file_location("dynbench_download", SCRIPT)
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


def test_destination_scope_rejects_encoded_escape_and_external_root(tmp_path):
    root = "https://example.invalid/catalog/6.0"
    expected = tmp_path / "Isaac/Props/Forklift/forklift.usd"
    assert DOWNLOADER.relative_destination(root + "/Isaac/Props/Forklift/forklift.usd", root, tmp_path) == expected
    for uri in [root + "/%2e%2e/%2e%2e/escape.usd", "https://other.invalid/catalog/6.0/file.usd",
                root + "/file.usd?token=secret", root + "../file.usd"]:
        with pytest.raises(ValueError):
            DOWNLOADER.relative_destination(uri, root, tmp_path)


def test_usd_dependency_scan_includes_composition_arrays_and_time_samples(tmp_path):
    path = tmp_path / "model.usda"
    stage = Usd.Stage.CreateNew(str(path))
    prim = stage.DefinePrim("/Model", "Xform")
    prim.GetReferences().AddReference("geometry.usda")
    prim.CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("tex/default.png"))
    prim.GetAttribute("texture").Set(Sdf.AssetPath("tex/animated.png"), Usd.TimeCode(1))
    prim.CreateAttribute("textures", Sdf.ValueTypeNames.AssetArray).Set(
        [Sdf.AssetPath("tex/one.png"), Sdf.AssetPath("tex/two.png")])
    stage.GetRootLayer().subLayerPaths.append("appearance.usda")
    stage.GetRootLayer().Save()
    before = DOWNLOADER.digest(path)
    records = DOWNLOADER.dependency_records(path)
    assert {record["path"] for record in records} == {
        "geometry.usda", "appearance.usda", "tex/default.png", "tex/animated.png", "tex/one.png", "tex/two.png"}
    assert DOWNLOADER.digest(path) == before


def test_mdl_scan_retains_module_namespaces_and_static_resource_constructors():
    source = '''mdl 1.7;
    import ::nvidia::core_definitions::file_texture;
    import LocalMaterial::*;
    using .::aux_definitions import xy;
    texture_2d("texture/albedo.png");
    anno::thumbnail(".thumbs/editor.png");
    // import Ghost::*; texture_2d("ghost.png");
    '''
    records = DOWNLOADER.mdl_dependency_records(source)
    assert {record["path"] for record in records} == {
        "nvidia/core_definitions.mdl", "LocalMaterial.mdl", "./aux_definitions.mdl", "texture/albedo.png"}
    assert next(record for record in records if record["path"] == "nvidia/core_definitions.mdl")["absolute_module"]
    assert not next(record for record in records if record["path"] == "LocalMaterial.mdl")["absolute_module"]
    with pytest.raises(ValueError, match="Unsupported MDL"):
        DOWNLOADER.mdl_dependency_records("import ::a::b as renamed;")


def test_installed_material_source_closure_is_real_and_not_compiler_claim():
    if not DOWNLOADER.SDK_MDL.is_dir():
        pytest.skip("Isaac Sim material sources are not installed in this environment")
    result = DOWNLOADER.verify_installed_mdl(DOWNLOADER.SDK_MDL / "OmniPBR.mdl")
    assert result["source_closure_verified"] and not result["mdl_compiler_verified"]
    assert len(result["files"]) > 10
    assert all(Path(record["installed_path"]).is_file() and record["bytes"] > 0
               and DOWNLOADER.digest(record["installed_path"]) == record["sha256"] for record in result["files"])


def test_offline_model_validation_loads_actual_geometry_and_units(tmp_path):
    path = tmp_path / "model.usda"
    stage = Usd.Stage.CreateNew(str(path))
    cube = UsdGeom.Cube.Define(stage, "/Model")
    cube.CreateSizeAttr(2.0)
    stage.SetDefaultPrim(cube.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, "Y")
    stage.GetRootLayer().Save()
    result = DOWNLOADER.validate_local_model(path)
    assert result["gprim_count"] == 1
    assert result["default_prim"] == "/Model"
    assert result["meters_per_unit"] == 0.01
    assert result["up_axis"] == "Y"
    assert result["bbox_stage_units"] == [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]


def test_offline_model_validation_rejects_missing_composition(tmp_path):
    path = tmp_path / "missing.usda"
    stage = Usd.Stage.CreateNew(str(path))
    cube = UsdGeom.Cube.Define(stage, "/Model")
    cube.GetPrim().GetReferences().AddReference("absent.usda")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="composition"):
        DOWNLOADER.validate_local_model(path)


def _isolated_live_kit_contract(monkeypatch, tmp_path, *, app_exists=True, enable_succeeds=True):
    """Isolate extension ordering only, never claim real API availability."""
    specification = importlib.util.spec_from_file_location(
        "dynbench_resolver", SCRIPT.parent / "resolve_paper_scene_assets.py")
    resolver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(resolver)
    monkeypatch.setattr(resolver, "__file__", str(tmp_path / "scripts/resolve.py"))
    events = []
    for name in ["omni", "omni.kit", "omni.kit.app", "isaacsim"]:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "isaacsim.storage.native", raising=False)

    def enable(name, enabled):
        events.append(("enable", name, enabled))
        if enable_succeeds:
            storage = ModuleType("isaacsim.storage")
            storage.__path__ = []
            native = ModuleType("isaacsim.storage.native")

            def root_api(*, skip_check):
                events.append(("root_api", skip_check))
                return "https://example.invalid/assets/6.0"

            native.get_assets_root_path = root_api
            native.path_join = lambda root, relative: root.rstrip("/") + "/" + relative
            monkeypatch.setitem(sys.modules, "isaacsim.storage", storage)
            monkeypatch.setitem(sys.modules, "isaacsim.storage.native", native)
        return enable_succeeds

    manager = SimpleNamespace(set_extension_enabled_immediate=enable)
    app = SimpleNamespace(get_extension_manager=lambda: manager) if app_exists else None
    sys.modules["omni.kit.app"].get_app = lambda: app
    return resolver, events


def test_resolver_enables_existing_extension_before_importing_public_api(monkeypatch, tmp_path):
    resolver, events = _isolated_live_kit_contract(monkeypatch, tmp_path)
    destination = tmp_path / "request.json"
    result = resolver.resolve_requests(destination)
    assert events == [("enable", "isaacsim.storage.native", True), ("root_api", True)]
    request = json.loads(destination.read_text())
    assert request["enabled_extension"] == "isaacsim.storage.native"
    assert not request["availability_asserted"]
    assert set(result) == {"request_path", "relative_paths", "availability_asserted"}
    json.dumps(result)


@pytest.mark.parametrize("app_exists,enable_succeeds,expected_events", [
    (False, True, []),
    (True, False, [("enable", "isaacsim.storage.native", True)]),
])
def test_resolver_does_not_import_sdk_or_write_request_if_live_enablement_fails(
        monkeypatch, tmp_path, app_exists, enable_succeeds, expected_events):
    resolver, events = _isolated_live_kit_contract(
        monkeypatch, tmp_path, app_exists=app_exists, enable_succeeds=enable_succeeds)
    destination = tmp_path / "request.json"
    with pytest.raises(RuntimeError):
        resolver.resolve_requests(destination)
    assert events == expected_events
    assert not destination.exists()
