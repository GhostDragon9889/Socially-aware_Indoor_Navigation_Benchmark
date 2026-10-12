"""Real file/gzip restoration and failure admission for portable asset releases."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from social_navigation import asset_release as release
from social_navigation import reproduction


def row(path, payload, **fields):
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "role": "test_resource",
        **fields,
    }


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "checkout 中文"
    root.mkdir()
    folder = root / release.RELEASE_PATH
    folder.mkdir(parents=True)
    model = b"fixture resource bytes; no trained model claim"
    usd = b'#usda 1.0\ndef Xform "World" {}\n'
    packed = gzip.compress(usd)
    (folder / "weight.bin").write_bytes(model)
    (folder / "original.usda.gz").write_bytes(packed)
    rows = [
        row("weight.bin", model, install_path="data/pretrained_models/fixture.bin"),
        row(
            "original.usda.gz",
            packed,
            role="exact_original_usd_gzip",
            source_sha256=hashlib.sha256(usd).hexdigest(),
            uncompressed_bytes=len(usd),
        ),
    ]
    manifest = {
        "schema": "social_navigation.github_asset_release.v1",
        "repository_relative_root": release.RELEASE_PATH.as_posix(),
        "files": rows,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return root, folder, manifest, usd


def save_manifest(folder, manifest):
    (folder / "manifest.json").write_text(json.dumps(manifest))


def test_verify_install_and_reuse_preserve_exact_bytes(bundle):
    root, folder, _, _ = bundle
    report = release.verify(folder)
    assert report["files"] == 2 and report["original_usd_snapshots_verified"] == 1
    assert release.install_models(root, folder)["new_files"] == 1
    target = root / "data/pretrained_models/fixture.bin"
    assert target.read_bytes() == (folder / "weight.bin").read_bytes()
    assert release.install_models(root, folder)["new_files"] == 0
    target.write_bytes(b"previous user data")
    with pytest.raises(ValueError, match="retained without overwrite"):
        release.install_models(root, folder)
    assert target.read_bytes() == b"previous user data"


@pytest.mark.parametrize(
    "defect",
    ["schema", "duplicate", "size", "sha256", "source_sha256", "uncompressed_bytes"],
)
def test_release_corruption_is_rejected(bundle, defect):
    _, folder, manifest, _ = bundle
    if defect == "schema":
        manifest["schema"] = "unknown"
    elif defect == "duplicate":
        manifest["files"].append(dict(manifest["files"][0]))
    elif defect in ("size", "sha256"):
        manifest["files"][0]["bytes" if defect == "size" else "sha256"] = (
            0 if defect == "size" else "0" * 64
        )
    else:
        manifest["files"][1][defect] = "0" * 64 if defect == "source_sha256" else 0
    save_manifest(folder, manifest)
    with pytest.raises(ValueError):
        release.verify(folder)


@pytest.mark.parametrize("path", ["../outside", "/outside", "missing"])
def test_release_file_must_be_contained_and_present(bundle, path):
    _, folder, _, _ = bundle
    with pytest.raises(ValueError):
        release.contained_file(folder, path)


def test_release_dependency_symlink_cannot_escape(bundle, tmp_path):
    _, folder, _, _ = bundle
    outside = tmp_path / "outside"
    outside.write_bytes(b"real file")
    (folder / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        release.contained_file(folder, "link")


@pytest.mark.parametrize("install", ["../outside", "/outside", "resources/foreign"])
def test_model_install_rejects_invalid_destination(bundle, install):
    root, folder, manifest, _ = bundle
    manifest["files"][0]["install_path"] = install
    save_manifest(folder, manifest)
    with pytest.raises(ValueError, match="Invalid install path"):
        release.install_models(root, folder)
    assert not (root / "data").exists()


def test_model_install_rejects_directory_symlink_escape(bundle, tmp_path):
    root, folder, _, _ = bundle
    (root / "data").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes checkout"):
        release.install_models(root, folder)


def test_plain_resource_copy_rejects_source_and_destination_corruption(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "destination"
    with pytest.raises(ValueError, match="source differs"):
        reproduction._copy_exact(source, target, "0" * 64)
    assert not target.exists()
    assert reproduction._copy_exact(source, target, expected)
    assert not reproduction._copy_exact(source, target, expected)
    target.write_bytes(b"user modified resource")
    with pytest.raises(ValueError, match="retained"):
        reproduction._copy_exact(source, target, expected)
    assert target.read_bytes() == b"user modified resource"


def bind_restorations(root, folder, usd, *, target="data/scene.usda", compressed=True):
    config = root / "configs/reproduction"
    config.mkdir(parents=True)
    document = {
        "release_manifest_sha256": release.sha256(folder / "manifest.json"),
        "restorations": [
            {
                "release_path": "original.usda.gz" if compressed else "weight.bin",
                "target": target,
                "sha256": hashlib.sha256(
                    usd if compressed else (folder / "weight.bin").read_bytes()
                ).hexdigest(),
                "bytes": len(usd),
                "gzip": compressed,
            }
        ],
    }
    (config / "resource_bindings.json").write_text(json.dumps(document))
    visuals = root / "resources/pedestrian_visuals"
    visuals.mkdir(parents=True)
    payload = b"visual resource fixture"
    (visuals / "mesh.bin").write_bytes(payload)
    (visuals / "manifest.json").write_text(
        json.dumps(
            {"files": [row("mesh.bin", payload, install_path="data/visuals/mesh.bin")]}
        )
    )
    return document


@pytest.mark.parametrize("compressed", [False, True])
def test_resource_restoration_and_repeat_are_exact(bundle, compressed):
    root, folder, _, usd = bundle
    bind_restorations(root, folder, usd, compressed=compressed)
    result = reproduction.prepare_resources(root, folder)
    assert result["new_files"] == 2
    assert reproduction.prepare_resources(root, folder)["new_files"] == 0
    if compressed:
        target = root / "data/scene.usda"
        assert target.read_bytes() == usd
        target.write_text("user altered USD")
        with pytest.raises(ValueError, match="Existing exact USD differs"):
            reproduction.prepare_resources(root, folder)
        assert target.read_text() == "user altered USD"


def test_gzip_checksum_failure_retains_partial_diagnostic(bundle):
    root, folder, _, usd = bundle
    config = bind_restorations(root, folder, usd)
    config["restorations"][0]["sha256"] = "0" * 64
    (root / "configs/reproduction/resource_bindings.json").write_text(
        json.dumps(config)
    )
    with pytest.raises(ValueError, match="decompression checksum"):
        reproduction.prepare_resources(root, folder)
    assert not (root / "data/scene.usda").exists()
    assert (root / "data/scene.usda.partial").read_bytes() == usd


@pytest.mark.parametrize("defect", ["manifest", "target", "visual_target"])
def test_resource_binding_failure_does_not_modify_external_paths(bundle, defect):
    root, folder, _, usd = bundle
    config = bind_restorations(root, folder, usd)
    if defect == "manifest":
        config["release_manifest_sha256"] = "0" * 64
    elif defect == "target":
        config["restorations"][0]["target"] = "../outside"
    else:
        p = root / "resources/pedestrian_visuals/manifest.json"
        value = json.loads(p.read_text())
        value["files"][0]["install_path"] = "../outside"
        p.write_text(json.dumps(value))
    (root / "configs/reproduction/resource_bindings.json").write_text(
        json.dumps(config)
    )
    with pytest.raises(ValueError, match="differs|escapes"):
        reproduction.prepare_resources(root, folder)
    assert not (root.parent / "outside").exists()


def test_fetch_without_project_remote_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        release.subprocess,
        "check_output",
        lambda command, **kw: (
            "origin\n"
            if command[-1] == "remote"
            else "https://github.com/NVlabs/ProtoMotions.git\n"
        ),
    )
    with pytest.raises(ValueError, match="No project GitHub remote"):
        release.fetch(tmp_path)


def test_invalid_model_source_never_publishes_corrupt_destination(bundle):
    root, folder, _, _ = bundle
    (folder / "weight.bin").write_bytes(b"corrupted source")
    with pytest.raises(ValueError):
        release.install_models(root, folder)
    assert not (root / "data/pretrained_models/fixture.bin").exists()


def test_copy_failure_cannot_leave_partial_resource_as_the_destination(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.write_bytes(b"complete resource")
    target = tmp_path / "installed.bin"
    expected = release.sha256(source)

    def interrupted_copy(source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(reproduction.shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError, match="interrupted"):
        reproduction._copy_exact(source, target, expected)
    assert not target.exists()


def test_model_install_rejects_conflicting_destinations_before_copying(bundle):
    root, folder, manifest, _ = bundle
    manifest["files"].append(dict(manifest["files"][0], path="other.bin"))
    (folder / "other.bin").write_bytes((folder / "weight.bin").read_bytes())
    save_manifest(folder, manifest)
    with pytest.raises(ValueError, match="Duplicate model installation"):
        release.install_models(root, folder)
    assert not (root / "data").exists()


def test_equivalent_install_paths_cannot_publish_a_partial_model_set(bundle):
    root, folder, manifest, _ = bundle
    second = dict(
        manifest["files"][0],
        path="other.bin",
        install_path="data/pretrained_models/./fixture.bin",
    )
    (folder / "other.bin").write_bytes((folder / "weight.bin").read_bytes())
    manifest["files"].append(second)
    save_manifest(folder, manifest)
    with pytest.raises(ValueError, match="Duplicate model installation"):
        release.install_models(root, folder)
    assert not (root / "data/pretrained_models/fixture.bin").exists()


@pytest.mark.parametrize(
    "dependency", ["none", "builtin", "missing", "external", "bad_catalog"]
)
def test_openusd_dependency_closure_uses_real_layers(bundle, tmp_path, dependency):
    _, folder, manifest, _ = bundle
    content = '#usda 1.0\ndef Xform "World" {}\n'
    if dependency != "none":
        asset = {"builtin": "OmniPBR.mdl", "missing": "missing_texture.png"}.get(
            dependency
        )
        if dependency == "external":
            outside = tmp_path / "outside.usda"
            outside.write_text(content)
            asset = str(outside)
        if asset is not None:
            content = (
                '#usda 1.0\ndef Xform "World" {\n    custom asset dependency = @'
                + asset
                + "@\n}\n"
            )
    stage = folder / "scene.usda"
    stage.write_text(content)
    manifest["files"].append(row("scene.usda", stage.read_bytes()))
    save_manifest(folder, manifest)
    entry = manifest["repository_relative_root"] + "/scene.usda"
    catalog = {
        "robot": {"stage": entry},
        "scenes": {
            "fixture": {"current_stage": entry, "tested_static_stage_relocated": entry}
        },
    }
    if dependency == "bad_catalog":
        catalog["robot"]["stage"] = "outside/scene.usda"
    (folder / "catalog.json").write_text(json.dumps(catalog))
    if dependency in ("none", "builtin"):
        report = release.verify(folder, check_usd=True)
        assert report["status"] == "verified" and len(report["usd_stages"]) == 3
        assert all(r["local_dependencies"] == 1 for r in report["usd_stages"])
        assert report["usd_stages"][0]["sdk_mdl_dependencies"] == (
            [] if dependency == "none" else ["OmniPBR.mdl"]
        )
    else:
        with pytest.raises(
            ValueError,
            match="Unresolved USD asset|USD dependency escapes|Invalid catalog stage",
        ):
            release.verify(folder, check_usd=True)


@pytest.mark.parametrize("race", ["same", "different", "disappeared"])
def test_concurrent_publication_never_replaces_existing_user_bytes(
    tmp_path, monkeypatch, race
):
    from social_navigation import resource_copy

    source = tmp_path / "source"
    source.write_bytes(b"verified bytes")
    target = tmp_path / "destination"

    def competing_writer(temporary, destination):
        if race != "disappeared":
            target.write_bytes(source.read_bytes() if race == "same" else b"user bytes")
        raise FileExistsError("another writer")

    monkeypatch.setattr(resource_copy.os, "link", competing_writer)
    if race == "same":
        assert resource_copy.copy_exact(source, target, release.sha256(source)) is False
    else:
        with pytest.raises((ValueError, RuntimeError), match="retained|disappeared"):
            resource_copy.copy_exact(source, target, release.sha256(source))
    if race == "different":
        assert target.read_bytes() == b"user bytes"
    assert not list(tmp_path.glob("*.partial")) and not list(
        tmp_path.glob(".*.partial")
    )


def test_corrupt_copy_is_never_published_and_existing_link_is_preserved(
    tmp_path, monkeypatch
):
    from social_navigation import resource_copy

    source = tmp_path / "source"
    source.write_bytes(b"verified")
    target = tmp_path / "destination"
    monkeypatch.setattr(
        resource_copy.shutil, "copyfile", lambda a, b: Path(b).write_bytes(b"corrupt")
    )
    with pytest.raises(ValueError, match="Copy checksum"):
        resource_copy.copy_exact(source, target, release.sha256(source))
    assert not target.exists()
    target.symlink_to(source)
    with pytest.raises(ValueError, match="resource link"):
        resource_copy.copy_exact(source, target, release.sha256(source))
    assert target.is_symlink() and source.read_bytes() == b"verified"
