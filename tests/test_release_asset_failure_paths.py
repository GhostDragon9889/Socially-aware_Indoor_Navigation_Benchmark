"""Real file/gzip restoration and failure admission for portable asset releases."""
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from social_navigation import asset_release as release
from social_navigation import reproduction


def row(path, payload, **fields):
    return {'path': path, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
            'role': 'test_resource', **fields}


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'checkout 中文'; root.mkdir()
    folder = root / release.RELEASE_PATH; folder.mkdir(parents=True)
    model = b'fixture resource bytes; no trained model claim'
    usd = b'#usda 1.0\ndef Xform "World" {}\n'
    packed = gzip.compress(usd)
    (folder / 'weight.bin').write_bytes(model); (folder / 'original.usda.gz').write_bytes(packed)
    rows = [row('weight.bin', model, install_path='data/pretrained_models/fixture.bin'),
            row('original.usda.gz', packed, role='exact_original_usd_gzip',
                source_sha256=hashlib.sha256(usd).hexdigest(), uncompressed_bytes=len(usd))]
    manifest = {'schema': 'social_navigation.github_asset_release.v1', 'repository_relative_root': release.RELEASE_PATH.as_posix(), 'files': rows}
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    return root, folder, manifest, usd


def save_manifest(folder, manifest):
    (folder / 'manifest.json').write_text(json.dumps(manifest))


def test_verify_install_and_reuse_preserve_exact_bytes(bundle):
    root, folder, _, _ = bundle
    report = release.verify(folder)
    assert report['files'] == 2 and report['original_usd_snapshots_verified'] == 1
    assert release.install_models(root, folder)['new_files'] == 1
    target = root / 'data/pretrained_models/fixture.bin'
    assert target.read_bytes() == (folder / 'weight.bin').read_bytes()
    assert release.install_models(root, folder)['new_files'] == 0
    target.write_bytes(b'previous user data')
    with pytest.raises(ValueError, match='retained without overwrite'): release.install_models(root, folder)
    assert target.read_bytes() == b'previous user data'


@pytest.mark.parametrize('defect', ['schema', 'duplicate', 'size', 'sha256', 'source_sha256', 'uncompressed_bytes'])
def test_release_corruption_is_rejected(bundle, defect):
    _, folder, manifest, _ = bundle
    if defect == 'schema': manifest['schema'] = 'unknown'
    elif defect == 'duplicate': manifest['files'].append(dict(manifest['files'][0]))
    elif defect in ('size', 'sha256'):
        manifest['files'][0]['bytes' if defect == 'size' else 'sha256'] = 0 if defect == 'size' else '0'*64
    else: manifest['files'][1][defect] = '0'*64 if defect == 'source_sha256' else 0
    save_manifest(folder, manifest)
    with pytest.raises(ValueError): release.verify(folder)


@pytest.mark.parametrize('path', ['../outside', '/outside', 'missing'])
def test_release_file_must_be_contained_and_present(bundle, path):
    _, folder, _, _ = bundle
    with pytest.raises(ValueError): release.contained_file(folder, path)


def test_release_dependency_symlink_cannot_escape(bundle, tmp_path):
    _, folder, _, _ = bundle
    outside = tmp_path / 'outside'; outside.write_bytes(b'real file')
    (folder / 'link').symlink_to(outside)
    with pytest.raises(ValueError): release.contained_file(folder, 'link')


@pytest.mark.parametrize('install', ['../outside', '/outside', 'resources/foreign'])
def test_model_install_rejects_invalid_destination(bundle, install):
    root, folder, manifest, _ = bundle
    manifest['files'][0]['install_path'] = install; save_manifest(folder, manifest)
    with pytest.raises(ValueError, match='Invalid install path'): release.install_models(root, folder)
    assert not (root / 'data').exists()


def test_model_install_rejects_directory_symlink_escape(bundle, tmp_path):
    root, folder, _, _ = bundle
    (root / 'data').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='escapes checkout'): release.install_models(root, folder)


def test_plain_resource_copy_rejects_source_and_destination_corruption(tmp_path):
    source = tmp_path / 'source'; source.write_bytes(b'original')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / 'destination'
    with pytest.raises(ValueError, match='source differs'): reproduction._copy_exact(source, target, '0'*64)
    assert not target.exists()
    assert reproduction._copy_exact(source, target, expected)
    assert not reproduction._copy_exact(source, target, expected)
    target.write_bytes(b'user modified resource')
    with pytest.raises(ValueError, match='retained'): reproduction._copy_exact(source, target, expected)
    assert target.read_bytes() == b'user modified resource'


def bind_restorations(root, folder, usd, *, target='data/scene.usda', compressed=True):
    config = root / 'configs/reproduction'; config.mkdir(parents=True)
    document = {'release_manifest_sha256': release.sha256(folder / 'manifest.json'), 'restorations': [
        {'release_path': 'original.usda.gz' if compressed else 'weight.bin', 'target': target,
         'sha256': hashlib.sha256(usd if compressed else (folder/'weight.bin').read_bytes()).hexdigest(),
         'bytes': len(usd), 'gzip': compressed}]}
    (config / 'resource_bindings.json').write_text(json.dumps(document))
    visuals = root / 'resources/pedestrian_visuals'; visuals.mkdir(parents=True)
    payload = b'visual resource fixture'
    (visuals / 'mesh.bin').write_bytes(payload)
    (visuals / 'manifest.json').write_text(json.dumps({'files': [row('mesh.bin', payload, install_path='data/visuals/mesh.bin')]}))
    return document


@pytest.mark.parametrize('compressed', [False, True])
def test_resource_restoration_and_repeat_are_exact(bundle, compressed):
    root, folder, _, usd = bundle
    bind_restorations(root, folder, usd, compressed=compressed)
    result = reproduction.prepare_resources(root, folder)
    assert result['new_files'] == 2
    assert reproduction.prepare_resources(root, folder)['new_files'] == 0
    if compressed:
        target = root / 'data/scene.usda'; assert target.read_bytes() == usd
        target.write_text('user altered USD')
        with pytest.raises(ValueError, match='Existing exact USD differs'): reproduction.prepare_resources(root, folder)
        assert target.read_text() == 'user altered USD'


def test_gzip_checksum_failure_retains_partial_diagnostic(bundle):
    root, folder, _, usd = bundle
    config = bind_restorations(root, folder, usd)
    config['restorations'][0]['sha256'] = '0'*64
    (root / 'configs/reproduction/resource_bindings.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='decompression checksum'): reproduction.prepare_resources(root, folder)
    assert not (root / 'data/scene.usda').exists()
    assert (root / 'data/scene.usda.partial').read_bytes() == usd


@pytest.mark.parametrize('defect', ['manifest', 'target', 'visual_target'])
def test_resource_binding_failure_does_not_modify_external_paths(bundle, defect):
    root, folder, _, usd = bundle
    config = bind_restorations(root, folder, usd)
    if defect == 'manifest': config['release_manifest_sha256'] = '0'*64
    elif defect == 'target': config['restorations'][0]['target'] = '../outside'
    else:
        p = root / 'resources/pedestrian_visuals/manifest.json'; value = json.loads(p.read_text())
        value['files'][0]['install_path'] = '../outside'; p.write_text(json.dumps(value))
    (root / 'configs/reproduction/resource_bindings.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='differs|escapes'): reproduction.prepare_resources(root, folder)
    assert not (root.parent / 'outside').exists()


def test_fetch_without_project_remote_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(release.subprocess, 'check_output', lambda command, **kw: 'origin\n' if command[-1]=='remote' else 'https://github.com/NVlabs/ProtoMotions.git\n')
    with pytest.raises(ValueError, match='No project GitHub remote'): release.fetch(tmp_path)
