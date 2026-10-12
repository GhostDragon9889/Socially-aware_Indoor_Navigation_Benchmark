"""Cache cleanup must preserve source, model inputs and external link targets."""

import json
import shutil

import pytest

from social_navigation import cache_cleanup as cleaner


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".social-navigation-root").touch()
    return root


def cache(root, relative, contents=b"generated"):
    path = root / relative
    path.mkdir(parents=True)
    (path / "state").write_bytes(contents)
    return path


def test_only_generated_caches_and_unreferenced_backups_are_removed(checkout, tmp_path):
    root = checkout
    for name in [
        "src/__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".hypothesis",
        ".cache/pip",
        ".cache/uv",
        ".cache/pytest",
        ".cache/ruff",
        ".git/lfs/cache",
        ".git/lfs/tmp",
        ".runtime/SDK/.venv/lib/python3.12/site-packages/isaacsim/kit/cache",
        ".runtime/.external_sdk_packages_backup-old",
    ]:
        cache(root, name)
    for name in [
        "src/implementation",
        "data/pretrained_models",
        "data/datasets",
        ".git/lfs/objects",
        ".runtime/SDK/.venv/lib/python3.12/site-packages/isaacsim/extscache",
        ".cache/models",
        "src/package/cache",
        ".runtime/external_sdk_packages",
    ]:
        cache(root, name, b"required")
    binding = root / ".runtime/external_sdk.json"
    binding.write_text(
        json.dumps(
            {
                "schema": "social_navigation.external_sdk.v1",
                "addons": str(root / ".runtime/external_sdk_packages"),
            }
        )
    )
    outside = cache(tmp_path, "external/__pycache__", b"external")
    (root / "external").symlink_to(outside.parent, target_is_directory=True)
    (root / "linked_cache").symlink_to(
        root / "src/__pycache__", target_is_directory=True
    )
    (root / "src/__pycache__/external").symlink_to(outside, target_is_directory=True)
    (root / "src/__pycache__/external_file").symlink_to(outside / "state")
    inventory = cleaner.plan(root)
    assert len(inventory["entries"]) == 13
    assert inventory["logical_bytes"] == 13 * len(b"generated")
    assert cleaner.apply(inventory)["applied"]
    assert outside.is_dir()
    assert (root / "data/pretrained_models/state").read_bytes() == b"required"
    assert (root / ".git/lfs/objects/state").read_bytes() == b"required"
    assert (root / ".runtime/external_sdk_packages/state").read_bytes() == b"required"
    assert cleaner.plan(root)["entries"] == []


@pytest.mark.parametrize(
    "binding",
    [
        None,
        [],
        {},
        {"schema": "wrong"},
        {"schema": "social_navigation.external_sdk.v1", "addons": 42},
        {"schema": "social_navigation.external_sdk.v1", "addons": "elsewhere"},
    ],
)
def test_unbound_backups_are_preserved(checkout, binding):
    backup = cache(checkout, ".runtime/.external_sdk_packages_backup-recovery")
    if binding is not None:
        (checkout / ".runtime/external_sdk.json").write_text(json.dumps(binding))
    assert cleaner.plan(checkout)["entries"] == []
    assert backup.exists()


def test_binding_and_active_addons_symlinks_do_not_authorize_backup_deletion(
    checkout, tmp_path
):
    backup = cache(checkout, ".runtime/.external_sdk_packages_backup-recovery")
    active = cache(checkout, ".runtime/external_sdk_packages")
    real = tmp_path / "binding.json"
    real.write_text(
        json.dumps(
            {"schema": "social_navigation.external_sdk.v1", "addons": str(active)}
        )
    )
    binding = checkout / ".runtime/external_sdk.json"
    binding.symlink_to(real)
    assert cleaner.plan(checkout)["entries"] == []
    binding.unlink()
    binding.write_text(real.read_text())
    shutil.rmtree(active)
    active.symlink_to(cache(tmp_path, "packages"), target_is_directory=True)
    assert cleaner.plan(checkout)["entries"] == []
    assert backup.exists()


def test_marker_is_required_and_cannot_be_an_external_link(checkout, tmp_path):
    marker = checkout / ".social-navigation-root"
    marker.unlink()
    with pytest.raises(ValueError, match="marked project"):
        cleaner.plan(checkout)
    marker.symlink_to(tmp_path / "marker")
    with pytest.raises(ValueError, match="marked project"):
        cleaner.plan(checkout)


@pytest.mark.parametrize(
    "replacement", ["symlink", "directory", "file", "foreign", "traversal"]
)
def test_changed_or_forged_plan_cannot_delete_new_contents(
    checkout, tmp_path, replacement
):
    path = cache(checkout, "src/__pycache__")
    inventory = cleaner.plan(checkout)
    if replacement == "foreign":
        inventory["entries"][0]["path"] = str(tmp_path)
    elif replacement == "traversal":
        inventory["entries"][0]["path"] = "src/../data"
    else:
        path.rename(path.with_name("saved"))
        if replacement == "symlink":
            path.symlink_to(cache(tmp_path, "foreign"), target_is_directory=True)
        elif replacement == "file":
            path.write_bytes(b"required")
        else:
            cache(checkout, "src/__pycache__", b"new contents")
    with pytest.raises(ValueError):
        cleaner.apply(inventory)
    assert path.exists()


def test_all_candidates_are_admitted_before_any_deletion(checkout):
    first = cache(checkout, "src/__pycache__")
    cache(checkout, "scripts/__pycache__")
    inventory = cleaner.plan(checkout)
    inventory["entries"][-1]["inode"] = -1
    with pytest.raises(ValueError, match="changed after"):
        cleaner.apply(inventory)
    assert first.exists()


def test_cli_default_lists_and_explicit_apply_removes(checkout, tmp_path, capsys):
    path = cache(checkout, "__pycache__")
    output = tmp_path / "report/cleanup.json"
    assert cleaner.main(["--root", str(checkout)]) == 0 and path.exists()
    assert not json.loads(capsys.readouterr().out)["applied"]
    assert (
        cleaner.main(["--root", str(checkout), "--apply", "--output", str(output)]) == 0
    )
    assert json.loads(output.read_text())["removed"] == ["__pycache__"]
    assert json.loads(capsys.readouterr().out)["applied"]


@pytest.mark.parametrize("replacement", ["symlink", "file"])
def test_replacement_between_discovery_and_admission_is_rejected(
    checkout, tmp_path, monkeypatch, replacement
):
    path = cache(checkout, "__pycache__")
    inventory = cleaner.plan(checkout)

    def replaced_plan(root):
        path.rename(path.with_name("original"))
        if replacement == "symlink":
            path.symlink_to(cache(tmp_path, "outside"), target_is_directory=True)
        else:
            path.write_bytes(b"user data")
        return inventory

    monkeypatch.setattr(cleaner, "plan", replaced_plan)
    with pytest.raises(ValueError, match="escaped|regular directory"):
        cleaner.apply(inventory)
    assert path.exists()


def test_last_inode_recheck_protects_replaced_cache_before_removal(
    checkout, monkeypatch
):
    path = cache(checkout, "__pycache__")
    inventory = cleaner.plan(checkout)
    monkeypatch.setattr(cleaner, "plan", lambda root: inventory)
    original = cleaner._identity
    calls = []

    def replaced_identity(path):
        device, inode = original(path)
        calls.append(path)
        return device, inode if len(calls) == 1 else inode + 1

    monkeypatch.setattr(cleaner, "_identity", replaced_identity)
    with pytest.raises(ValueError, match="before removal"):
        cleaner.apply(inventory)
    assert path.exists()
