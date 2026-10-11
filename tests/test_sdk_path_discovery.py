from types import SimpleNamespace
import pytest
from social_navigation import sdk_paths


@pytest.mark.parametrize('locations', [['/isaaclab'], ['/tmp/isaaclab'], []])
def test_short_or_shadow_namespace_fails_without_index_error(monkeypatch, locations):
    monkeypatch.delenv('ISAACLAB_ROOT', raising=False)
    monkeypatch.setattr(sdk_paths.util, 'find_spec', lambda _: SimpleNamespace(submodule_search_locations=locations))
    with pytest.raises(RuntimeError, match='IsaacLab'):
        sdk_paths.lab_root()


def test_one_real_checkout_can_follow_an_irrelevant_namespace(monkeypatch, tmp_path):
    root = tmp_path/'lab'
    package = root/'source/isaaclab/isaaclab'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    monkeypatch.delenv('ISAACLAB_ROOT', raising=False)
    monkeypatch.setattr(sdk_paths.util,'find_spec',lambda _:SimpleNamespace(submodule_search_locations=['/tmp/isaaclab',str(package)]))
    assert sdk_paths.lab_root() == root
