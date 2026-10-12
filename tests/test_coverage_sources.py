"""Credit virtual frames only after a byte-for-byte executable-code proof."""

import atexit
import hashlib
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from social_navigation import coverage_sources as sources


@pytest.fixture
def source_checkout(tmp_path):
    root = tmp_path / "source's 目录"
    root.mkdir()
    source = root / "program.py"
    source.write_text(
        "def unchanged(value):\n"
        "    return value + 7\n"
        "\n"
        "def changed(value):\n"
        "    return value * 2\n"
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "program.py",
                        "bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    return root, source, baseline, tmp_path / "proofs"


@pytest.fixture
def plugin(source_checkout):
    root, _, baseline, proofs = source_checkout
    instance = sources.VerifiedSources(root, baseline, proofs)
    yield instance
    atexit.unregister(instance.save_proofs)


def test_identity_changes_only_filename_and_preserves_nested_code(source_checkout):
    _, source, _, _ = source_checkout
    code = compile(source.read_bytes(), str(source), "exec", dont_inherit=True)
    virtual = compile(
        source.read_bytes(), str(source) + ":test", "exec", dont_inherit=True
    )
    assert sources.code_identity(code) == sources.code_identity(virtual)
    assert sources.compiled_identities(code) == sources.compiled_identities(virtual)
    assert len(sources.compiled_identities(code)) == 3
    altered = compile(source.read_text().replace("* 2", "* 3"), str(source), "exec")
    assert sources.code_identity(code) != sources.code_identity(altered)
    normalized = sources.normalize_code(code)
    assert normalized.co_filename == ""
    assert normalized.co_consts[0].co_filename == ""


def test_real_and_virtual_tracers_keep_actual_line_numbers(plugin, source_checkout):
    _, source, _, _ = source_checkout
    real = plugin.file_tracer(str(source))
    assert real.source_filename() == str(source)
    assert not real.has_dynamic_source_filename()
    assert real.line_number_range(SimpleNamespace(f_lineno=17)) == (17, 17)
    assert plugin.file_reporter(str(source)) == "python"
    virtual_name = str(source) + ":exercise"
    tracer = plugin.file_tracer(virtual_name)
    assert tracer.has_dynamic_source_filename()
    code = compile(source.read_bytes(), virtual_name, "exec", dont_inherit=True)
    frame = SimpleNamespace(f_code=code, f_lineno=1)
    assert tracer.dynamic_source_filename(virtual_name, frame) == str(source)
    assert tracer.dynamic_source_filename(virtual_name, frame) == str(source)
    assert len(plugin.proofs) == 1
    assert plugin.file_tracer(str(source.parent / "not_owned.py")) is None
    assert plugin.file_tracer(str(source.parent / "not_owned.py:case")) is None


@pytest.mark.parametrize(
    "alter", [lambda text: "\n" + text, lambda text: text.replace("+ 7", "+ 8")]
)
def test_changed_or_shifted_functions_receive_no_source_credit(
    plugin, source_checkout, alter
):
    _, source, _, _ = source_checkout
    virtual_name = str(source) + ":altered"
    code = compile(alter(source.read_text()), virtual_name, "exec", dont_inherit=True)
    tracer = plugin.file_tracer(virtual_name)
    for original in (code, code.co_consts[0]):
        assert (
            tracer.dynamic_source_filename(
                virtual_name, SimpleNamespace(f_code=original)
            )
            == virtual_name
        )
    assert not plugin.proofs


def test_same_lines_with_different_qualname_or_flags_are_rejected(
    plugin, source_checkout
):
    _, source, _, _ = source_checkout
    virtual_name = str(source) + ":altered"
    code = compile(
        source.read_bytes(), virtual_name, "exec", dont_inherit=True
    ).co_consts[0]
    tracer = plugin.file_tracer(virtual_name)
    for altered in (
        code.replace(co_qualname="other_function"),
        code.replace(co_flags=code.co_flags ^ 0x1000000),
    ):
        assert (
            tracer.dynamic_source_filename(
                virtual_name, SimpleNamespace(f_code=altered)
            )
            == virtual_name
        )
    assert not plugin.proofs


def test_stale_source_cannot_prove_virtual_frames(plugin, source_checkout):
    _, source, _, _ = source_checkout
    virtual_name = str(source) + ":stale"
    code = compile(source.read_bytes(), virtual_name, "exec", dont_inherit=True)
    source.write_text(source.read_text().replace("+ 7", "+ 8"))
    assert plugin.identities(source) == set()
    assert (
        plugin.file_tracer(virtual_name).dynamic_source_filename(
            virtual_name, SimpleNamespace(f_code=code)
        )
        == virtual_name
    )


def test_inventory_cannot_escape_checkout(source_checkout):
    root, _, baseline, proofs = source_checkout
    baseline.write_text(json.dumps({"files": [{"path": "../outside.py"}]}))
    with pytest.raises(ValueError, match="escapes"):
        sources.VerifiedSources(root, baseline, proofs)


def test_registration_and_proofs_bind_source_and_code(
    plugin, source_checkout, monkeypatch
):
    root, source, baseline, proofs = source_checkout
    registered = []
    monkeypatch.setattr(
        sources, "VerifiedSources", lambda *args: (registered.append(args), plugin)[1]
    )
    registry = SimpleNamespace(add_file_tracer=lambda value: registered.append(value))
    sources.coverage_init(
        registry,
        {"root": str(root), "baseline": str(baseline), "proof_dir": str(proofs)},
    )
    assert registered == [(root, baseline, proofs), plugin]
    virtual_name = str(source) + ":verified"
    code = compile(source.read_bytes(), virtual_name, "exec", dont_inherit=True)
    plugin.file_tracer(virtual_name).dynamic_source_filename(
        virtual_name, SimpleNamespace(f_code=code)
    )
    plugin.save_proofs()
    rows = json.loads(next(proofs.glob("*.json")).read_text())["frames"]
    assert rows == [
        {
            "virtual_filename": virtual_name,
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "qualified_name": "<module>",
            "first_line": 1,
            "code_sha256": sources.code_identity(code).hex(),
        }
    ]
    plugin.save_proofs()
    assert len(list(proofs.glob("*.json"))) == 1


def test_plugin_file_spec_load_and_registration_are_independent_of_startup_import(
    source_checkout,
):
    # coverage imports its plugin before measurement starts. A file-spec load
    # also verifies deployment tooling can create and register a fresh module,
    # with the same code identities and without changing the active plugin.
    spec = importlib.util.spec_from_file_location(
        "coverage_sources_fresh", sources.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, source, baseline, proofs = source_checkout
    registered = []
    module.coverage_init(
        SimpleNamespace(add_file_tracer=registered.append),
        {"root": str(root), "baseline": str(baseline), "proof_dir": str(proofs)},
    )
    plugin = registered[0]
    try:
        code = compile(source.read_bytes(), str(source), "exec", dont_inherit=True)
        assert plugin.identities(source) == sources.compiled_identities(code)
        assert module.code_identity(code) == sources.code_identity(code)
        assert plugin.file_tracer(str(source)).source_filename() == str(source)
    finally:
        atexit.unregister(plugin.save_proofs)


def test_real_coverage_merges_direct_and_identical_frames_but_rejects_edits(
    source_checkout, tmp_path
):
    root, source, baseline, proofs = source_checkout
    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\nbranch = true\nplugins = social_navigation.coverage_sources\n"
        + f"data_file = {tmp_path / '.coverage'}\n"
        + f"include = {root}/*\n"
        + "[social_navigation.coverage_sources]\n"
        + f"root = {root}\nbaseline = {baseline}\nproof_dir = {proofs}\n"
    )
    program = r"""
import coverage, json, sys
from pathlib import Path
source, config = map(Path, sys.argv[1:])
cov = coverage.Coverage(config_file=str(config)); cov.start()
raw = source.read_text()
direct = {}; exec(compile(raw, str(source), 'exec', dont_inherit=True), direct)
assert direct['changed'](5) == 10
virtual = str(source) + ':transformed'
namespace = {}; exec(compile(raw.replace('* 2', '* 3'), virtual, 'exec', dont_inherit=True), namespace)
assert namespace['unchanged'](5) == 12
assert namespace['changed'](5) == 15
cov.stop(); cov.save()
data = cov.get_data()
print(json.dumps({name: {'lines': data.lines(name), 'tracer': data.file_tracer(name)} for name in data.measured_files()}))
"""
    child = subprocess.run(
        [sys.executable, "-c", program, str(source), str(config)],
        text=True,
        capture_output=True,
        check=True,
    )
    measured = json.loads(child.stdout)
    assert {1, 2, 4, 5}.issubset(measured[str(source)]["lines"])
    assert 2 not in measured[str(source) + ":transformed"]["lines"]
    assert 5 in measured[str(source) + ":transformed"]["lines"]
    assert (
        measured[str(source)]["tracer"]
        == "social_navigation.coverage_sources.VerifiedSources"
    )
    proof = json.loads(next(proofs.glob("*.json")).read_text())
    assert [row["qualified_name"] for row in proof["frames"]] == ["unchanged"]
