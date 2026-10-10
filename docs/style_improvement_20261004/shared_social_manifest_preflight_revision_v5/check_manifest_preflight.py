"""Regression checks for writable fixed data and rejection before App startup."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace


def main():
    entry = Path(__file__).with_name("probe_shared_world.py")
    spec = importlib.util.spec_from_file_location("_manifest_preflight_test", entry)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    original_sha = probe.MANIFEST_SHA
    manifest = probe.ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
    record, document = probe._manifest_preflight(manifest)
    assert record["sha256"] == original_sha
    assert document["splits"]["train"]["clip_count"] == 2848
    with tempfile.TemporaryDirectory(prefix="shared-social-manifest-") as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_bytes(manifest.read_bytes())
        path.chmod(0o664)
        assert probe._manifest_preflight(path)[0]["sha256"] == original_sha
        path.write_text("{}")
        report = {}
        try:
            probe._run(SimpleNamespace(manifest=path), report)
        except ValueError as error:
            assert str(error) == "Fixed v7 manifest SHA mismatch"
        else:
            raise AssertionError("A changed manifest was admitted")
        assert "AMASS_manifest" not in report
        assert "torch" not in sys.modules and "isaaclab.app" not in sys.modules
        path.write_text(json.dumps({"schema": "navisaaclab.amass_reconstruction.v1",
            "author_clip_list": False, "splits": {"train": {"clip_count": 0, "clips": []}}}))
        probe.MANIFEST_SHA = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            probe._manifest_preflight(path)
        except ValueError as error:
            assert "train2848" in str(error)
        else:
            raise AssertionError("An incomplete train split was admitted")
        finally:
            probe.MANIFEST_SHA = original_sha
    print(json.dumps({"status": "passed_stdlib_regression", "checks": [
        "fixed_manifest", "writable_data_copy", "changed_hash_rejected_before_App",
        "incomplete_split_rejected"], "SDK_or_NN_started": False}))


if __name__ == "__main__":
    main()
