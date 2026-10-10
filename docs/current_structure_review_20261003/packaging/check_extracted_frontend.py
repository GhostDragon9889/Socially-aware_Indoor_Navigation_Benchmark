"""Check extracted sdist frontends without external resources or SDK startup."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile

OUTPUT = Path(__file__).resolve().parent
report = json.loads((OUTPUT / "sdist_verification.json").read_text())
destination = Path(tempfile.mkdtemp(prefix="protomotions-current-sdist-extracted-"))
with tarfile.open(report["artifact"]["path"]) as archive:
    archive.extractall(destination, filter="data")
root = destination / "protomotions-3.1.0"
sys.path[:] = [str(root), *(value for value in sys.path if "/ProtoMotions" not in value)]
cases = []
for script in ("scripts/simulate_paper_pedestrians.py", "scripts/train_paper_amp.py"):
    original = sys.argv
    output = io.StringIO()
    try:
        sys.argv = [str(root / script), "--help"]
        with redirect_stdout(output):
            try:
                runpy.run_path(root / script, run_name="__main__")
            except SystemExit as error:
                assert error.code == 0
    finally:
        sys.argv = original
    assert "usage:" in output.getvalue()
    cases.append({"path": script, "help_return_code": 0})
module = sys.modules["CrowdSim.humanoid.application.configuration"]
assert Path(module.__file__).is_relative_to(root)
profile = module.load_amp_profile(root / "configs/diffusion_amp/amp_style_mixed_state_action_terrain_protected.yaml")
assert profile["training"]["task_terrain_statistics_mode"] == "preserve_declared_suffix"
heavy = sorted(name for name in sys.modules if name.split(".")[0] in {
    "torch", "numpy", "isaacsim", "isaaclab", "omni", "warp", "pxr"})
assert not heavy
result = {"status": "passed", "extracted_path": str(root), "cases": cases,
    "configuration_module": module.__file__, "protected_profile_loaded": True,
    "heavy_modules_imported": heavy,
    "scope": "Actual extracted-sdist cold help/profile only; required external licensed data absent by design; no runtime execution"}
(OUTPUT / "extracted_frontend_checks.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
