"""Check the actual new sdist in /tmp with no production module fallback."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile

OUTPUT = Path(__file__).resolve().parent
PRODUCTION = OUTPUT.parents[2]
HEAVY = {"torch", "numpy", "isaacsim", "isaaclab", "omni", "warp", "pxr"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heavy_modules():
    return sorted(name for name in sys.modules if name.split(".")[0] in HEAVY)


def main():
    report = json.loads((OUTPUT / "sdist_verification.json").read_text())
    artifact = Path(report["artifact"]["path"])
    require(artifact.stat().st_size == report["artifact"]["bytes"] and
            sha256(artifact) == report["artifact"]["sha256"], "Built archive changed")
    destination = Path(tempfile.mkdtemp(prefix="protomotions-v11-sdist-extracted-"))
    with tarfile.open(artifact) as archive:
        archive.extractall(destination, filter="data")
    root = destination / "protomotions-3.1.0"
    # Preserve installed dependencies, removing every source-checkout import path
    # and the empty cwd sentinel, including when resource cwd is changed later.
    sys.path[:] = [str(root), *(value for value in sys.path if value and
                    not Path(value).resolve().is_relative_to(PRODUCTION))]
    require(not heavy_modules(), "Heavy module already loaded")
    os.chdir(root)
    cases = []
    for script in ("scripts/simulate_paper_pedestrians.py", "scripts/train_paper_amp.py"):
        original = sys.argv
        stdout, stderr = io.StringIO(), io.StringIO()
        return_code = None
        try:
            sys.argv = [str(root / script), "--help"]
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    runpy.run_path(str(root / script), run_name="__main__")
                    return_code = 0
                except SystemExit as error:
                    return_code = error.code
        finally:
            sys.argv = original
        require(return_code == 0 and "usage:" in stdout.getvalue(), "Help failed: " + script)
        require(not stderr.getvalue(), "Help emitted an error: " + script)
        name = Path(script).stem + "_help.txt"
        (OUTPUT / name).write_text(stdout.getvalue())
        cases.append({"path": script, "help_return_code": return_code,
                      "stdout_artifact": name, "stdout_sha256": sha256(OUTPUT / name)})

    from CrowdSim.humanoid.application import configuration, preflight
    from CrowdSim.humanoid.application import state_prior_initialization
    profile_path = root / "configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml"
    profile = configuration.load_amp_profile(profile_path)
    training = profile["training"]
    require(training["warm_start_action_discriminator_from_state_prior"] is True,
            "New public state prior flag missing")
    require(training["warm_start_discriminator"] is False and
            training["task_terrain_statistics_mode"] == "preserve_declared_suffix" and
            training["discriminator_input"] == "state_action" and
            training["amp_normalization"] == "per_history_coordinate" and
            training["history_frames"] == 10 and training["amp_frame_dim"] == 195,
            "New profile coupled initialization contract changed")
    require(profile["initialization"]["discriminator"] ==
            "public_pacer_state_prior_zero_action_extension", "New strategy metadata missing")
    shared = ["--config", str(profile_path), "--output", str(destination / "unused-output"),
              "--diffusion-checkpoint", str(PRODUCTION / "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt"),
              "--trajectory-source", "mixed_amass_procedural", "--num-envs", "256",
              "--updates", "1", "--rollout-steps", "32", "--terrain-extent-m", "192",
              "--spawn-margin-m", "48", "--curriculum-speed-max-mps", "2",
              "--curriculum-accel-max-mps2", "1.5", "--synchronized-replanning", "--physics-only"]
    preflights = []
    for label, cwd, extra, expected in (
        ("packaged_manifest_absent", root, [], 2),
        ("from_scratch_rejected", root, ["--from-scratch"], 2),
        ("explicit_external_resources", PRODUCTION,
         ["--manifest", str(PRODUCTION / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json")], 0),
    ):
        os.chdir(cwd)
        if expected == 0:
            os.environ["PROTOMOTIONS_ASSET_ROOT"] = str(PRODUCTION / "protomotions/data/assets")
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = preflight.main(shared + extra)
            except SystemExit as error:
                raise ValueError(f"Preflight parser exited {error.code}: {stderr.getvalue()}") from error
        require(code == expected, f"Resource preflight {label} code {code} != {expected}: {stderr.getvalue()}")
        if label == "packaged_manifest_absent":
            require("Required manifest is missing" in stderr.getvalue(), "Missing-resource error changed")
        elif label == "from_scratch_rejected":
            require("from-scratch" in stderr.getvalue(), "Fresh prior from-scratch was accepted")
        else:
            parsed = json.loads(stdout.getvalue())
            resource = parsed["resource_preflight"]
            require(resource["status"] == "local_file_preflight_passed", "No resource success")
            require(resource["sdk_started"] is False and resource["models_deserialized"] is False,
                    "Resource preflight started a runtime")
            public = next(row for row in resource["records"] if row["name"] == "warm_start_checkpoint")
            require(public["declared_sha256"] == state_prior_initialization.PUBLIC_PACER_SHA256 and
                    public["sha256_recomputed"] is True, "Actual public prior checksum not verified")
        require(not heavy_modules(), "Resource preflight imported heavy modules")
        evidence = {"arguments": shared + extra, "working_directory": str(cwd),
                    "return_code": code, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
        evidence_path = OUTPUT / (label + ".json")
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        preflights.append({"case": label, "return_code": code,
                           "evidence": evidence_path.name, "sha256": sha256(evidence_path)})

    owned_modules = []
    for name, module in sorted(sys.modules.items()):
        if name.split(".")[0] not in {"CrowdSim", "protomotions"} or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(root), "Production module leaked into extracted check: " + name)
        owned_modules.append({"module": name, "path": str(path), "sha256": sha256(path)})
    result = {"schema": "navisaaclab.state_prior_v11_extracted_frontend_gate.v1", "status": "passed",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "extracted_path": str(root),
              "artifact": report["artifact"], "cases": cases, "new_profile_loaded": True,
              "new_profile_sha256": sha256(profile_path), "resource_preflight_cases": preflights,
              "configuration_module": configuration.__file__, "owned_modules": owned_modules,
              "heavy_modules_imported": heavy_modules(),
              "scope": "Actual extracted sdist help/profile and SDK-free local resource preflight. Positive resource check reads explicitly external original files; none is added to archive. No tensor deserialization, training, GPU or physics."}
    (OUTPUT / "extracted_frontend_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "extracted_path", "cases", "resource_preflight_cases", "heavy_modules_imported")}, indent=2))


if __name__ == "__main__":
    main()
