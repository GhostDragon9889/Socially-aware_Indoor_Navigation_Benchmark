"""Register the full fresh episode candidate from actual completed S3-R evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import episode_serial_release as release  # noqa: E402
from scripts import own_shared_social_episode_pipeline as owner  # noqa: E402
from scripts import train_shared_social_episode_after_recovery as training  # noqa: E402
from scripts import run_shared_social_episode_after_recovery_evaluation as evaluation  # noqa: E402

HERE = Path(__file__).resolve().parent
if (HERE / "serial_release.json").exists() or (HERE / "owner_plan.json").exists():
    raise FileExistsError("Preserve existing candidate release and registration")
stages = release.stages()
if any(Path(row["output_directory"]).exists() for row in stages) or evaluation.admission.OUTPUT.exists():
    raise FileExistsError("Candidate output already exists")
serial_release = HERE / "serial_release.json"
owner.io.write_json(serial_release, release.release_document(), readonly=True)
preflight = {"schema": "shared_social.episode_after_recovery_actual_cold_preflight.v1", "steps": []}
commands = [(row["stage"], [sys.executable, str(Path(training.__file__)), "--serial-release", str(serial_release),
                          "--stage", row["stage"], "--check-sources"]) for row in stages]
commands.append(("evaluation", [sys.executable, str(Path(evaluation.__file__)), "--serial-release", str(serial_release), "--check-sources"]))
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
for name, command in commands:
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
    (HERE / (name + "_preflight.stdout")).write_text(completed.stdout)
    (HERE / (name + "_preflight.stderr")).write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Actual {name} cold preflight exit {completed.returncode}: {completed.stderr}")
    document = json.loads(completed.stdout)
    release.recovery.require(document.get("Torch_imported") is False and document.get("SDK_started") is False,
                             "Cold preflight unexpectedly imported Torch/SDK")
    preflight["steps"].append({"stage": name, "command": command, "actual_exit_code": completed.returncode,
        "SDK_started": False, "Torch_imported": False, "stdout": release.record(HERE / (name + "_preflight.stdout")),
        "stderr": release.record(HERE / (name + "_preflight.stderr"))})
preflight_path = HERE / "cold_preflight.json"
owner.io.write_json(preflight_path, preflight, readonly=True)
module, _ = training.original.prepare_entry()
rows = module._sources()
probe = module._load(module.PINS["probe_setup"][0], "_after_recovery_plan_probe", rows["probe_setup"])
setup_rows = probe._sources()
paths = [Path(owner.__file__), Path(training.__file__), Path(evaluation.__file__), Path(release.__file__),
         Path(owner.io.__file__), Path(release.consumers.__file__), Path(__file__), ROOT / "tests/test_shared_social_episode_after_recovery.py",
         ROOT / "scripts/audit_shared_social_popart.py", ROOT / "scripts/audit_shared_social_training.py",
         Path(evaluation.original.__file__), Path(evaluation.admission.__file__), evaluation.admission.REGISTRATION,
         evaluation.admission.PLAN, evaluation.admission.PROBE]
paths.extend(Path(row["path"]) for row in [*rows.values(), *setup_rows.values(),
             *release.release_document()["candidate_training_binding"]["World_dependencies"].values()])
paths = list(dict.fromkeys(paths))
sources = [release.record(path) for path in paths]
manifest = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
bank = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7_train.pt"
release.recovery.require(release.record(manifest)["sha256"] == "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca", "Fixed real2848 manifest changed")
bank_identity = release.record(bank)
release.recovery.require(bank_identity["sha256"] == "46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0",
                         "Fixed real2848 AMASS bank changed")
inputs = [release.record(serial_release), release.record(preflight_path), release.record(manifest), bank_identity]
inputs.extend(release.record(Path(row["profile"]["path"])) for row in stages)
plan = {"schema": owner.PLAN_SCHEMA, "recorded_at_utc": owner.now(), "serial_release": release.record(serial_release),
        "stages": stages, "sources": sources, "inputs": inputs, "cold_preflight": release.record(preflight_path),
        "GPU_resource_policy": owner.POLICY, "fixed_real_AMASS_clips": 2848,
        "reference_change": "reset-only procedural path ownership; unchanged AMASS31",
        "preselected_checkpoint": "step_002000.pt", "fresh_initialization": True,
        "formal_independent_evaluation": str(evaluation.admission.OUTPUT),
        "paper_effect_achieved": False, "default_model_promoted": False}
owner.io.write_json(HERE / "owner_plan.json", plan, readonly=True)
owner.admit_plan(HERE / "owner_plan.json")
snapshot_dir = HERE / "source_snapshots"
snapshot_dir.mkdir()
snapshots = []
for index, row in enumerate(sources):
    target = snapshot_dir / f"{index:03d}_{Path(row['path']).name}"
    shutil.copyfile(row["path"], target)
    target.chmod(0o444)
    copy = release.record(target)
    release.recovery.require((copy["sha256"], copy["bytes"]) == (row["sha256"], row["bytes"]), "Source snapshot mismatch")
    snapshots.append({"executed_source": row, "readonly_snapshot": copy})
owner.io.write_json(HERE / "source_snapshot_index.json", {"sources": snapshots}, readonly=True)
for path in HERE.glob("*_preflight.*"):
    path.chmod(0o444)
print(json.dumps({"status": "episode_after_recovery_pipeline_prepared", "actual_cold_preflights": len(commands),
                  "sources": len(sources), "inputs": len(inputs), "plan": release.record(HERE / "owner_plan.json"),
                  "Torch_imported": "torch" in sys.modules, "SDK_started": False}))
