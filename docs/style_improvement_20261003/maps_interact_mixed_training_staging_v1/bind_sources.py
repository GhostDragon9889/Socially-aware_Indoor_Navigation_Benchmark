"""Metadata-only source binding, timing table and prepared command."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


files = ["mixed_corpus.py", "train_mixed_diffusion.py", "build_trainer.py", "trainer_derivation.json", "README.md"]
reused = [ROOT / "CrowdSim/humanoid/orca_corpus.py", ROOT / "CrowdSim/humanoid/diffusion/paper.py",
          ROOT / "CrowdSim/humanoid/amp_world.py", ROOT / "CrowdSim/humanoid/training/reference.py",
          ROOT / "scripts/train_paper_amp.py", ROOT / "scripts/train_orca_diffusion.py",
          HERE.parent / "orca_interact_usable_staging_v1/interaction_corpus.py",
          HERE.parent / "orca_interact_usable_staging_v1/paper_interact.py",
          HERE.parent / "orca_interact_usable_staging_v1/train_interact_diffusion.py",
          ROOT / "third_party/trace/configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json",
          ROOT / "third_party/trace/tbsim/datasets/trajdata_datamodules.py"]
paths = {name: ROOT / f"data/diffusion_amp/{folder}/manifest.json"
         for name, folder in (("maps", "orca_maps_expanded_1000_v1"), ("interact", "orca_interact_usable_1000_v1"))}
counts = {}
for name, path in paths.items():
    manifest = json.loads(path.read_text())
    for split, entry in manifest["splits"].items():
        counts.setdefault(split, {})[name] = sum(episode["conditional_window_count"]
                                               for scene in entry["scenes"]
                                               for episode in scene["episodes"])
for entry in counts.values():
    entry["total"] = entry["maps"] + entry["interact"]
    entry["interact_uniform_window_probability"] = entry["interact"] / entry["total"]
document = {"schema": "local_orca.maps_interact_staged_sources.v1",
            "status": "source_staged_awaiting_root_review_and_actual_CPU_model_smoke",
            "source_files": [record(HERE / name) for name in files],
            "reused_sources": [record(path) for path in reused],
            "dataset_manifests": {name: record(path) for name, path in paths.items()},
            "window_counts_from_metadata_only": counts,
            "sampling": "uniform_over_union_valid_windows_with_replacement",
            "domain_order": ["maps", "interact"], "max_neighbors": 19,
            "missing_map_encoding": {"default": "trace_unknown_raster", "ablation": "skip_encoder",
                                     "data_presence_saved": True, "encoder_availability_separate": True},
            "official_TRACE_commit": "594b2af0824c5853e1a1884e167cd5ed097dd81c",
            "new_model_or_dataset_read": False,
            "production_or_frozen_source_modified": False,
            "actual_model_smoke_completed": False, "actual_training_started": False,
            "future_checkpoint_sha256": None,
            "planned_training_command": ["python", str(HERE / "train_mixed_diffusion.py"),
                "--maps-manifest", str(paths["maps"]), "--interact-manifest", str(paths["interact"]),
                "--missing-map-encoding", "trace_unknown_raster", "--init-checkpoint",
                str(ROOT / "output/diffusion_amp/diffusion_training/step_1000.ckpt"),
                "--output-dir", str(ROOT / "output/diffusion_amp/social_diffusion_maps_interact_staged_v1"),
                "--steps", "1000", "--batch-size", "8", "--seed", "20261001", "--device", "cuda"]}
(HERE / "source_binding.json").write_text(json.dumps(document, indent=2) + "\n")
print(json.dumps({"status": document["status"], "window_counts": counts,
                  "binding": record(HERE / "source_binding.json")}))
text = (HERE.parent / "paper_fidelity_research_v1/trace_and_pace_text.txt").read_text()
for phrase in ("points for the next 5 seconds", "During training, the neighbor history"):
    index = text.index(phrase)
    page = text[:index].count("\f") + 1
    print(json.dumps({"phrase": phrase, "PDF_page_1_based": page,
                      "trailing_page_text": text[:index].rsplit("\f", 1)[0][-100:]}))
