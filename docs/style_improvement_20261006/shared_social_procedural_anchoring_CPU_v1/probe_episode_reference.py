"""Actual all2848 CPU check of the new episode reference primitive."""
from pathlib import Path
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_shared_social_training import identity  # noqa: E402
from scripts.diagnose_shared_social_procedural_anchoring import require  # noqa: E402

HERE = Path(__file__).resolve().parent
REPORT = HERE / "actual_bound/result.json"
require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only candidate probe")
report = json.loads(REPORT.read_text())
inputs = [identity(REPORT), *report["inputs"], *report["sources"], *report["artifacts"]]
require(all(identity(row["path"]) == row for row in inputs), "Original bound diagnostic changed")
sources = [identity(path) for path in (Path(__file__), ROOT / "CrowdSim/humanoid/training/episode_trajectories.py")]

import torch  # noqa: E402
from CrowdSim.humanoid.training.episode_trajectories import EpisodeProceduralReference  # noqa: E402

torch.set_num_threads(4)
require(not torch.cuda.is_initialized(), "CUDA initialized")
global_rng = torch.random.get_rng_state().clone()
with np.load(report["artifacts"][0]["path"], allow_pickle=False) as saved:
    require(np.array_equal(saved["motion_ids"], np.arange(2848))
        and np.array_equal(saved["source_frames"], np.full(2848, 9))
        and not saved["declared_root_vel"].any(), "Require all original stationary probe roots")
    state = SimpleNamespace(root_pos=torch.from_numpy(saved["initial_root_pos"].copy()),
        root_rot=torch.from_numpy(saved["initial_root_rot_xyzw"].copy()), root_vel=torch.from_numpy(saved["declared_root_vel"].copy()))
    original = torch.from_numpy(saved["canonical_generated_plans"][0].copy())
    original_targets = torch.from_numpy(saved["target_xy"][2].copy())
before = {key: value.clone() for key, value in vars(state).items()}
reference = EpisodeProceduralReference(2848)
episodes = torch.ones(2848, dtype=torch.long)
reference.reset(state, episodes)
require(torch.equal(reference.paths[:, :51], original), "Original first five seconds changed")
private_rng = reference.source.generator.get_state().clone()
queries = torch.stack([reference.query(torch.full((2848,), step), episodes, [0.])[:, 0]
                       for step in range(1, 151)])
require(torch.equal(queries, original_targets), "Persistent reference differs from the isolated episode-clock target")
future = reference.query(torch.full((2848,), 300), episodes, torch.arange(1, 11).float() * .5)
require(torch.equal(future[:, -1], reference.paths[:, -1])
    and torch.equal(private_rng, reference.source.generator.get_state())
    and all(torch.equal(getattr(state, key), value) for key, value in before.items()), "Queries changed RNG/state or final endpoint")
restored = EpisodeProceduralReference(2848, seed=999)
restored.load_state_dict(reference.state_dict())
require(torch.equal(restored.query(torch.full((2848,), 300), episodes, torch.arange(1, 11).float() * .5), future), "Restored reference changed")
require(torch.equal(global_rng, torch.random.get_rng_state()) and not torch.cuda.is_initialized()
    and not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "RNG changed or SDK/CUDA initialized")
require(all(identity(row["path"]) == row for row in inputs + sources), "Input/source changed during probe")
output = HERE / "episode_reference_candidate.npz"
require(not output.exists(), "Fresh candidate output required")
np.savez_compressed(output, paths=reference.paths.numpy(), first150_post_targets=queries.numpy(),
    final_episode_future10=future.numpy(), episode_ids=episodes.numpy(), motion_ids=np.arange(2848, dtype=np.int64))
result = {"schema": "shared_social.episode_reference_candidate_CPU.v1", "status": "actual_all2848_CPU_probe_completed",
    "inputs": inputs, "sources": sources, "artifact": identity(output), "config": reference.config,
    "actors": 2848, "original_first51_exact": True, "all_first150_post_targets_exact": True,
    "query_target_rows": 2848 * 160, "query_draws_no_RNG": True, "physical_root_fields_unchanged": True,
    "reference_state_roundtrip_exact": True, "CUDA_initialized": False, "SDK_imported": False,
    "World_integrated": False, "SDK_curriculum_history_resume_proven": False, "quality_improvement_proven": False}
(HERE / "episode_reference_candidate.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("status", "actors", "query_target_rows", "original_first51_exact", "World_integrated")}, indent=2))
