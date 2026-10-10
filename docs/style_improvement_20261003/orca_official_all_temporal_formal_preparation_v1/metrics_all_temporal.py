"""Original domain metrics plus explicit known-empty recorded-peer semantics.

No model execution. Multi-agent numeric formulas and the all-ego-point collision
denominator are unchanged. Single-agent windows remain in every metric batch.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FORMAL = ROOT / "docs/style_improvement_20261003/orca_official_mixed_training_staging_v2"
if str(FORMAL) not in sys.path:
    sys.path.append(str(FORMAL))

from metrics import original_maps, _interaction_offline_plan_metrics  # noqa: E402


def derive(function, namespace, replacements):
    before = inspect.getsource(function)
    after = before
    for old, new in replacements:
        if after.count(old) != 1:
            raise ValueError("Expected the pinned metric source expression")
        after = after.replace(old, new, 1)
    inverse = after
    for old, new in reversed(replacements):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous inverse metric source delta")
        inverse = inverse.replace(new, old, 1)
    if inverse != before:
        raise ValueError("Original metric source bytes differ")
    scope = dict(namespace)
    exec(compile(after, str(__file__) + ":known_empty_peers", "exec"), scope)
    return scope[function.__name__]


MAP_PEERS = 'pair_gaps.append(np.linalg.norm(predicted_xy[row][None] - others, axis=-1).min(axis=0) - .8)'
MAP_EMPTY = 'pair_gaps.append(np.full(predicted_xy.shape[1], np.inf) if len(others) == 0 else np.linalg.norm(predicted_xy[row][None] - others, axis=-1).min(axis=0) - .8)'
MAP_POLYGONS = 'json.loads((corpus.manifest_path.parent / scene["folder"] / "obstacles.json").read_text())'
INTERACT_PEERS = 'pair_gaps.append((gaps - (radii[agent] + np.delete(radii, agent))[:, None]).min(axis=0))'
INTERACT_EMPTY = 'pair_gaps.append(np.full(predicted_xy.shape[1], np.inf) if len(others) == 0 else (gaps - (radii[agent] + np.delete(radii, agent))[:, None]).min(axis=0))'

_maps = derive(original_maps.offline_plan_metrics, original_maps.__dict__,
               [(MAP_POLYGONS, 'scene["obstacles"]'), (MAP_PEERS, MAP_EMPTY)])
_interact = derive(_interaction_offline_plan_metrics,
                   _interaction_offline_plan_metrics.__globals__,
                   [(INTERACT_PEERS, INTERACT_EMPTY)])


def offline_plan_metrics(corpus, sample, predicted_xy):
    core = corpus.corpus
    if corpus.domain == "maps":
        result = _maps(core, sample, predicted_xy)
        if result["min_obstacle_surface_gap_m"] == math.inf:
            result["min_obstacle_surface_gap_m"] = None
            result["min_obstacle_clearance_reason"] = "known_empty_map_no_finite_boundary"
    elif corpus.domain == "interact":
        result = _interact(core, sample, predicted_xy)
    else:
        raise ValueError("Expected the registered Maps or Interact domain")
    other_counts = [len(core.episodes[core.windows[index][0]]["states"]) - 1
                    for index in np.asarray(sample["window_indices"])]
    result["recorded_peer_observations"] = {
        "all_ego_future_points_collision_denominator": len(other_counts) * 50,
        "windows_with_recorded_other_agents": sum(count > 0 for count in other_counts),
        "known_no_other_agent_windows": sum(count == 0 for count in other_counts),
        "recorded_other_agent_point_comparisons": sum(other_counts) * 50,
        "known_empty_windows_retained_in_denominator": True,
        "conditioning_radius_restricts_metric_peers": False,
    }
    if result["min_neighbor_surface_gap_m"] == math.inf and not any(other_counts):
        result["min_neighbor_surface_gap_m"] = None
        result["min_neighbor_clearance_reason"] = "known_no_other_recorded_agents"
    return result
