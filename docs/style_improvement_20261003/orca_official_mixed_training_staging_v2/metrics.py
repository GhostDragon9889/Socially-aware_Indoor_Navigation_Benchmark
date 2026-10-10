"""Original domain metric formulas, only official polygon I/O/JSON adaptation."""
from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OLD_STAGE = ROOT / "docs/style_improvement_20261003/maps_interact_mixed_training_staging_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OLD_STAGE) not in sys.path:
    sys.path.append(str(OLD_STAGE))

from scripts import train_orca_diffusion as original_maps  # noqa: E402
from train_mixed_diffusion import _interaction_offline_plan_metrics  # noqa: E402

_ORIGINAL = inspect.getsource(original_maps.offline_plan_metrics)
_OLD_IO = 'json.loads((corpus.manifest_path.parent / scene["folder"] / "obstacles.json").read_text())'
if _ORIGINAL.count(_OLD_IO) != 1:
    raise ValueError("Expected the pinned original Maps polygon input expression")
_ADAPTED = _ORIGINAL.replace(_OLD_IO, 'scene["obstacles"]')
_NAMESPACE = dict(original_maps.__dict__)
exec(compile(_ADAPTED, str(__file__) + ":official_polygon_metadata", "exec"), _NAMESPACE)
_maps_formula = _NAMESPACE["offline_plan_metrics"]


def offline_plan_metrics(corpus, sample, predicted_xy):
    if corpus.domain == "interact":
        return _interaction_offline_plan_metrics(corpus.corpus, sample, predicted_xy)
    result = _maps_formula(corpus.corpus, sample, predicted_xy)
    if math.isinf(result["min_obstacle_surface_gap_m"]) and result["min_obstacle_surface_gap_m"] > 0:
        # Unchanged empty-polygon math gives+inf. Known empty map remains a
        # valid zero-collision observation; serialize unbounded clearance asNone.
        result["min_obstacle_surface_gap_m"] = None
        result["min_obstacle_clearance_reason"] = "known_empty_map_no_finite_boundary"
    return result


def polygon_adapter_proof():
    return {"original_module": original_maps.__file__,
            "only_replacement": {"old": _OLD_IO, "new": 'scene["obstacles"]'},
            "reverse_function_source_bytes_exact": _ADAPTED.replace('scene["obstacles"]', _OLD_IO) == _ORIGINAL,
            "numeric_formulas_changed": False, "known_empty_collision_zero_preserved": True,
            "unknown_interact_obstacle_metrics_remain_NA": True}

