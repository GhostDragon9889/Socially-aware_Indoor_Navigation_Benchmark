"""Occupancy map metadata helpers for CrowdSim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

from CrowdSim.utils.config_loader import load_config


DEFAULT_MAP_RESOLUTION = 100.0 / 1999.0
DEFAULT_FREE_THRESHOLD = 200


@dataclass(frozen=True)
class OccupancyMapMetadata:
    image_path: Path
    metadata_path: Path | None
    resolution: float
    origin_xy: tuple[float, float]
    free_threshold: int
    negate: int = 0
    occupied_thresh: float | None = None
    free_thresh: float | None = None
    origin_yaw: float = 0.0


def load_occupancy_map_metadata(path_like: str | Path) -> OccupancyMapMetadata:
    """Load map image path, resolution, origin, and free threshold from a map YAML.

    If ``path_like`` points to an image, a sidecar YAML with the same stem is used
    when present. Otherwise the historical CrowdSim defaults are used.
    """
    path = Path(path_like).expanduser().resolve()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_from_yaml(path)

    sidecar = path.with_suffix(".yaml")
    if sidecar.exists():
        return _load_from_yaml(sidecar, image_override=path)

    return OccupancyMapMetadata(
        image_path=path,
        metadata_path=None,
        resolution=DEFAULT_MAP_RESOLUTION,
        origin_xy=_center_origin(path, DEFAULT_MAP_RESOLUTION),
        free_threshold=DEFAULT_FREE_THRESHOLD,
    )


def _load_from_yaml(
    yaml_path: Path,
    image_override: Path | None = None,
) -> OccupancyMapMetadata:
    # Share the config loader so valid YAML block sequences (as emitted by
    # map exporters) and historical inline origins have the same meaning.
    data = load_config(yaml_path)
    image_path = image_override or (yaml_path.parent / str(data["image"])).resolve()
    resolution = float(data["resolution"])
    origin = data.get("origin", [0.0, 0.0, 0.0])
    negate = int(data.get("negate", 0))
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("Occupancy map resolution must be positive and finite")
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        raise ValueError("Occupancy map origin must contain [x, y, yaw]")
    origin = tuple(float(value) for value in origin)
    if not all(math.isfinite(value) for value in origin):
        raise ValueError("Occupancy map origin must be finite")
    if negate not in (0, 1):
        raise ValueError("Occupancy map negate must be 0 or 1")
    free_thresh = data.get("free_thresh")
    occupied_thresh = data.get("occupied_thresh")
    if occupied_thresh is not None and (
        not math.isfinite(float(occupied_thresh)) or not 0.0 <= float(occupied_thresh) <= 1.0
    ):
        raise ValueError("Occupancy map occupied_thresh must lie in [0, 1]")
    if free_thresh is not None and occupied_thresh is not None and float(free_thresh) >= float(occupied_thresh):
        raise ValueError("Occupancy map free_thresh must be below occupied_thresh")
    free_threshold = _free_thresh_to_gray_threshold(
        free_thresh=float(free_thresh) if free_thresh is not None else None,
        negate=negate,
    )
    return OccupancyMapMetadata(
        image_path=Path(image_path).expanduser().resolve(),
        metadata_path=yaml_path,
        resolution=resolution,
        origin_xy=(float(origin[0]), float(origin[1])),
        free_threshold=free_threshold,
        negate=negate,
        occupied_thresh=float(occupied_thresh) if occupied_thresh is not None else None,
        free_thresh=float(free_thresh) if free_thresh is not None else None,
        origin_yaw=origin[2],
    )


def _free_thresh_to_gray_threshold(free_thresh: float | None, negate: int) -> int:
    """Return a gray boundary for free occupancy probability <= free_thresh.

    Non-negated images use gray >= threshold; negated images use <=. Gray cells
    between the free and occupied probability thresholds remain non-traversable.
    """
    if negate not in (0, 1):
        raise ValueError("Occupancy map negate must be 0 or 1")
    if free_thresh is None:
        return 255 - DEFAULT_FREE_THRESHOLD if negate else DEFAULT_FREE_THRESHOLD
    if not math.isfinite(free_thresh) or not 0.0 <= free_thresh <= 1.0:
        raise ValueError("Occupancy map free_thresh must lie in [0, 1]")
    if negate:
        threshold = math.floor(255.0 * free_thresh)
    else:
        threshold = math.ceil(255.0 - 255.0 * free_thresh)
    return int(threshold)


def _center_origin(image_path: Path, resolution: float) -> tuple[float, float]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return (0.0, 0.0)
    return (
        -0.5 * width * resolution,
        -0.5 * height * resolution,
    )

