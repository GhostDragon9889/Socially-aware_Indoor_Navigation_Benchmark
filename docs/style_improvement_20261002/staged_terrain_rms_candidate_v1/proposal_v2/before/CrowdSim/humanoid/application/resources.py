"""SDK-free local-file preflight; no model deserialization or downloads.

CLI resource paths retain cwd semantics. Package paths embedded in a motion
manifest are resolved relative to that manifest, matching its integrity gate.
Presence/header checks do not establish checkpoint architecture compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET

from .configuration import load_yaml_mapping


PACER_CHECKPOINT = "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


class ResourceError(ValueError):
    """A configured required local resource is missing or inconsistent."""


def resolve_local_path(value, *, relative_to=None):
    """Resolve explicit paths, without replacing a wrong path with a fallback."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ResourceError("A local resource needs an explicit nonempty path")
    if "://" in str(value):
        raise ResourceError(f"Preflight requires local resources, not remote URLs: {value}")
    path = Path(value)
    if not path.is_absolute() and relative_to is not None:
        path = Path(relative_to) / path
    return path.resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResourceRecord:
    name: str
    path: str
    bytes: int
    mtime_ns: int
    header_kind: str
    declared_sha256: str | None = None
    sha256_recomputed: bool = False


@dataclass(frozen=True)
class ResourceReport:
    records: tuple[ResourceRecord, ...]
    split_clip_counts: dict[str, int]
    working_directory: str
    optional_capture_textures_present: bool

    def require(self, name):
        for record in self.records:
            if record.name == name:
                return Path(record.path)
        raise ResourceError(f"Resource was not selected by this run: {name}")

    def as_dict(self):
        return {
            "status": "local_file_preflight_passed",
            "records": [asdict(record) for record in self.records],
            "split_clip_counts": dict(self.split_clip_counts),
            "working_directory": self.working_directory,
            "optional_capture_textures_present": self.optional_capture_textures_present,
            "sdk_started": False, "torch_imported_by_preflight": False,
            "models_deserialized": False, "author_experiment_reproduced": False,
            "limits": "Presence, header, size and selected checksums only; actual tensor IDs, FK and model/config compatibility retain their dedicated runtime gates. Optional texture presence is an inventory check, not UV/image validation.",
        }


def inspect_resource(name, path, *, kind="opaque", expected_bytes=None,
                     declared_sha256=None, recompute_sha256=False):
    path = resolve_local_path(path)
    if not path.is_file():
        raise ResourceError(f"Required {name} is missing or not a file: {path}")
    stat = path.stat()
    if stat.st_size == 0:
        raise ResourceError(f"Required {name} is empty: {path}")
    if expected_bytes is not None and stat.st_size != expected_bytes:
        raise ResourceError(f"Required {name} byte size changed: {path}; {stat.st_size} != {expected_bytes}")
    with path.open("rb") as stream:
        header = stream.read(160)
    if header.startswith(LFS_HEADER):
        raise ResourceError(f"Required {name} is an unresolved Git LFS pointer: {path}")
    if kind == "torch":
        if header.startswith(b"PK\x03\x04"):
            header_kind = "torch_zip_signature_only"
        elif header[:1] == b"\x80" and header[1:2] in (b"\x02", b"\x03", b"\x04", b"\x05"):
            header_kind = "legacy_pickle_signature_only"
        else:
            raise ResourceError(f"Required {name} has no recognized Torch container signature: {path}")
    else:
        header_kind = "nonempty_local_file"
    if declared_sha256 is not None:
        if not isinstance(declared_sha256, str) or len(declared_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in declared_sha256
        ):
            raise ResourceError(f"Required {name} has invalid declared SHA256")
        if recompute_sha256 and file_sha256(path) != declared_sha256:
            raise ResourceError(f"Required {name} checksum changed: {path}")
    elif recompute_sha256:
        raise ResourceError("Checksum verification requires an expected SHA256")
    return ResourceRecord(name, str(path), stat.st_size, stat.st_mtime_ns,
                          header_kind, declared_sha256, recompute_sha256)


def _manifest_resources(manifest_path):
    try:
        document = json.loads(manifest_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResourceError(f"Cannot read AMASS manifest {manifest_path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != "navisaaclab.amass_reconstruction.v1":
        raise ResourceError("AMASS manifest must declare navisaaclab.amass_reconstruction.v1")
    splits = document.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ResourceError("AMASS manifest must explicitly declare train, validation and test splits")
    records, counts, split_sources = [], {}, {}
    for split, bundle in splits.items():
        if not isinstance(bundle, dict):
            raise ResourceError(f"AMASS {split} bundle must be a mapping")
        clips = bundle.get("clips")
        count = bundle.get("clip_count", len(clips) if isinstance(clips, list) else None)
        if not isinstance(clips, list) or not clips or type(count) is not int or count != len(clips):
            raise ResourceError(f"AMASS {split} requires nonempty clips and a consistent count")
        if type(bundle.get("bytes")) is not int or bundle["bytes"] <= 0:
            raise ResourceError(f"AMASS {split} requires a positive package byte count")
        if bundle.get("sha256") is None:
            raise ResourceError(f"AMASS {split} requires a declared source SHA256")
        package = resolve_local_path(bundle.get("path"), relative_to=manifest_path.parent)
        records.append(inspect_resource(f"amass_{split}", package, kind="torch",
                                       expected_bytes=bundle["bytes"], declared_sha256=bundle["sha256"]))
        ids, sources, parents = [], [], []
        for clip in clips:
            if not isinstance(clip, dict) or type(clip.get("motion_id")) is not int or clip["motion_id"] < 0:
                raise ResourceError(f"AMASS {split} contains an invalid clip ID")
            source = clip.get("source_path")
            if not isinstance(source, str) or not source:
                raise ResourceError(f"AMASS {split} contains an empty source identity")
            parent = clip.get("parent_source_path", source)
            if not isinstance(parent, str) or not parent:
                raise ResourceError(f"AMASS {split} contains an empty parent source identity")
            parent = posixpath.normpath(parent.replace("\\", "/").split("#frames=", 1)[0])
            if parent != posixpath.normpath(source.replace("\\", "/").split("#frames=", 1)[0]):
                raise ResourceError(f"AMASS {split} parent identity disagrees with its clip source")
            ids.append(clip["motion_id"])
            sources.append(posixpath.normpath(source.replace("\\", "/")))
            parents.append(parent)
        if len(set(ids)) != len(ids) or len(set(sources)) != len(sources):
            raise ResourceError(f"AMASS {split} contains duplicate clip IDs or source identities")
        counts[split] = len(clips)
        split_sources[split] = set(parents)
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_sources[first] & split_sources[second]:
            raise ResourceError(f"AMASS source identity leakage between {first} and {second}")
    return records, counts


def _scene_shape_gate(path, num_envs):
    document = load_yaml_mapping(path)
    if document.get("schema") == "crowdsim.native_pedestrian_scene.v1":
        from .native_deployment import preflight_native_scene
        try:
            preflight_native_scene(path, actors=num_envs)
        except (ValueError, OSError, KeyError) as error:
            raise ResourceError(f"Native shared scene resource gate failed: {error}") from error
        return
    if document.get("schema") != "navisaaclab.reconstruction_scene.v1":
        raise ResourceError("Shared scene requires navisaaclab.reconstruction_scene.v1")
    origins, routes = document.get("origins_xy"), document.get("routes_xy")
    if not isinstance(origins, list) or not isinstance(routes, list) or len(origins) != num_envs or len(routes) != num_envs:
        raise ResourceError("Shared scene needs one explicit spawn and route per configured actor")
    bounds = document.get("bounds_xy")
    if not isinstance(bounds, list) or len(bounds) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds):
        raise ResourceError("Shared scene bounds must contain four finite world coordinates")
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ResourceError("Shared scene bounds must have positive area")
    for points in [[point] for point in origins] + routes:
        if not isinstance(points, list) or not points:
            raise ResourceError("Shared scene needs nonempty finite routes/spawns")
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
                raise ResourceError("Shared scene routes/spawns require finite XY points")
            if not (bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]):
                raise ResourceError("Shared scene spawn/route is outside the declared world bounds")
    if any(len(route) < 2 for route in routes):
        raise ResourceError("Shared scene needs at least two route vertices per actor")
    boxes = document.get("cuboids", [])
    if not isinstance(boxes, list):
        raise ResourceError("Shared scene cuboids must be an explicit list")
    for box in boxes:
        if not isinstance(box, dict):
            raise ResourceError("Shared scene cuboids must contain center/size mappings")
        for key in ("center", "size"):
            values = box.get(key)
            if not isinstance(values, list) or len(values) != 3 or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in values
            ):
                raise ResourceError(f"Shared scene cuboid {key} must contain three finite values")
        if any(value <= 0 for value in box["size"]):
            raise ResourceError("Shared scene cuboid sizes must be positive")
    for key, default in (("arrival_radius_m", .75), ("goal_lookahead_m", 5.)):
        value = document.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ResourceError(f"Shared scene {key} must be positive and finite")


def _capture_resources():
    """Mirror the visualizer's explicit model override and neutral-file order.

    This stays pure Python. The safe numeric SMPL decoder, UV checks and image
    loading remain in the actual visualizer after SDK startup.
    """
    default = Path(__file__).resolve().parents[3] / "data/smpl"
    model_dir = Path(os.environ.get("CROWDSIM_SMPL_MODEL_DIR", str(default))).expanduser()
    candidates = [model_dir] if model_dir.is_file() else [
        model_dir / "smpl/SMPL_NEUTRAL.pkl", model_dir / "SMPL_NEUTRAL.pkl"
    ]
    model = next((path for path in candidates if path.exists()), candidates[-1])
    texture_dir = Path(os.environ.get("CROWDSIM_SMPL_TEXTURE_DIR", str(model_dir))).expanduser()
    explicit = os.environ.get("CROWDSIM_SMPL_TEXTURE_PATH")
    images = [Path(explicit).expanduser()] if explicit else []
    if texture_dir.is_dir():
        images.extend(path for path in texture_dir.rglob("*") if path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".tif", ".tiff"
        })
    uv = model_dir / "smpl_uv.npz"
    optional_present = uv.is_file() and uv.stat().st_size > 0 and any(
        path.is_file() and path.stat().st_size > 0 for path in images
    )
    return model, optional_present


def preflight_amp_resources(args, profile, *, verify_warm_start_checksum=True):
    """Check files before Kit; do not replace the tensor/model runtime gates."""
    from .configuration import validate_training_source
    validate_training_source(args, profile)
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise ResourceError("Run seed must be an integer in the NumPy-compatible range [0,2**32)")
    if args.from_scratch and profile["training"].get("warm_start_discriminator", False):
        raise ResourceError("From-scratch training conflicts with the declared pretrained discriminator; choose an explicit random-discriminator profile")
    records = [inspect_resource("profile", args.config),
               inspect_resource("manifest", args.manifest),
               inspect_resource("diffusion_checkpoint", args.diffusion_checkpoint, kind="torch")]
    manifest_records, counts = _manifest_resources(resolve_local_path(args.manifest))
    records.extend(manifest_records)
    if args.resume:
        records.append(inspect_resource("controller_checkpoint", args.resume, kind="torch"))
    elif not args.from_scratch:
        initialization = profile.get("initialization", {})
        checkpoint = initialization.get("checkpoint", PACER_CHECKPOINT)
        declared = initialization.get("checkpoint_sha256")
        records.append(inspect_resource("warm_start_checkpoint", checkpoint, kind="torch",
                                       declared_sha256=declared,
                                       recompute_sha256=bool(verify_warm_start_checksum and declared)))
    from protomotions.assets import get_asset_root
    rig = get_asset_root() / "mjcf/smpl_humanoid.xml"
    records.append(inspect_resource("smpl_physics_rig", rig))
    try:
        document = ET.parse(rig)
    except ET.ParseError as error:
        raise ResourceError(f"SMPL physics rig is malformed XML: {rig}") from error
    if document.getroot().tag != "mujoco":
        raise ResourceError(f"SMPL physics rig must be an MJCF document: {rig}")
    if args.scene_config:
        scene = resolve_local_path(args.scene_config)
        records.append(inspect_resource("shared_scene", scene))
        _scene_shape_gate(scene, args.num_envs)
    optional_textures = False
    if args.capture:
        model, optional_textures = _capture_resources()
        records.append(inspect_resource("smpl_visual_model", model))
    return ResourceReport(tuple(records), counts, str(Path.cwd()), bool(optional_textures))


__all__ = ["ResourceError", "ResourceRecord", "ResourceReport", "resolve_local_path",
           "inspect_resource", "preflight_amp_resources"]
