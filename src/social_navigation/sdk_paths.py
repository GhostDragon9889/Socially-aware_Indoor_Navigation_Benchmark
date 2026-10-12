"""Locate installed SDK files without importing Torch, pxr, or starting Kit."""
from importlib import metadata, util
import os
import json
from pathlib import Path

from protomotions.repository import repository_root

# Portable resource namespaces; machine locations are supplied at runtime.
LEGACY_PROJECT = PUBLISHED_PROJECT = "project:"
LEGACY_LAB = PUBLISHED_LAB = "lab:"


def lab_root():
    override = os.environ.get("ISAACLAB_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
    else:
        spec = util.find_spec("isaaclab")
        if spec is None or not spec.submodule_search_locations:
            binding = repository_root() / '.runtime/external_sdk.json'
            if spec is None and binding.is_file():
                path = Path(json.loads(binding.read_text())['lab_root']).resolve()
                if (path / 'source/isaaclab/isaaclab/__init__.py').is_file():
                    return path
            raise RuntimeError("IsaacLab source checkout not found; run scripts/bootstrap_reproduction.py")
        candidates = []
        for location in spec.submodule_search_locations:
            package = Path(location).resolve()
            if len(package.parents) >= 3:
                candidate = package.parents[2]
                if (candidate / "source/isaaclab/isaaclab").resolve() == package and (package / "__init__.py").is_file():
                    candidates.append(candidate)
        if len(set(candidates)) != 1:
            raise RuntimeError("IsaacLab import path does not identify one source checkout; check ISAACLAB_ROOT and PYTHONPATH")
        path = candidates[0]
    if not (path / "source/isaaclab/isaaclab").is_dir():
        raise RuntimeError("ISAACLAB_ROOT must identify the pinned source checkout")
    return path


def sim_root():
    return Path(metadata.distribution("isaacsim").locate_file("isaacsim")).resolve()


def relocate_sdk_path(value, *, root=None):
    """Resolve portable references and recorded checkout names without host paths."""
    value = str(value)
    root = Path(root or repository_root()).resolve()
    for prefix, provider in (("project:", lambda: root), ("lab:", lab_root), ("sim:", sim_root)):
        if value.startswith(prefix):
            relative = Path(value[len(prefix):].lstrip("/"))
            if ".." in relative.parts:
                raise ValueError("Resource reference contains parent traversal")
            return str(provider() / relative)
    path = Path(value)
    if path.is_absolute():
        if path.is_relative_to(root):
            return value
        # Read old serialized metadata by complete directory components only.
        # No installation location from the original workstation is embedded.
        for name in ("IsaacLab", "IsaacLab-Protomotions"):
            if name in path.parts:
                relative = Path(*path.parts[path.parts.index(name) + 1:])
                parts = relative.parts
                if "site-packages" in parts and "isaacsim" in parts:
                    return str(sim_root() / Path(*parts[parts.index("isaacsim") + 1:]))
                return str(lab_root() / relative)
        for name in ("ProtoMotions", "Socially-aware_Indoor_Navigation_Benchmark"):
            if name in path.parts:
                return str(root / Path(*path.parts[path.parts.index(name) + 1:]))
    return value


def require_cuda_driver():
    """Check driver access without importing Torch before Isaac Sim."""
    import ctypes
    try:
        driver = ctypes.CDLL('libcuda.so.1')
        code = driver.cuInit(0)
        count = ctypes.c_int()
        if code == 0:
            code = driver.cuDeviceGetCount(ctypes.byref(count))
        if code != 0 or count.value < 1:
            raise RuntimeError(f'CUDA driver unavailable (code={code}, devices={count.value}); Kit was not started')
    except OSError as error:
        raise RuntimeError('CUDA driver library unavailable; Kit was not started') from error
