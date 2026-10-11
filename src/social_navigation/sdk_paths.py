"""Locate installed SDK files without importing Torch, pxr, or starting Kit."""
from importlib import metadata, util
import os
from pathlib import Path

from protomotions.repository import repository_root

LEGACY_PROJECT = "/project/EXP/SangforAndSaut/ProtoMotions"
LEGACY_LAB = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions"


def lab_root():
    override = os.environ.get("ISAACLAB_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
    else:
        spec = util.find_spec("isaaclab")
        if spec is None or not spec.submodule_search_locations:
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
    value = str(value)
    root = Path(root or repository_root()).resolve()
    site_prefix = LEGACY_LAB + "/.venv/lib/python3.12/site-packages/isaacsim/"
    if value.startswith(site_prefix):
        return str(sim_root() / value[len(site_prefix):])
    if value.startswith(LEGACY_LAB + "/"):
        return str(lab_root() / value[len(LEGACY_LAB) + 1:])
    if value.startswith(LEGACY_PROJECT + "/"):
        return str(root / value[len(LEGACY_PROJECT) + 1:])
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
