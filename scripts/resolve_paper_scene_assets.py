"""Call from the parent's already running Kit; enable storage.native there.

This module does not import or construct SimulationApp. The public storage
API determines the actual configured root; no endpoint is hardcoded here.
Returned URLs are an internal download request, not availability assertions.
"""
from pathlib import Path
import datetime
import json
from urllib.parse import urlparse


ASSETS = {
    "Forklift": "Isaac/Props/Forklift/forklift.usd",
    "KLT_Bin": "Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd",
}
SOURCES = {
    "Forklift": "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_scene_based_sdg.html",
    "KLT_Bin": "https://forums.developer.nvidia.com/t/why-physicsmaterial-destroys-the-klt-bin/248866",
}


def resolve_requests(output_path="data/scene_resources/dynbench_missing/official_api_request.json"):
    """Resolve names through the actual live public API, without a download."""
    from omni.kit.app import get_app

    app = get_app()
    if app is None:
        raise RuntimeError("Asset resolution requires the parent's existing live Kit application")
    manager = app.get_extension_manager()
    if not manager.set_extension_enabled_immediate("isaacsim.storage.native", True):
        raise RuntimeError("The existing Kit could not enable isaacsim.storage.native")
    from isaacsim.storage.native import get_assets_root_path, path_join

    root = get_assets_root_path(skip_check=True)
    parsed = urlparse(root)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Authenticated/query-bearing asset roots must not be written into this public download request")
    request = {"schema": "navisaaclab.official_asset_request.v1",
               "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "resolution_api": "isaacsim.storage.native.get_assets_root_path(skip_check=True) + path_join",
               "extension_enable_api": "omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate",
               "enabled_extension": "isaacsim.storage.native",
               "live_configured_root": root,
               "availability_asserted": False,
               "author_asset_release_or_hash_known": False,
               "assets": [{"name": name, "relative_path": relative,
                           "resolved_uri": path_join(root, relative), "primary_source": SOURCES[name],
                           "exact_author_bytes_claim": False} for name, relative in ASSETS.items()]}
    path = Path(output_path).resolve()
    repo = Path(__file__).resolve().parents[1]
    if not path.is_relative_to(repo):
        raise ValueError("Resolver only writes the current workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=2) + "\n")
    return {"request_path": str(path), "relative_paths": list(ASSETS.values()),
            "availability_asserted": False}
