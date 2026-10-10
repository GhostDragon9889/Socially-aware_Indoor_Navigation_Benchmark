"""Refresh small LFS pointer/API identities, without retrieving model objects."""
import concurrent.futures
import json
from pathlib import Path

from fetch_metadata import fetch

REQUESTS = {
    "first_author_tree.json": "https://api.github.com/repos/leele-bing/ProtoMotions/git/trees/main?recursive=1",
    "first_author_gitattributes.txt": "https://raw.githubusercontent.com/leele-bing/ProtoMotions/main/.gitattributes",
    "first_author_mm_pointer.json": "https://api.github.com/repos/leele-bing/ProtoMotions/git/blobs/1683ad8929b7dbbb61e864fc0d57048c84895f99",
    "first_author_tracker_pointer.json": "https://api.github.com/repos/leele-bing/ProtoMotions/git/blobs/66220062ddd289611d54b12f9d33465936e0c4da",
    "author_generic_amp_main.json": "https://api.github.com/repos/BRoln7/IsaacLab_AMP_rl-games/branches/main",
    "author_generic_amp_releases.json": "https://api.github.com/repos/BRoln7/IsaacLab_AMP_rl-games/releases?per_page=100",
    "author_socnav_main.json": "https://api.github.com/repos/BRoln7/IsaacLab-socnav/branches/main",
    "author_socnav_releases.json": "https://api.github.com/repos/BRoln7/IsaacLab-socnav/releases?per_page=100",
}

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(fetch, REQUESTS.items()))
    (Path(__file__).resolve().parent / "lfs_fetch_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"requests": len(records), "success": sum(row.get("status") == 200 for row in records), "failures": [row for row in records if row.get("status") != 200]}, indent=2))
