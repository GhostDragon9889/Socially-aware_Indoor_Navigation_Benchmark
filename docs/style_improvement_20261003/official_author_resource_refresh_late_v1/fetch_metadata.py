"""Anonymous, bounded metadata-only refresh; never follows model download links."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
REQUESTS = {
    "official_main.json": "https://api.github.com/repos/yh83305/NavIsaaclab2.0/branches/main",
    "official_tree.json": "https://api.github.com/repos/yh83305/NavIsaaclab2.0/git/trees/main?recursive=1",
    "official_releases.json": "https://api.github.com/repos/yh83305/NavIsaaclab2.0/releases?per_page=100",
    "official_tags.json": "https://api.github.com/repos/yh83305/NavIsaaclab2.0/tags?per_page=100",
    "author_fork_main.json": "https://api.github.com/repos/BRoln7/NavIsaaclab2.0/branches/main",
    "first_author_main.json": "https://api.github.com/repos/leele-bing/ProtoMotions/branches/main",
    "first_author_releases.json": "https://api.github.com/repos/leele-bing/ProtoMotions/releases?per_page=100",
    "author_site_master.json": "https://api.github.com/repos/BRoln7/NavIsaacLab-web/branches/master",
    "author_site_tree.json": "https://api.github.com/repos/BRoln7/NavIsaacLab-web/git/trees/master?recursive=1",
    "author_project_live.html": "https://broln7.github.io/NavIsaacLab-web/",
    "official_full_pipeline.md": "https://raw.githubusercontent.com/yh83305/NavIsaaclab2.0/main/FULL_PIPELINE.md",
    "official_crowdsim_readme.md": "https://raw.githubusercontent.com/yh83305/NavIsaaclab2.0/main/CrowdSim/README.md",
    "official_humanoid_readme.md": "https://raw.githubusercontent.com/yh83305/NavIsaaclab2.0/main/CrowdSim/humanoid/README.md",
    "official_gitattributes.txt": "https://raw.githubusercontent.com/yh83305/NavIsaaclab2.0/main/.gitattributes",
    "official_license.txt": "https://raw.githubusercontent.com/yh83305/NavIsaaclab2.0/main/LICENSE",
}


def fetch(item):
    name, url = item
    record = {"file": name, "url": url, "requested_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    request = urllib.request.Request(url, headers={"User-Agent": "NavIsaacLab-primary-resource-audit", "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = response.read(6_000_001)
            if len(payload) > 6_000_000:
                raise ValueError("metadata size exceeded 6 MB; response not saved")
            record.update(status=response.status, final_url=response.url, http_date=response.headers.get("Date"), etag=response.headers.get("ETag"), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            (HERE / name).write_bytes(payload)
    except (urllib.error.URLError, ValueError) as error:
        record.update(status=getattr(error, "code", None), error=str(error))
    return record


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(fetch, REQUESTS.items()))
    (HERE / "fetch_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"requests": len(records), "success": sum(row.get("status") == 200 for row in records), "failures": [row for row in records if row.get("status") != 200]}, indent=2))
