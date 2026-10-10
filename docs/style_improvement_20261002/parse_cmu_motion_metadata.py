"""Parse a saved official CMU HTML metadata page; never fetch motion files."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class MotionPage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.row = None
        self.cell = None
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th") and self.row is not None:
            self.cell = {"text": "", "links": []}
        if tag == "a" and self.cell is not None:
            self.cell["links"].append(dict(attrs).get("href", ""))

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.cell["text"] = " ".join(self.cell["text"].split())
            self.row.append(self.cell)
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
            self.cell = None

    def handle_data(self, text):
        if self.cell is not None:
            self.cell["text"] += text


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_metadata(html):
    parser = MotionPage()
    parser.feed(html)
    trials = {}
    subjects = {}
    current_subject = None
    for row in parser.rows:
        for cell in row:
            match = re.search(r"Subject #(\d+) \((.*)\) file index", cell["text"])
            if match:
                current_subject = int(match[1])
                subjects[current_subject] = match[2]
        links = [url for cell in row for url in cell["links"]]
        feedback = [
            match for url in links
            if (match := re.search(r"badtrial.php\?subjectnumber=(\d+)&motion=(\d+)", url))
        ]
        if not feedback:
            continue
        if len(feedback) != 1 or len(row) != 10 or not row[1]["text"].isdigit():
            raise ValueError("Unexpected official motion-row layout")
        subject, take = map(int, feedback[0].groups())
        if subject != current_subject or take != int(row[1]["text"]):
            raise ValueError("Subject header, trial cell and feedback IDs disagree")
        amc_links = [url for url in links if url.endswith(".amc")]
        if len(amc_links) != 1:
            raise ValueError("Expected one AMC metadata link per trial")
        match = re.fullmatch(r"/subjects/(\d+)/(\d+)_(\d+)\.amc", amc_links[0])
        if not match or list(map(int, match.groups())) != [subject, subject, take]:
            raise ValueError("Official AMC filename does not match row IDs")
        key = f"{subject}_{take}"
        if key in trials:
            raise ValueError("Duplicate numeric trial in official page")
        description = row[2]["text"]
        trials[key] = {
            "subject": subject,
            "take": take,
            "description": description or None,
            "description_status": "documented" if description else "official_description_blank",
            "subject_session_description": subjects[subject],
            "source_url": f"https://mocap.cs.cmu.edu/search.php?subjectnumber={subject}&trinum={take}",
            "source_snapshot_url": "https://mocap.cs.cmu.edu/search.php",
            "amc_metadata_link": f"https://mocap.cs.cmu.edu{amc_links[0]}",
            "original_capture_framerate_text": row[-2]["text"],
            "normalized_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        }
    return trials, subjects


def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument("--html", type=Path, required=True)
    args_parser.add_argument("--local-ids", type=Path, required=True)
    args_parser.add_argument("--output", type=Path, required=True)
    args_parser.add_argument("--join-output", type=Path, required=True)
    args = args_parser.parse_args()
    if args.output.exists() or args.join_output.exists():
        raise FileExistsError("Metadata outputs already exist; refuse overwrite")
    trials, subjects = parse_metadata(args.html.read_text())
    local = json.loads(args.local_ids.read_text())
    records = []
    join = {}
    for item in local:
        match = re.fullmatch(r"(\d+)_(\d+)_poses\.motion", Path(item["source_path"]).name)
        if not match or list(map(int, match.groups())) != [item["subject"], item["take"]]:
            raise ValueError("Actual basename differs from exported numeric IDs")
        key = f"{item['subject']}_{item['take']}"
        metadata = trials.get(key, {
            "subject": item["subject"],
            "take": item["take"],
            "description": None,
            "description_status": "not_present_in_current_full_motions_page",
            "source_url": "https://mocap.cs.cmu.edu/search.php",
        })
        records.append({**item, **metadata})
        if metadata["description"]:
            if item["source_path"] in join:
                raise ValueError("Actual source path duplicated")
            join[item["source_path"]] = {
                **metadata,
                "official_page_sha256": sha256(args.html),
                "official_page_snapshot": str(args.html.resolve()),
            }
    status_counts = Counter(record["description_status"] for record in records)
    report = {
        "schema": "navisaaclab.cmu_motion_labels.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "canonical_url": "https://mocap.cs.cmu.edu/search.php",
            "retrieval_url": "http://mocap.cs.cmu.edu/search.php",
            "transport": "HTTP; HTTPS direct/proxy failed with TLS EOF; cached official search metadata spot-checked",
            "snapshot": str(args.html.resolve()),
            "bytes": args.html.stat().st_size,
            "sha256": sha256(args.html),
            "second_independent_http_route_identical_sha256": sha256(args.html),
            "retrieval_date": "2026-10-02",
            "scope": "One complete motions-description HTML page; no AMC/C3D/ASF/movie data downloaded; no recursive crawl",
        },
        "local_inventory": {
            "path": str(args.local_ids.resolve()),
            "sha256": sha256(args.local_ids),
            "rows": len(local),
            "numeric_trials": len({(row["subject"], row["take"]) for row in local}),
            "numeric_subjects": len({row["subject"] for row in local}),
            "split": "Existing AMASS SMPL train bank only",
        },
        "coverage": {
            "official_trials_parsed": len(trials),
            "official_subjects_parsed": len(subjects),
            "local_description_status_counts": dict(status_counts),
            "not_present_by_subject": dict(Counter(
                str(row["subject"]) for row in records
                if row["description_status"] == "not_present_in_current_full_motions_page"
            )),
        },
        "join_contract": {
            "identity": "Integer subject/take from actual basename, never directory name",
            "records": "One row per installed motion_id/source_path; directory aliases remain separate",
            "join_file": str(args.join_output.resolve()),
            "join_format": "actual source_path -> metadata with nonempty official description + source_url",
            "unknown": "Blank/missing trial descriptions omitted from join; never replaced with session description",
            "classification": "No navigation/style label inferred here; downstream explicit selection policy must reject mixed actions independently",
        },
        "limitations": [
            "Official full-page response currently lists 2435 trials; homepage cached total2605 is not asserted as current full metadata coverage.",
            "Numeric subject IDs are recording sessions, not verified unique identities; CMU states one person may have multiple subject numbers.",
            "Labels describe complete trials; mixed walk+throw/jump/boxing/object interaction is not a pure locomotion clip.",
            "Stylized walking descriptions are retained literally, not asserted ordinary pedestrian gait.",
            "Official toe/hand mocap can be noisy; labels do not certify tracking quality or foot-contact ground truth.",
            "This metadata join does not recover the author's unpublished 2848-clip selection or prove paper-equivalent resources.",
        ],
        "site_policy_sources": ["https://mocap.cs.cmu.edu/", "https://mocap.cs.cmu.edu/faqs.php"],
        "records": records,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.join_output.write_text(json.dumps(join, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"coverage": report["coverage"], "join_records": len(join)}))


if __name__ == "__main__":
    main()
