#!/usr/bin/env python3
"""Download five public PACER references without modifying the runtime environment.

Requires existing gdown/requests tools. Uses no account, cookie file or netrc.
All IDs come from the fixed official download_data.sh; checkpoints are stored
apart from current ProtoMotions policies and are never deserialized here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import re
import shutil
import signal
import tempfile
import time
import zipfile
from pathlib import Path

COMMIT = "baeb6241f03fc9d78370a64e3c2dcef9d830d97f"
SOURCE = f"https://github.com/nv-tlabs/pacer/blob/{COMMIT}/download_data.sh"
ARTIFACTS = {
    "no_shape": ("1srmXSH8c-4Recy1g4jgWCNzZhDQ7mWfH", "output/release/pacer_no_shape/Humanoid.pth"),
    "standing_neutral": ("1rmxR4I2_bHazkaTujV48R1xZSVaK5tjb", None),
    "main": ("1CF8mA9L3cAAOdDk6qtwDnAOTrQI8mien", "output/release/pacer/Humanoid.pth"),
    "filtered_shapes": ("12xDUruRQHMMeuYhFh9ON3AVzXPyMOShT", None),
    "amass_occlusion": ("1rUSjVUYE8bIPZOyhbdQrKZ9-otE3fHpr", None),
}


class DownloadDeadline(BaseException):
    """Bypass network libraries' ordinary timeout wrapping at the batch limit."""


def evidence(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        head = stream.read(16)
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    is_zip = zipfile.is_zipfile(path)
    if is_zip:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Reference archive CRC failed")
    if head.lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise ValueError("Download is an HTML page")
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(), "head_hex": head.hex(),
            "zip_crc_valid": True if is_zip else None,
            "deserialized": False}


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--proxy", default=None, help="Explicit optional HTTP/SOCKS proxy URL")
    parser.add_argument("--roles", nargs="+", choices=list(ARTIFACTS), default=list(ARTIFACTS))
    parser.add_argument("--resume-output", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("--deadline-seconds", type=int, default=600)
    parser.add_argument("--max-file-bytes", type=int, default=150 * 1024**2)
    parser.add_argument("--max-total-bytes", type=int, default=750 * 1024**2)
    args = parser.parse_args()
    if args.deadline_seconds < 1:
        parser.error("deadline-seconds must be positive")
    if not 0 < args.max_file_bytes <= 150 * 1024**2:
        parser.error("max-file-bytes must be positive and at most 150 MiB")
    if not 0 < args.max_total_bytes <= 750 * 1024**2:
        parser.error("max-total-bytes must be positive and at most 750 MiB")
    overrides = dict(value.split("=", 1) for value in args.resume_output)
    if any(role not in ARTIFACTS for role in overrides):
        parser.error("Unknown resume-output role")

    import gdown
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    repo = args.repo_root.absolute()
    root = repo / "data/pretrained_models/paper_references/pacer"
    report_path = repo / "docs/resource_completion_20261001/pacer_download_manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 2 * args.max_total_bytes + 256 * 1024**2:
        raise ValueError("Insufficient free space for bounded staging and published copies")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({"source": SOURCE, "commit": COMMIT, "date": "2026-10-01",
                   "license": "CC-BY-NC-SA-4.0 (official repository); original data/model rights also apply",
                   "license_url": f"https://github.com/nv-tlabs/pacer/blob/{COMMIT}/LICENSE",
                   "authentication": "anonymous; use_cookies=False; Session.trust_env=False",
                   "gdown": gdown.__version__, "upstream_checksums_published": False,
                   "max_file_bytes": args.max_file_bytes,
                   "max_total_bytes": args.max_total_bytes,
                   "requested_roles": args.roles,
                   "free_bytes_preflight": free_bytes,
                   "runtime_integration": False})
    records = report.setdefault("artifacts", {})
    module = importlib.import_module("gdown.download")
    original_get_session = module._get_session
    current = {}

    def get_session(**kwargs):
        session, cookies_file = original_get_session(**kwargs)
        session.trust_env = False
        retry = Retry(total=3, connect=3, read=3, status=3, backoff_factor=0.5,
                      status_forcelist=(429, 502, 503, 504), allowed_methods=("GET",))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        original_get = session.get

        def bounded_get(url, **options):
            options.setdefault("timeout", (30, 60))
            response = original_get(url, **options)
            response.raise_for_status()
            start = re.fullmatch(r"bytes=(\d+)-", options.get("headers", {}).get("Range", ""))
            if start:
                content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if response.status_code != 206 or not content_range or int(content_range[1]) != int(start[1]):
                    response.close()
                    raise ValueError("Server ignored resume Range; refusing to append a full file")
                current["server_expected_bytes"] = int(content_range[3])
            elif "Content-Disposition" in response.headers:
                size = response.headers.get("Content-Length")
                if size is not None:
                    current["server_expected_bytes"] = int(size)
            if "Content-Disposition" in response.headers:
                expected = current.get("server_expected_bytes")
                if expected is None or not 0 < expected <= args.max_file_bytes:
                    response.close()
                    raise ValueError("Public file size missing or above the 150 MiB artifact limit")
                planned = sum(records.get(role, {}).get("server_expected_bytes", 0) for role in args.roles)
                if planned > args.max_total_bytes:
                    response.close()
                    raise ValueError("Requested artifacts exceed the bounded batch size")
                current["original_filename"] = module._get_filename_from_response(response=response)
                save_report(report_path, report)
            return response

        session.get = bounded_get
        return session, cookies_file

    module._get_session = get_session
    started = time.monotonic()
    previous_elapsed = report.get("cumulative_download_seconds", report.get("elapsed_seconds", 0))

    def expire(*_):
        raise DownloadDeadline("PACER batch reached its wall-clock deadline")

    previous_handler = signal.signal(signal.SIGALRM, expire)
    signal.alarm(args.deadline_seconds)
    log_path = report_path.with_suffix(".log")
    try:
        with log_path.open("a") as log:
            for role in args.roles:
                public_id, relative = ARTIFACTS[role]
                current = records.setdefault(role, {})
                current.update({"id": public_id, "source_url": f"https://drive.google.com/uc?id={public_id}"})
                if current.get("status") in ("downloaded", "verified_existing"):
                    old = Path(current["file"]["path"])
                    actual = evidence(old)
                    if actual["sha256"] != current["file"]["sha256"] or actual["bytes"] != current["file"]["bytes"]:
                        raise ValueError(f"Existing verified {role} changed; refusing to replace it")
                    current["status"] = "verified_existing"
                    current["partials"] = []
                    save_report(report_path, report)
                    print(f"{role}: verified existing {actual['bytes']} bytes", flush=True)
                    continue
                stage = Path(overrides.get(role, str(root / ".downloads" / f"{role}.download")))
                stage.parent.mkdir(parents=True, exist_ok=True)
                current["staging_output"] = str(stage)
                for attempt in range(1, 4):
                    current["attempt"] = attempt
                    current["status"] = "downloading"
                    save_report(report_path, report)
                    try:
                        with contextlib.redirect_stderr(log):
                            gdown.download(id=public_id, output=str(stage), proxy=args.proxy,
                                           use_cookies=False, resume=True, quiet=True)
                        actual = evidence(stage)
                        expected = current.get("server_expected_bytes")
                        if expected is None or actual["bytes"] != expected:
                            raise ValueError("Completed file differs from server Content-Length/Range total")
                        original = current.get("original_filename", "")
                        if relative is None:
                            if not original or Path(original).name != original:
                                raise ValueError("Cannot determine a safe official sample filename")
                            destination = root / "sample_data" / original
                        else:
                            destination = root / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            existing = evidence(destination)
                            if existing["sha256"] != actual["sha256"] or existing["bytes"] != actual["bytes"]:
                                raise ValueError("Existing destination differs; refusing to overwrite")
                        else:
                            # Validate a same-directory copy, then publish atomically without
                            # replacement. An interruption cannot create a partial final file.
                            temporary = None
                            try:
                                with tempfile.NamedTemporaryFile(
                                    dir=destination.parent, prefix=destination.name + ".publish_",
                                    suffix=".tmp", delete=False,
                                ) as output, stage.open("rb") as source:
                                    temporary = Path(output.name)
                                    shutil.copyfileobj(source, output, 1024 * 1024)
                                    output.flush()
                                    os.fsync(output.fileno())
                                copied = evidence(temporary)
                                if copied["sha256"] != actual["sha256"] or copied["bytes"] != actual["bytes"]:
                                    raise ValueError("Temporary publication copy differs from staging")
                                try:
                                    os.link(temporary, destination)
                                except FileExistsError:
                                    existing = evidence(destination)
                                    if existing["sha256"] != actual["sha256"] or existing["bytes"] != actual["bytes"]:
                                        raise ValueError("Concurrent destination differs; refusing to overwrite")
                            finally:
                                if temporary is not None:
                                    temporary.unlink(missing_ok=True)
                        published = evidence(destination)
                        if published["sha256"] != actual["sha256"]:
                            raise ValueError("Published copy differs from staging")
                        current.update({"status": "downloaded", "file": published})
                        current["partials"] = []
                        current.pop("error", None)
                        save_report(report_path, report)
                        print(f"{role}: downloaded {published['bytes']} bytes -> {destination}", flush=True)
                        break
                    except DownloadDeadline:
                        raise
                    except Exception as error:
                        current.update({"status": "partial_or_failed", "error": f"{type(error).__name__}: {str(error)[:240]}"})
                        current["partials"] = [{"path": str(p), "bytes": p.stat().st_size}
                                               for p in stage.parent.glob(stage.name + "*.part")]
                        save_report(report_path, report)
                        print(f"{role}: attempt {attempt} failed ({type(error).__name__}); partial kept", flush=True)
    except DownloadDeadline as error:
        current.update({"status": "deadline_exceeded", "error": str(error)})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        module._get_session = original_get_session
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["cumulative_download_seconds"] = round(previous_elapsed + report["elapsed_seconds"], 3)
        report["status"] = "complete" if all(records.get(role, {}).get("status") in ("downloaded", "verified_existing") for role in args.roles) else "partial"
        save_report(report_path, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
