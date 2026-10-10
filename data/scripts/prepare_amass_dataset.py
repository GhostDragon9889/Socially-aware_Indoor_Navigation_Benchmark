"""Prepare the repository's SMPL AMASS splits without changing source archives.

Run from any directory with the conversion dependencies installed. This uses the
repository converter and MotionLib, not a substitute kinematic implementation.
The generic splits are NOT the NavIsaacLab paper's unpublished 2,848-clip subset.
"""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data/scripts"))

import yaml

ARCHIVES = {
    "MPI_HDM05": "HDM05", "MPI_mosh": "MoSh", "MPI_Limits": "PosePrior",
    "BioMotionLab_NTroje": "BMLrub", "Eyes_Japan_Dataset": "EyesJapanDataset",
    "TCD_handMocap": "TCDHands", "DFaust_67": "DFaust",
    "SSM_synced": "SSM", "Transitions_mocap": "Transitions",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def extract_archive(args):
    root, dataset = args
    raw = root / "raw"
    archive = raw / (ARCHIVES.get(dataset, dataset) + ".tar.bz2")
    sha = digest(archive)
    stamp = root / "preparation" / (archive.stem + ".extracted.json")
    if stamp.exists():
        saved = json.loads(stamp.read_text())
        if saved["sha256"] == sha and all(
            (raw / f["path"]).is_file() and (raw / f["path"]).stat().st_size == f["bytes"]
            for f in saved["files"]
        ):
            print(f"Extraction already complete: {archive.name}", flush=True)
            return
    print(f"Extracting {archive.name}", flush=True)
    files = []
    with tarfile.open(archive, "r|bz2") as tf:
        for member in tf:
            p = PurePosixPath(member.name)
            # Preserve each archive's top-level license under its dataset.
            if len(p.parts) == 1 and p.name == "LICENSE.txt":
                p = PurePosixPath(dataset) / p
            if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != dataset:
                raise ValueError(f"Unexpected archive path: {member.name}")
            dest = raw.joinpath(*p.parts)
            if not dest.resolve().is_relative_to(raw.resolve()):
                raise ValueError(f"Archive path escapes raw directory: {member.name}")
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or dest.is_symlink():
                raise ValueError(f"Unsupported archive member: {member.name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            stream = tf.extractfile(member)
            if dest.exists():
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(block)
                if dest.stat().st_size != member.size or digest(dest) != h.hexdigest():
                    raise FileExistsError(f"Existing file differs: {dest}")
            else:
                tmp = dest.with_name(dest.name + ".amass-partial")
                with open(tmp, "wb") as out:
                    shutil.copyfileobj(stream, out, 8 * 1024 * 1024)
                if tmp.stat().st_size != member.size:
                    raise IOError(f"Incomplete extraction: {member.name}")
                tmp.replace(dest)
            files.append({"path": str(p), "bytes": member.size})
    json_write(stamp, {"archive": str(archive), "sha256": sha, "files": files})
    print(f"Extracted {archive.name}: {len(files)} files", flush=True)


def init_worker():
    global converter, torch, kinematics
    import torch
    import convert_amass_to_proto as converter
    torch.set_num_threads(1)
    kinematics = converter.extract_kinematic_info(
        str(REPO / "protomotions/data/assets/mjcf/smpl_humanoid.xml")
    )


def convert_one(job):
    import numpy as np
    source, output, entry, signature = job
    record_path = output.with_suffix(".json")
    source_sha = digest(source)
    if output.exists():
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if (record["signature"] == signature and record["source_sha256"] == source_sha
                    and record["output_sha256"] == digest(output)):
                return record
        raise FileExistsError(f"Unverified existing output: {output}")
    with np.load(source, allow_pickle=False) as data:
        poses = data["poses"]
        trans = data["trans"]
        fps_key = "mocap_framerate" if "mocap_framerate" in data else "mocap_frame_rate"
        source_fps = float(data[fps_key])
    fps = int(round(source_fps))
    if fps <= 0 or poses.ndim != 2 or poses.shape[1] < 66 or trans.shape != (len(poses), 3):
        raise ValueError(f"Invalid AMASS arrays: {source}")
    if not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError(f"Nonfinite source arrays: {source}")
    timing = entry["sub_motions"][0]["timings"]
    start = max(0, int(timing.get("start", 0) * fps))
    end = min(len(poses), int(timing["end"] * fps)) if timing.get("end") is not None else len(poses)
    if end - start < 2:
        raise ValueError(f"Empty/short clip: {source} [{start}:{end}]")
    motion, output_fps = converter.convert_amass_to_motion(
        poses[start:end], trans[start:end], fps, 30, "smpl",
        converter.SMPL_BONE_ORDER_NAMES, converter.SMPL_MUJOCO_NAMES,
        kinematics, torch.device("cpu"), torch.float32,
    )
    tensors = motion.to_dict()
    frames = len(motion.rigid_body_pos)
    if frames < 2 or motion.dof_pos.shape != (frames, 69) or motion.rigid_body_pos.shape != (frames, 24, 3):
        raise ValueError(f"Unexpected output dimensions: {source}")
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite {key}: {source}")
    quat_error = float((motion.rigid_body_rot.norm(dim=-1) - 1).abs().max())
    if quat_error > 1e-4:
        raise ValueError(f"Nonunit quaternion: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".motion.partial")
    torch.save(tensors, tmp)
    output_sha = digest(tmp)
    record = {
        "source": str(source), "source_sha256": source_sha,
        "output": str(output), "output_sha256": output_sha, "signature": signature,
        "source_frames": len(poses), "source_fps": source_fps,
        "yaml_fps": entry.get("fps"), "source_frame_slice": [start, end],
        "output_frames": frames, "output_fps": output_fps,
        "duration_seconds": float(motion.motion_length), "quat_max_norm_error": quat_error,
        "contact_labels_true": int(motion.rigid_body_contacts.sum()),
    }
    json_write(record_path, record)
    tmp.replace(output)
    return record


def load_config(split):
    path = REPO / f"data/yaml_files/amass_smpl_{split}.yaml"
    entries = yaml.safe_load(path.read_text())["motions"]
    if len({e["file"] for e in entries}) != len(entries):
        raise ValueError("Duplicate motions require explicit disambiguation")
    if any(len(e.get("sub_motions", [])) != 1 for e in entries):
        raise ValueError("Expected exactly one timing range per motion")
    return path, entries


def prepare(root, split, workers):
    config, entries = load_config(split)
    datasets = sorted({Path(e["file"]).parts[0] for e in entries})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(extract_archive, [(root, d) for d in datasets]))
    index = {}
    for dataset in datasets:
        for source in sorted((root / "raw" / dataset).rglob("*.npz")):
            name = source.stem
            for char in "- ()":
                name = name.replace(char, "_")
            relative = source.relative_to(root / "raw").with_name(name + ".motion").as_posix()
            if relative in index:
                raise ValueError(f"Normalized filename collision: {relative}")
            index[relative] = source
    missing = [e["file"] for e in entries if e["file"] not in index]
    json_write(root / "preparation" / f"{split}_coverage.json", {
        "expected": len(entries), "found": len(entries) - len(missing), "missing": missing,
    })
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} motions; see {split}_coverage.json")
    sources = [config, Path(__file__), REPO / "data/scripts/convert_amass_to_proto.py",
               REPO / "data/scripts/contact_detection.py", REPO / "protomotions/components/pose_lib.py",
               REPO / "protomotions/data/assets/mjcf/smpl_humanoid.xml"]
    signature = {str(p.relative_to(REPO)): digest(p) for p in sources}
    jobs = [(index[e["file"]], root / "processed/smpl" / e["file"], e, signature) for e in entries]
    records = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker,
    ) as pool:
        for i, record in enumerate(pool.map(convert_one, jobs, chunksize=1), 1):
            records.append(record)
            if i % 100 == 0 or i == len(jobs):
                print(f"Converted {split}: {i}/{len(jobs)}", flush=True)
    json_write(root / "preparation" / f"{split}_conversion.json", records)
    converted_config = {"motions": [
        {"file": r["output"], "fps": r["output_fps"], "idx": i, "weight": e.get("weight", 1.0)}
        for i, (e, r) in enumerate(zip(entries, records))
    ]}
    (root / "preparation" / f"amass_smpl_{split}.yaml").write_text(yaml.safe_dump(converted_config))
    for stage in ("pack", "verify"):
        with open(root / "preparation" / f"{split}_{stage}.log", "w") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), str(root),
                            "--splits", split, "--stage", stage], check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f"{stage.capitalize()} complete: {split}", flush=True)


def pack_or_verify(root, split, stage):
    import torch
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig
    torch.set_num_threads(2)
    output = root / f"amass_smpl_{split}.pt"
    config = root / "preparation" / f"amass_smpl_{split}.yaml"
    if stage == "pack":
        if output.exists():
            raise FileExistsError(f"Will not overwrite existing package: {output}")
        lib = MotionLib(MotionLibConfig(motion_file=str(config)), device="cpu")
        tmp = output.with_name(output.stem + ".partial.pt")
        lib.save_to_file(tmp)
        tmp.replace(output)
        return
    lib = MotionLib(MotionLibConfig(motion_file=str(output)), device="cpu")
    _, entries = load_config(split)
    assert lib.num_motions() == len(entries)
    frames = int(lib.motion_num_frames.sum())
    assert tuple(lib.gts.shape) == (frames, 24, 3)
    assert tuple(lib.dps.shape) == (frames, 69)
    assert lib.contacts is not None and tuple(lib.contacts.shape) == (frames, 24)
    assert (lib.motion_num_frames >= 2).all() and (lib.motion_dt > 0).all()
    expected_starts = lib.motion_num_frames.cumsum(0) - lib.motion_num_frames
    assert torch.equal(expected_starts, lib.length_starts)
    assert torch.allclose(lib.motion_lengths, (lib.motion_num_frames - 1) * lib.motion_dt)
    for field in lib._fields:
        data = getattr(lib, field)
        if isinstance(data, torch.Tensor) and data.is_floating_point():
            for chunk in data.split(100000):
                assert torch.isfinite(chunk).all(), field
    quat_error = max(float((chunk.norm(dim=-1) - 1).abs().max()) for chunk in lib.grs.split(100000))
    assert quat_error < 1e-4
    ids = torch.arange(lib.num_motions())
    for fraction in (0, 0.371, 1):
        state = lib.get_motion_state(ids, lib.motion_lengths * fraction)
        assert torch.isfinite(state.dof_pos).all() and torch.isfinite(state.rigid_body_pos).all()
    from collections import Counter
    records = json.loads((root / "preparation" / f"{split}_conversion.json").read_text())
    summary = {
        "status": "verified", "split": split, "file": str(output),
        "bytes": output.stat().st_size, "sha256": digest(output),
        "motions": lib.num_motions(), "frames": frames,
        "duration_seconds": float(lib.motion_lengths.double().sum()),
        "fps_distribution": dict(Counter(str(r["output_fps"]) for r in records)),
        "dofs": 69, "bodies": 24, "quaternion_order": "xyzw",
        "max_quaternion_norm_error": quat_error, "contacts_true": int(lib.contacts.sum()),
        "clips_without_contacts": sum(r["contact_labels_true"] == 0 for r in records),
        "finite_all_float_fields": True, "interpolation_checked_every_clip": [0, 0.371, 1],
        "python": sys.version, "torch": torch.__version__,
        "scope": "Repository generic AMASS split; not the paper's 2848-clip locomotion selection",
    }
    json_write(root / "preparation" / f"{split}_verification.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_root", type=Path)
    parser.add_argument("--splits", nargs="+", choices=["validation", "test", "train"], default=["validation", "test", "train"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage", choices=["prepare", "pack", "verify"], default="prepare")
    args = parser.parse_args()
    root = args.motion_root.resolve()
    os.chdir(REPO)
    started = time.time()
    for split in args.splits:
        if args.stage == "prepare":
            prepare(root, split, args.workers)
        else:
            pack_or_verify(root, split, args.stage)
    print(f"Finished in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
