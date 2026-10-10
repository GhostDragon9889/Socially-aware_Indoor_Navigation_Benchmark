"""Preserve stable published intermediate bytes; never deserialize a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickletools
import stat
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/last.ckpt"
ARCHIVE = ROOT / "output/diffusion_amp/official_ORCA_all_temporal_available_history_v1"
SCHEMA = "crowdsim.paper_five_state_diffusion.v1"
KIND = "local_five_state.full_raw_EMA_accumulation.v2"
KEYS = {"schema", "checkpoint_kind", "step", "optimizer_updates", "microsteps",
        "training_windows", "microsteps_in_current_update"}
MARK, OPAQUE = object(), object()


class HeaderError(ValueError):
    pass


class SourceChanged(RuntimeError):
    pass


class Fields:
    def __init__(self):
        self.values = {}

    def put(self, pairs):
        if len(pairs) % 2:
            raise HeaderError("Odd dictionary item stack")
        for key, value in zip(pairs[::2], pairs[1::2]):
            if type(key) is str and key in KEYS:
                if key in self.values:
                    raise HeaderError("Duplicate counted metadata key")
                self.values[key] = value


def pickle_header(data):
    """Passive opcode stack: GLOBAL/REDUCE/persistent IDs stay opaque."""
    stack, memo = [], {}

    def consume(count):
        if len(stack) < count or any(value is MARK for value in stack[-count:]):
            raise HeaderError("Missing passive opcode operands")
        values = stack[-count:]
        del stack[-count:]
        return values

    def target():
        if not stack or stack[-1] is MARK:
            raise HeaderError("Missing passive mutation target")
        return stack[-1]

    def marked():
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is MARK:
                values = stack[index + 1:]
                del stack[index:]
                return values
        raise HeaderError("Missing pickle MARK")

    try:
        for number, (opcode, arg, position) in enumerate(pickletools.genops(data)):
            if number > 1000000 or len(stack) > 1000000 or len(memo) > 1000000:
                raise HeaderError("Oversized passive pickle stack")
            name = opcode.name
            if name == "PROTO":
                if arg not in (2, 3, 4, 5):
                    raise HeaderError("Unsupported Torch pickle protocol")
            elif name == "FRAME":
                pass
            elif name == "MARK":
                stack.append(MARK)
            elif name == "EMPTY_DICT":
                stack.append(Fields())
            elif name == "DICT":
                value = Fields()
                value.put(marked())
                stack.append(value)
            elif name in ("SETITEM", "SETITEMS"):
                pairs = consume(2) if name == "SETITEM" else marked()
                if isinstance(target(), Fields):
                    target().put(pairs)
                elif target() is not OPAQUE:
                    raise HeaderError("Invalid passive dictionary target")
            elif name in ("INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4",
                          "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "STRING"):
                stack.append(arg)
            elif name in ("NEWTRUE", "NEWFALSE", "NONE"):
                stack.append({"NEWTRUE": True, "NEWFALSE": False, "NONE": None}[name])
            elif name in ("PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"):
                key = len(memo) if name == "MEMOIZE" else int(arg)
                memo[key] = target()
            elif name in ("GET", "BINGET", "LONG_BINGET"):
                stack.append(memo[int(arg)])
            elif name in ("GLOBAL", "EXT1", "EXT2", "EXT4", "PERSID", "FLOAT", "BINFLOAT",
                          "BINSTRING", "SHORT_BINSTRING", "BINBYTES", "SHORT_BINBYTES", "BINBYTES8",
                          "BYTEARRAY8", "EMPTY_LIST", "EMPTY_TUPLE", "EMPTY_SET"):
                stack.append(OPAQUE)
            elif name in ("STACK_GLOBAL", "REDUCE", "NEWOBJ", "NEWOBJ_EX"):
                consume(3 if name == "NEWOBJ_EX" else 2)
                stack.append(OPAQUE)
            elif name in ("BINPERSID", "TUPLE1", "TUPLE2", "TUPLE3"):
                consume({"TUPLE2": 2, "TUPLE3": 3}.get(name, 1))
                stack.append(OPAQUE)
            elif name in ("LIST", "TUPLE", "FROZENSET"):
                marked()
                stack.append(OPAQUE)
            elif name in ("APPENDS", "ADDITEMS"):
                marked()
                target()
            elif name == "POP_MARK":
                marked()
            elif name in ("APPEND", "BUILD"):
                consume(1)
                target()
            elif name == "POP":
                consume(1)
            elif name == "DUP":
                stack.append(target())
            elif name == "STOP":
                if position != len(data) - 1 or len(stack) != 1 or not isinstance(stack[0], Fields):
                    raise HeaderError("Expected one complete root dictionary, no trailing bytes")
                return stack[0].values
            else:
                raise HeaderError(f"Unsupported passive opcode: {name}")
    except (ValueError, IndexError, KeyError, OverflowError) as error:
        raise HeaderError(f"Invalid passive pickle header: {error}") from error
    raise HeaderError("No complete pickle STOP")


def read_header(stream):
    with zipfile.ZipFile(stream) as archive:
        rows = [row for row in archive.infolist()
                if row.filename == "data.pkl" or row.filename.endswith("/data.pkl")]
        if len(rows) != 1 or rows[0].flag_bits & 1 or not 0 < rows[0].file_size <= 32 << 20:
            raise HeaderError("Expected one unencrypted bounded ZIP data.pkl")
        value = pickle_header(archive.read(rows[0]))
    if (value.get("schema") != SCHEMA or value.get("checkpoint_kind") != KIND
            or any(type(value.get(key)) is not int for key in
                   ("step", "optimizer_updates", "microsteps", "training_windows", "microsteps_in_current_update"))
            or value["step"] < 1 or value["step"] != value["optimizer_updates"]
            or value["microsteps"] != value["step"] * 2
            or value["training_windows"] != value["step"] * 400
            or value["microsteps_in_current_update"] != 0):
        raise HeaderError("Not the registered complete optimizer-update header")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(stream, path):
    stream.seek(0)
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Checkpoint must be a regular file")
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return {"path": str(path), "bytes": info.st_size, "sha256": digest.hexdigest(),
            "device": info.st_dev, "inode": info.st_ino, "nlink": info.st_nlink}


def process_observation(pid):
    """Optional observation; an invisible/exited PID never ends the watcher."""
    try:
        base = Path("/proc") / str(pid)
        raw = (base / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        command = (base / "cmdline").read_bytes().split(b"\0")
        return {"pid": pid, "parent_process_identity_observed": True,
                "ppid": int(fields[1]), "starttime_ticks": int(fields[19]),
                "state": fields[0], "cmdline": [arg.decode(errors="replace") for arg in command if arg],
                "parent_exit_observed": False, "formal_terminal": False}
    except (OSError, ValueError, IndexError) as error:
        return {"pid": pid, "parent_process_identity_observed": False,
                "reason": "sandbox_visibility_or_unavailable_proc", "error_type": type(error).__name__,
                "parent_exit_observed": False, "formal_terminal": False}


def generation_identity(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)


def write_json_exclusive(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def append_observation(destination, value):
    path = destination / "observations.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "a") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Observation journal must be an ordinary independent single-link file")
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_locations(source, destination):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    resolved_source, resolved_destination = source.resolve(), destination.resolve()
    formal_output = SOURCE.resolve().parent
    if (source.is_symlink() or destination.is_symlink() or resolved_source.parent == resolved_destination
            or resolved_source.parent in resolved_destination.parents or resolved_destination in resolved_source.parents
            or resolved_destination == formal_output or formal_output in resolved_destination.parents):
        raise ValueError("Archive must be independent of the formal source output")
    return source, destination


def archive_once(source, destination):
    """Open one generation; compare opened-fd bytes and the current path."""
    source, destination = validate_locations(source, destination)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with source.open("rb") as original:
            header = read_header(original)
            step = header["step"]
            if step == 40000:
                return {"status": "observed_40000_not_archived", "header": header, "formal_terminal": False}
            if not 15000 <= step <= 39000 or step % 1000:
                return {"status": "outside_registered_history_range", "header": header, "formal_terminal": False}
            before = fingerprint(original, source)
            fd, name = tempfile.mkstemp(prefix=".copy-", suffix=".tmp", dir=destination)
            temporary = Path(name)
            with os.fdopen(fd, "wb") as copied:
                original.seek(0)
                for chunk in iter(lambda: original.read(1 << 20), b""):
                    copied.write(chunk)
                copied.flush()
                os.fsync(copied.fileno())
            with temporary.open("rb") as copied:
                replica = fingerprint(copied, temporary)
            opened_after = fingerprint(original, source)
            with source.open("rb") as current:
                path_after = fingerprint(current, source)
            if (before != opened_after or before != path_after
                    or replica["bytes"] != before["bytes"] or replica["sha256"] != before["sha256"]
                    or replica["nlink"] != 1 or replica["inode"] == before["inode"]):
                raise SourceChanged("Source generation changed during independent copy; own temporary discarded")
            with temporary.open("rb") as copied:
                actual_header = read_header(copied)
            if actual_header != header:
                raise SourceChanged("Header generation changed before copy; own temporary discarded")
        target = destination / f"observed_step_{actual_header['step']}.ckpt"
        metadata = target.with_suffix(".json")
        if target.exists():
            if target.is_symlink():
                raise FileExistsError("Existing history checkpoint is a symlink")
            with target.open("rb") as stream:
                existing = fingerprint(stream, target)
            if (existing["bytes"] != replica["bytes"] or existing["sha256"] != replica["sha256"]
                    or existing["nlink"] != 1 or target.stat().st_mode & 0o222 or not metadata.is_file()
                    or metadata.is_symlink() or metadata.stat().st_mode & 0o222):
                raise FileExistsError("Existing step differs or has no archive record; never overwrite")
            saved = json.loads(metadata.read_text())
            if (saved.get("header") != actual_header or saved.get("checkpoint") != existing
                    or saved.get("tensor_validation_performed") is not False or saved.get("formal_terminal") is not False):
                raise FileExistsError("Existing archive record differs; never overwrite")
            return {"status": "already_archived_identical", "header": actual_header, "checkpoint": existing,
                    "formal_terminal": False}
        # Exclusive publication, never os.replace/hard-link a history checkpoint.
        with temporary.open("rb") as copied, target.open("xb") as final:
            for chunk in iter(lambda: copied.read(1 << 20), b""):
                final.write(chunk)
            final.flush()
            os.fsync(final.fileno())
        with target.open("rb") as final:
            checkpoint = fingerprint(final, target)
        if (checkpoint["sha256"] != replica["sha256"] or checkpoint["bytes"] != replica["bytes"]
                or checkpoint["nlink"] != 1 or checkpoint["inode"] == before["inode"]):
            raise ValueError("Published independent history bytes differ; retain negative evidence")
        target.chmod(0o444)
        result = {"schema": "official_orca.available_nonterminal_checkpoint.v1", "status": "archived_stable_bytes",
                  "observed_at_UTC": utc_now(), "header": actual_header, "source_before": before,
                  "source_opened_after": opened_after, "source_path_after": path_after,
                  "checkpoint": checkpoint, "tensor_validation_performed": False, "pickle_loads_used": False,
                  "formal_terminal": False, "history_only_not_checkpoint_selection": True}
        write_json_exclusive(metadata, result)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE)
    parser.add_argument("--parent-pid", type=int, default=873975)
    parser.add_argument("--poll-seconds", type=float, default=15.)
    parser.add_argument("--deadline-seconds", type=float, default=43200.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.parent_pid < 1 or args.poll_seconds <= 0 or args.deadline_seconds <= 0:
        parser.error("Positive PID, poll interval and deadline required")
    args.source, args.archive_dir = validate_locations(args.source, args.archive_dir)
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    deadline, last_observation = time.monotonic() + args.deadline_seconds, None
    handled_generation, handled_step = None, None
    while True:
        try:
            current_generation = generation_identity(args.source)
            if not args.once and current_generation == handled_generation:
                result = {"status": "unchanged_generation_not_rehashed", "step": handled_step, "formal_terminal": False}
            else:
                result = archive_once(args.source, args.archive_dir)
                if (result["status"] in ("archived_stable_bytes", "already_archived_identical",
                                         "outside_registered_history_range", "observed_40000_not_archived")
                        and generation_identity(args.source) == current_generation):
                    handled_generation, handled_step = current_generation, result["header"]["step"]
        except (SourceChanged, FileNotFoundError, zipfile.BadZipFile, EOFError) as error:
            result = {"status": "transient_retry", "error_type": type(error).__name__, "reason": str(error),
                      "formal_terminal": False}
        except (HeaderError, ValueError, OSError) as error:
            result = {"status": "failed_preserved", "error_type": type(error).__name__, "reason": str(error),
                      "formal_terminal": False}
        result["observed_at_UTC"] = utc_now()
        result["parent_terminal_not_observed"] = True
        if result["status"] == "observed_40000_not_archived":
            result["missing_steps"] = [step for step in range(15000, 40000, 1000)
                                       if not (args.archive_dir / f"observed_step_{step}.json").is_file()]
        result["parent_process_observation"] = process_observation(args.parent_pid)
        signature = (result["status"], result.get("header", {}).get("step"), result.get("reason"))
        if signature != last_observation:
            append_observation(args.archive_dir, result)
            print(json.dumps(result, allow_nan=False), flush=True)
            last_observation = signature
        if result["status"] == "failed_preserved":
            return 1
        if args.once or result["status"] == "observed_40000_not_archived":
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missed = [step for step in range(15000, 40000, 1000)
                      if not (args.archive_dir / f"observed_step_{step}.json").is_file()]
            append_observation(args.archive_dir, {"status": "deadline_reached", "missing_steps": missed,
                                                  "observed_at_UTC": utc_now(), "formal_terminal": False,
                                                  "parent_terminal_not_observed": True})
            return 2 if missed else 0
        time.sleep(min(args.poll_seconds, remaining))


if __name__ == "__main__":
    raise SystemExit(main())
