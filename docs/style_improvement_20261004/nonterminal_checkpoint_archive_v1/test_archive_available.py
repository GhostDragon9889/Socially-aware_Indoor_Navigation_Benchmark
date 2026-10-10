"""Small synthetic ZIP/pickle cases, not checkpoint/model validation."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import pickletools
import tempfile
import unittest
from unittest import mock
import zipfile

import archive_available as saver


def never_deserialize():
    raise AssertionError("A GLOBAL/REDUCE callable must never execute")


class OpaqueTensor:
    def __reduce__(self):
        return never_deserialize, ()


def payload(step=15000):
    # Adam's nested step occurs first; pickle memoizes this key for the root.
    return {"optimizer": {"state": {0: {"step": OpaqueTensor()}}},
            "schema": saver.SCHEMA, "checkpoint_kind": saver.KIND,
            "step": step, "optimizer_updates": step, "microsteps": step * 2,
            "training_windows": step * 400, "microsteps_in_current_update": 0}


def make_zip(path, value=None, *, data=None, tensor=b"synthetic opaque storage"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", data if data is not None else pickle.dumps(value, protocol=2))
        archive.writestr("archive/data/0", tensor)


def replace_source(path, step):
    temporary = path.with_suffix(".publisher.tmp")
    make_zip(temporary, payload(step))
    os.replace(temporary, path)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "formal" / "last.ckpt"
        self.archive = self.base / "independent_history"

    def test_valid_root_with_nested_adam_memo_and_opaque_reduce(self):
        serialized = pickle.dumps(payload(), protocol=2)
        names = [opcode.name for opcode, _, _ in pickletools.genops(serialized)]
        self.assertIn("BINGET", names)
        self.assertIn("GLOBAL", names)
        self.assertIn("REDUCE", names)
        make_zip(self.source, data=serialized)
        source_bytes = self.source.read_bytes()
        source_mode = self.source.stat().st_mode
        result = saver.archive_once(self.source, self.archive)
        target = self.archive / "observed_step_15000.ckpt"
        self.assertEqual(result["header"]["step"], 15000)
        self.assertEqual(target.read_bytes(), source_bytes)
        self.assertEqual(target.stat().st_nlink, 1)
        self.assertNotEqual(target.stat().st_ino, self.source.stat().st_ino)
        self.assertFalse(target.stat().st_mode & 0o222)
        self.assertEqual(self.source.stat().st_mode, source_mode)
        self.assertFalse(result["tensor_validation_performed"])
        self.assertFalse(result["formal_terminal"])
        self.assertEqual(saver.archive_once(self.source, self.archive)["status"], "already_archived_identical")

    def test_bad_root_metadata_and_zip_rejected_without_deserialization(self):
        for key, value in (("step", True), ("optimizer_updates", 15001), ("microsteps", 1),
                           ("training_windows", 1), ("microsteps_in_current_update", 1), ("schema", "wrong")):
            with self.subTest(key=key):
                invalid = payload()
                invalid[key] = value
                make_zip(self.source, invalid)
                with self.source.open("rb") as stream, self.assertRaises(saver.HeaderError):
                    saver.read_header(stream)
        duplicate = b"\x80\x02}(X\x04\x00\x00\x00stepM\x98:X\x04\x00\x00\x00stepM\x98:u."
        with self.assertRaises(saver.HeaderError):
            saver.pickle_header(duplicate)
        with self.assertRaises(saver.HeaderError):
            saver.pickle_header(pickle.dumps(payload(), protocol=2) + b"trailing")
        for invalid in (b"\x80\x02R.", b"\x80\x02}R.", b"\x80\x02(\x85.", b"\x80\x02a."):
            with self.subTest(invalid=invalid), self.assertRaises(saver.HeaderError):
                saver.pickle_header(invalid)
        make_zip(self.source, payload())
        with zipfile.ZipFile(self.source, "a") as archive:
            archive.writestr("another/data.pkl", pickle.dumps(payload(), protocol=2))
        with self.source.open("rb") as stream, self.assertRaises(saver.HeaderError):
            saver.read_header(stream)

    def test_atomic_source_switch_discards_temp_then_retry(self):
        make_zip(self.source, payload())
        original = saver.fingerprint
        count = 0

        def changing(stream, path):
            nonlocal count
            result = original(stream, path)
            if Path(path) == self.source:
                count += 1
                if count == 2:
                    replace_source(self.source, 16000)
            return result

        with mock.patch.object(saver, "fingerprint", side_effect=changing), self.assertRaises(saver.SourceChanged):
            saver.archive_once(self.source, self.archive)
        self.assertEqual(list(self.archive.iterdir()), [])
        self.assertEqual(saver.archive_once(self.source, self.archive)["header"]["step"], 16000)

    def test_same_step_different_bytes_fails_preserving_first_copy(self):
        make_zip(self.source, payload())
        saver.archive_once(self.source, self.archive)
        target = self.archive / "observed_step_15000.ckpt"
        original = target.read_bytes()
        make_zip(self.source, payload(), tensor=b"different synthetic opaque storage")
        with self.assertRaises(FileExistsError):
            saver.archive_once(self.source, self.archive)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         json.loads(target.with_suffix(".json").read_text())["checkpoint"]["sha256"])
        self.assertFalse(list(self.archive.glob(".copy-*")))

    def test_skip_14k_observe_40k_and_missing_pid_does_not_stop(self):
        make_zip(self.source, payload(14000))
        self.assertEqual(saver.archive_once(self.source, self.archive)["status"], "outside_registered_history_range")
        replace_source(self.source, 40000)
        self.assertEqual(saver.archive_once(self.source, self.archive)["status"], "observed_40000_not_archived")
        self.assertEqual(list(self.archive.iterdir()), [])
        with mock.patch.object(Path, "read_text", side_effect=FileNotFoundError):
            observation = saver.process_observation(123456789)
        self.assertFalse(observation["parent_process_identity_observed"])
        self.assertFalse(observation["parent_exit_observed"])
        replace_source(self.source, 14000)
        with mock.patch.object(saver, "process_observation", return_value=observation), \
                mock.patch.object(saver.time, "monotonic", side_effect=[0., 2.]), contextlib.redirect_stdout(io.StringIO()):
            result = saver.main(["--source", str(self.source), "--archive-dir", str(self.archive), "--deadline-seconds", "1"])
        self.assertEqual(result, 2)
        rows = [json.loads(line) for line in (self.archive / "observations.jsonl").read_text().splitlines()]
        self.assertEqual(rows[-1]["status"], "deadline_reached")
        self.assertEqual(len(rows[-1]["missing_steps"]), 25)
        self.assertTrue(rows[-1]["parent_terminal_not_observed"])
        with self.assertRaises(ValueError):
            saver.archive_once(self.source, self.source.parent / "bad_archive")
        with self.assertRaises(ValueError):
            saver.validate_locations(self.source, saver.SOURCE.parent / "bad_archive")
        alias = self.base / "formal_alias"
        alias.symlink_to(saver.SOURCE.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            saver.validate_locations(self.source, alias / "bad_archive")
        journal = self.archive / "observations.jsonl"
        journal.unlink()
        before = self.source.read_bytes()
        journal.symlink_to(self.source)
        with self.assertRaises(OSError):
            saver.append_observation(self.archive, {"status": "must_not_write"})
        self.assertEqual(self.source.read_bytes(), before)
        journal.unlink()
        os.link(self.source, journal)
        with self.assertRaises(ValueError):
            saver.append_observation(self.archive, {"status": "must_not_write"})
        self.assertEqual(self.source.read_bytes(), before)

    def test_watch_cache_skips_same_generation_but_inode_and_stat_changes_trigger(self):
        make_zip(self.source, payload())
        now, sleeps = 0., 0

        def sleep(_):
            nonlocal now, sleeps
            now += 1.
            sleeps += 1
            if sleeps == 2:
                replace_source(self.source, 16000)
            elif sleeps == 3:
                info = self.source.stat()
                os.utime(self.source, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
            elif sleeps == 4:
                replace_source(self.source, 40000)

        with mock.patch.object(saver, "archive_once", wraps=saver.archive_once) as copying, \
                mock.patch.object(saver.time, "sleep", side_effect=sleep), \
                mock.patch.object(saver.time, "monotonic", side_effect=lambda: now), \
                mock.patch.object(saver, "process_observation", return_value={"parent_process_identity_observed": False}), \
                contextlib.redirect_stdout(io.StringIO()):
            result = saver.main(["--source", str(self.source), "--archive-dir", str(self.archive), "--deadline-seconds", "20"])
        self.assertEqual(result, 0)
        self.assertEqual(copying.call_count, 4)  # 15k, new16k inode, same16k changedstat, observed40k.
        rows = [json.loads(line) for line in (self.archive / "observations.jsonl").read_text().splitlines()]
        self.assertIn("unchanged_generation_not_rehashed", [row["status"] for row in rows])
        self.assertEqual(rows[-1]["status"], "observed_40000_not_archived")
        self.assertEqual(len(rows[-1]["missing_steps"]), 23)
        self.assertFalse((self.archive / "observed_step_40000.ckpt").exists())


if __name__ == "__main__":
    unittest.main()
