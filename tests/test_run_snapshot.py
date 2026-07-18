"""Tests for the run-snapshot helpers in _tag.py.

These exercise the compute-and-write logic without touching real tags.
We run importable pieces directly against a temporary git repo.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import _tag as tag  # noqa: E402


def _make_tmp_repo():
    """Build an isolated git repo on tmpfs-style path for unit tests."""
    tmp = Path(tempfile.mkdtemp(prefix="tag_run_test_"))
    subprocess.run(
        ["git", "-C", str(tmp), "init", "--initial-branch=main"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp), "config", "user.email", "test@test"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )
    (tmp / "README.md").write_text("hello\n")
    subprocess.run(
        ["git", "-C", str(tmp), "add", "README.md"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp), "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )
    return tmp


class NextExpNumberTests(unittest.TestCase):
    def test_no_existing_returns_one(self):
        tmp = _make_tmp_repo()
        try:
            n = tag.next_exp_number(tmp, "fresh_topic")
            self.assertEqual(n, 1)
        finally:
            subprocess.run(["git", "-C", str(tmp), "tag", "exp/fresh_topic/003"], check=True, capture_output=True)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_existing_advances(self):
        tmp = _make_tmp_repo()
        try:
            subprocess.run(
                ["git", "-C", str(tmp), "tag", "exp/foo/001"], check=True, capture_output=True
            )
            subprocess.run(
                ["git", "-C", str(tmp), "tag", "exp/foo/002"], check=True, capture_output=True
            )
            n = tag.next_exp_number(tmp, "foo")
            self.assertEqual(n, 3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ConfigWriterTests(unittest.TestCase):
    def setUp(self):
        # Patch ROOT so writes go to a temp dir.
        self._orig_root = tag.ROOT
        self.tmp = _make_tmp_repo()
        # _write_config uses ROOT/EXPERIMENTS_DIR
        new_root = self.tmp
        tag.ROOT = new_root
        tag.EXPERIMENTS_DIR = new_root / "experiments"

    def tearDown(self):
        tag.ROOT = self._orig_root
        tag.EXPERIMENTS_DIR = self._orig_root / "experiments"
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_topic_seq_note(self):
        cfg_path = tag._write_config(
            exp_id="exp1",
            topic="byte_state_byte",
            n=5,
            note="smoke",
            user_config_path=None,
        )
        cfg = json.loads(cfg_path.read_text())
        self.assertEqual(cfg["tag_topic"], "byte_state_byte")
        self.assertEqual(cfg["tag_seq"], 5)
        self.assertEqual(cfg["tag_note"], "smoke")
        self.assertEqual(cfg_path.name, "config.json")

    def test_merges_user_config(self):
        user_cfg = self.tmp / "user_cfg.json"
        user_cfg.write_text(json.dumps({"dim": 32, "steps": 100}))
        cfg_path = tag._write_config(
            exp_id="exp2",
            topic="x",
            n=1,
            note=None,
            user_config_path=str(user_cfg),
        )
        cfg = json.loads(cfg_path.read_text())
        # user fields survive
        self.assertEqual(cfg["dim"], 32)
        self.assertEqual(cfg["steps"], 100)
        # tag fields also present
        self.assertEqual(cfg["tag_topic"], "x")
        self.assertEqual(cfg["tag_seq"], 1)


class RunCommandTests(unittest.TestCase):
    def test_captures_to_log(self):
        tmp = _make_tmp_repo()
        log = tmp / "log.txt"
        rc = tag._run_command(
            [sys.executable, "-c", "print('hi'); print('err', file=__import__('sys').stderr)"],
            log,
        )
        self.assertEqual(rc, 0)
        # stdout AND stderr captured together
        text = log.read_text()
        self.assertIn("hi", text)
        self.assertIn("err", text)

    def test_propagates_nonzero(self):
        tmp = _make_tmp_repo()
        log = tmp / "log.txt"
        rc = tag._run_command([sys.executable, "-c", "raise SystemExit(7)"], log)
        self.assertEqual(rc, 7)
        shutil.rmtree(tmp, ignore_errors=True)


class CleanTreeTests(unittest.TestCase):
    def test_clean_passes(self):
        tmp = _make_tmp_repo()
        try:
            tag._require_clean_tree(tmp)  # no exception
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dirty_exits(self):
        tmp = _make_tmp_repo()
        try:
            (tmp / "new.txt").write_text("hi")
            with self.assertRaises(SystemExit):
                tag._require_clean_tree(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
