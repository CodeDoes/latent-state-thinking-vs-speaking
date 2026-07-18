"""End-to-end test of `cmd_run` against a fresh git repo.

Uses runpy to invoke _tag.py as a module with --dry-run / --pre-only etc.,
on an isolated git working tree where it has full freedom to stage + commit.
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
TAG_PY = REPO_ROOT / "src" / "_tag.py"


def _bootstrap_tmp_repo():
    """Create a tmp repo that has _tag.py and the test inside it on a main
    branch with one commit, plus an `experiments/` empty dir tracked.
    Mirrors the real repo structure enough for cmd_run to operate.
    """
    tmp = Path(tempfile.mkdtemp(prefix="tag_run_e2e_"))
    subprocess.run(["git", "-C", str(tmp), "init", "--initial-branch=main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp), "config", "user.email", "test@test"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp), "config", "user.name", "Test"], check=True, capture_output=True)

    (tmp / "experiments").mkdir()
    (tmp / "src").mkdir()
    (tmp / "README.md").write_text("hello\n")
    (tmp / "src" / "dummy_train.py").write_text("print('training ok')\n")
    (tmp / "experiments").mkdir(exist_ok=True)

    subprocess.run(["git", "-C", str(tmp), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp), "commit", "-m", "init"], check=True, capture_output=True)
    return tmp


def _run_tag(argv, cwd):
    """Invoke the real _tag.py CLI inside cwd."""
    cmd = [sys.executable, str(TAG_PY)] + argv
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return res


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = _bootstrap_tmp_repo()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_prints_noop(self):
        res = _run_tag(
            ["run", "--id", "smoke1", "--topic", "smoke_topic",
             "--note", "dry run note", "--dry-run",
             "--command", sys.executable, str(TAG_PY), "list", "--exp"],
            cwd=self.tmp,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertIn("Experiment:", res.stdout)
        self.assertIn("dry-run", res.stdout)
        # No tags created
        tags = subprocess.run(
            ["git", "-C", str(self.tmp), "tag"],
            capture_output=True, text=True,
        ).stdout.split()
        self.assertEqual(tags, [])

    def test_pre_only_mints_pre_tag(self):
        res = _run_tag(
            ["run", "--id", "smoke2", "--topic", "smoke_topic", "--pre-only",
             "--command", sys.executable, "-c", "print('noop')"],
            cwd=self.tmp,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        tags = subprocess.run(
            ["git", "-C", str(self.tmp), "tag"],
            capture_output=True, text=True,
        ).stdout.split()
        self.assertIn("exp/smoke_topic/001-pre", tags)
        self.assertNotIn("exp/smoke_topic/001-post", tags)
        # config written
        cfg = json.loads((self.tmp / "experiments" / "smoke2" / "config.json").read_text())
        self.assertEqual(cfg["tag_seq"], 1)
        self.assertEqual(cfg["tag_topic"], "smoke_topic")

    def test_pre_then_post_pair(self):
        # 1) --pre-only
        res = _run_tag(
            ["run", "--id", "smoke3", "--topic", "smoke_topic", "--pre-only"],
            cwd=self.tmp,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        # 2) run something that writes a metrics file
        (self.tmp / "experiments" / "smoke3").mkdir(exist_ok=True)
        (self.tmp / "experiments" / "smoke3" / "metrics.json").write_text('{"final_loss":0.42}')
        # 3) --post-only reuses the seq 001
        res = _run_tag(
            ["run", "--id", "smoke3", "--topic", "smoke_topic", "--post-only",
             "--note", "smoke result"],
            cwd=self.tmp,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        tags = subprocess.run(
            ["git", "-C", str(self.tmp), "tag"],
            capture_output=True, text=True,
        ).stdout.split()
        self.assertIn("exp/smoke_topic/001-pre", tags)
        self.assertIn("exp/smoke_topic/001-post", tags)
        # The diff between -pre and -post contains metrics.json
        diff = subprocess.run(
            ["git", "-C", str(self.tmp), "diff", "--name-only",
             "exp/smoke_topic/001-pre", "exp/smoke_topic/001-post"],
            capture_output=True, text=True,
        ).stdout.split()
        self.assertIn("experiments/smoke3/metrics.json", diff)

    def test_post_only_requires_pre_tag(self):
        # No -pre tag exists → should refuse
        res = _run_tag(
            ["run", "--id", "smoke4", "--topic", "smoke_topic", "--post-only"],
            cwd=self.tmp,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--post-only", res.stderr)

    def test_full_run_in_one_call(self):
        res = _run_tag(
            ["run", "--id", "smoke5", "--topic", "smoke_topic",
             "--note", "one-shot",
             "--command", sys.executable, str(self.tmp / "src" / "dummy_train.py")],
            cwd=self.tmp,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        tags = subprocess.run(
            ["git", "-C", str(self.tmp), "tag"],
            capture_output=True, text=True,
        ).stdout.split()
        self.assertIn("exp/smoke_topic/001-pre", tags)
        self.assertIn("exp/smoke_topic/001-post", tags)
        log_text = (self.tmp / "experiments" / "smoke5" / "train.log").read_text()
        self.assertIn("training ok", log_text)


if __name__ == "__main__":
    unittest.main()
