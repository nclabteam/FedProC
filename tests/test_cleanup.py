import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cleanup import cleanup_interrupted_run, resolve_cleanup_path

TEST_TMP_ROOT = Path(".tmp-py") / "test-cleanup"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class TestCleanup(unittest.TestCase):
    def make_dir(self, name: str) -> Path:
        path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_resolve_cleanup_path_rejects_project_root(self):
        tmpdir = self.make_dir("project-root")
        with self.assertRaises(ValueError):
            resolve_cleanup_path(save_path=str(tmpdir), project_root=str(tmpdir))

    def test_resolve_cleanup_path_rejects_outside_project(self):
        project_dir = self.make_dir("project")
        outside_dir = self.make_dir("outside")
        with self.assertRaises(ValueError):
            resolve_cleanup_path(
                save_path=str(outside_dir),
                project_root=str(project_dir),
            )

    def test_cleanup_interrupted_run_removes_run_directory(self):
        project_dir = self.make_dir("project")
        run_dir = project_dir / "exp" / "1"
        run_dir.mkdir(parents=True)
        (run_dir / "results.csv").write_text("ok", encoding="utf-8")

        with patch("utils.cleanup.shutil.rmtree") as mock_rmtree:
            removed_path = cleanup_interrupted_run(
                save_path=str(run_dir),
                project_root=str(project_dir),
            )

        self.assertEqual(removed_path, run_dir.resolve())
        mock_rmtree.assert_called_once()
        self.assertEqual(mock_rmtree.call_args.args[0], run_dir.resolve())


if __name__ == "__main__":
    unittest.main()
