import json
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import polars as pl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compact import compact_experiment_runs

TEST_TMP_ROOT = Path(".tmp-py") / "test-compact"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class TestCompactResults(unittest.TestCase):
    def make_dir(self, name: str) -> Path:
        path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def write_csv(self, path: Path, data: dict[str, list]):
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(data).write_csv(path)

    def test_compact_results_merges_runs_and_deletes_allowlisted_artifacts(self):
        experiment_dir = self.make_dir("experiment")
        (experiment_dir / "results.csv").write_text(
            "metric,value\nloss,1.0\n", encoding="utf-8"
        )
        (experiment_dir / "config.json").write_text("{}", encoding="utf-8")

        for seed in (0, 1):
            run_dir = experiment_dir / str(seed)
            self.write_csv(
                run_dir / "results" / "server.csv",
                {"train_loss": [1.0 + seed, 2.0 + seed]},
            )
            self.write_csv(
                run_dir / "results" / "client_000.csv",
                {"train_loss": [3.0 + seed]},
            )
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            (run_dir / "models_info").mkdir(parents=True, exist_ok=True)
            (run_dir / "models").mkdir(parents=True, exist_ok=True)
            (run_dir / "logs" / "server.log").write_text("log", encoding="utf-8")
            (run_dir / "models_info" / "server.svg").write_text("svg", encoding="utf-8")
            (run_dir / "models" / "server_last.pt").write_text(
                "checkpoint", encoding="utf-8"
            )

        summary = compact_experiment_runs(experiment_dir)

        server_compact = pl.read_csv(experiment_dir / "compact" / "server.csv")
        clients_compact = pl.read_csv(experiment_dir / "compact" / "clients.csv")
        manifest = json.loads(
            (experiment_dir / "compact" / "manifest.json").read_text(encoding="utf-8")
        )

        self.assertEqual(summary["runs"], 2)
        self.assertEqual(server_compact.height, 4)
        self.assertEqual(clients_compact.height, 2)
        self.assertEqual(sorted(server_compact["seed"].to_list()), [0, 0, 1, 1])
        self.assertEqual(
            sorted(clients_compact["client"].to_list()), ["client_000", "client_000"]
        )
        self.assertEqual(manifest["runs"], 2)

        for seed in (0, 1):
            run_dir = experiment_dir / str(seed)
            self.assertTrue((run_dir / "results" / "server.csv").exists())
            self.assertFalse((run_dir / "results" / "client_000.csv").exists())
            self.assertFalse((run_dir / "logs").exists())
            self.assertFalse((run_dir / "models_info").exists())
            self.assertTrue((run_dir / "models" / "server_last.pt").exists())

        self.assertTrue((experiment_dir / "results.csv").exists())
        self.assertTrue((experiment_dir / "config.json").exists())

    def test_compact_results_noops_when_no_seed_runs_exist(self):
        experiment_dir = self.make_dir("empty-experiment")
        summary = compact_experiment_runs(experiment_dir)

        self.assertEqual(summary["runs"], 0)
        self.assertEqual(summary["server_rows"], 0)
        self.assertEqual(summary["client_rows"], 0)
        self.assertEqual(summary["generated_files"], [])
        self.assertEqual(summary["deleted_paths"], [])
        self.assertTrue((experiment_dir / "compact" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
