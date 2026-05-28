from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
CLI = BENCHMARKS_DIR / "scripts" / "benchmark.py"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=BENCHMARKS_DIR,
        check=True,
        text=True,
        capture_output=True,
    )


def test_relbench_inspect_fake_binary() -> None:
    result = run_cli("relbench", "inspect", "--fake", "binary", "--limit", "1")
    payload = json.loads(result.stdout)

    assert payload["dataset"] == "fake"
    assert payload["task"] == "binary"
    assert payload["splits"]["train"] == 3
    assert len(payload["sample_records"]) == 1


def test_relbench_run_dry_run(tmp_path: Path) -> None:
    result = run_cli(
        "relbench",
        "run",
        "--model",
        "json2vec",
        "--dry-run",
        "--tasks",
        "rel-f1:driver-dnf",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "dry",
        "--no-download",
    )
    payload = json.loads(result.stdout)

    assert payload["status"] == "dry-run"
    assert payload["models"] == ["json2vec"]
    assert (tmp_path / "dry" / "manifest.json").exists()


def test_tabarena_run_dry_run_resolves_config() -> None:
    result = run_cli(
        "tabarena",
        "run",
        "--model",
        "json2vec",
        "--dry-run",
        "--task-ids",
        "363621",
        "--d-model",
        "24",
        "--batch-size",
        "8",
        "--no-holdout",
    )
    payload = json.loads(result.stdout)

    assert payload["status"] == "dry-run"
    assert payload["suite"] == "tabarena"
    assert payload["run_config"]["task_ids"] == "363621"
    assert payload["model_config"]["d_model"] == 24
    assert payload["model_config"]["batch_size"] == 8
