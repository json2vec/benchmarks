from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CLI = EXPERIMENT_DIR / "scripts" / "benchmark.py"


def test_cli_inspect_fake_without_lightning_import() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI), "inspect", "--fake", "binary", "--limit", "1"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["dataset"] == "fake"
    assert payload["splits"]["train"] == 3
    assert payload["schema_fields"]


def test_cli_run_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "run",
            "--dry-run",
            "--tasks",
            "rel-f1:driver-dnf",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "dry",
            "--no-download",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry-run"
    assert (tmp_path / "dry" / "manifest.json").exists()
