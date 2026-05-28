from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "scope",
    "dataset",
    "task",
    "task_type",
    "model_or_config",
    "split",
    "metric",
    "value",
    "time_train_s",
    "time_infer_s",
    "notes",
]


def append_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
