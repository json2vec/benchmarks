from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "scope",
    "tid",
    "dataset",
    "problem_type",
    "model_or_config",
    "metric",
    "metric_error",
    "metric_error_val",
    "time_train_s",
    "time_infer_s",
    "notes",
]


def append_results_csv(
    path: Path,
    *,
    scope: str,
    model_or_config: str,
    results: list[dict[str, Any]],
    notes: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if should_write_header:
            writer.writeheader()

        for result in results:
            task = result["task_metadata"]
            writer.writerow(
                {
                    "scope": scope,
                    "tid": task["tid"],
                    "dataset": task["name"],
                    "problem_type": result["problem_type"],
                    "model_or_config": model_or_config,
                    "metric": result["metric"],
                    "metric_error": float(result["metric_error"]),
                    "metric_error_val": float(result.get("metric_error_val", float("nan"))),
                    "time_train_s": float(result["time_train_s"]),
                    "time_infer_s": float(result["time_infer_s"]),
                    "notes": notes,
                }
            )
