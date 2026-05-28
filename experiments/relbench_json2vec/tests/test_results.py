from __future__ import annotations

import csv

from relbench_json2vec.results import FIELDNAMES, append_results_csv


def test_append_results_csv_writes_normalized_schema(tmp_path) -> None:
    path = tmp_path / "results.csv"
    append_results_csv(
        path,
        [
            {
                "scope": "scope",
                "dataset": "fake",
                "task": "binary",
                "task_type": "binary_classification",
                "model_or_config": "model",
                "split": "val",
                "metric": "accuracy",
                "value": 1.0,
                "time_train_s": 0.1,
                "time_infer_s": 0.01,
                "notes": "note",
            }
        ],
    )
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["metric"] == "accuracy"
    assert list(rows[0].keys()) == FIELDNAMES
