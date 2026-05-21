from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from tabarena_json2vec.results import FIELDNAMES


RESULTS_CSV = EXPERIMENT_DIR / "results.csv"
TASK_IDS = [363621, 363629, 363625, 363626]
SCOPES = {
    "matrix_4task_json2vec_fixed": "json2vec_fixed",
    "matrix_4task_rf": "random_forest",
}


def _format_float(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.6g}"


def _load_matrix() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"results CSV not found: {RESULTS_CSV}")

    results = pd.read_csv(RESULTS_CSV)
    missing_columns = [column for column in FIELDNAMES if column not in results.columns]
    if missing_columns:
        raise ValueError(f"results CSV is missing column(s): {', '.join(missing_columns)}")

    matrix = results[
        results["scope"].isin(SCOPES)
        & results["tid"].isin(TASK_IDS)
    ].copy()
    matrix["model"] = matrix["scope"].map(SCOPES)

    expected_rows = len(TASK_IDS) * len(SCOPES)
    if len(matrix) != expected_rows:
        raise ValueError(f"expected {expected_rows} matrix rows, found {len(matrix)}")

    duplicates = matrix.duplicated(["tid", "model"], keep=False)
    if duplicates.any():
        duplicate_keys = matrix.loc[duplicates, ["tid", "model"]].drop_duplicates()
        raise ValueError(f"duplicate matrix rows found:\n{duplicate_keys.to_string(index=False)}")

    observed_tasks = set(matrix["tid"].astype(int))
    missing_tasks = sorted(set(TASK_IDS) - observed_tasks)
    if missing_tasks:
        raise ValueError(f"missing task(s): {missing_tasks}")

    missing_timings = matrix[["time_train_s", "time_infer_s"]].isna().any(axis=1)
    if missing_timings.any():
        rows = matrix.loc[missing_timings, ["tid", "model", "time_train_s", "time_infer_s"]]
        raise ValueError(f"matrix rows with missing timings:\n{rows.to_string(index=False)}")

    return matrix.sort_values(["tid", "model"]).reset_index(drop=True)


def _with_winners(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in matrix.groupby("tid", sort=False):
        winner_idx = group["metric_error"].astype(float).idxmin()
        winner_model = matrix.loc[winner_idx, "model"]
        task_rows = group.copy()
        task_rows["winner"] = task_rows["model"].eq(winner_model).map({True: "yes", False: ""})
        rows.append(task_rows)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    matrix = _with_winners(_load_matrix())
    display_columns = [
        "tid",
        "dataset",
        "problem_type",
        "model",
        "metric",
        "metric_error",
        "metric_error_val",
        "time_train_s",
        "time_infer_s",
        "winner",
    ]
    display = matrix[display_columns].copy()
    for column in ["metric_error", "metric_error_val", "time_train_s", "time_infer_s"]:
        display[column] = display[column].map(_format_float)

    print("4-task benchmark matrix")
    print(display.to_string(index=False))

    aggregate = (
        matrix.groupby("model", sort=True)
        .agg(
            wins=("winner", lambda values: int((values == "yes").sum())),
            median_train_s=("time_train_s", "median"),
            median_infer_s=("time_infer_s", "median"),
        )
        .reset_index()
    )
    for column in ["median_train_s", "median_infer_s"]:
        aggregate[column] = aggregate[column].map(_format_float)

    print()
    print("Aggregate")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
