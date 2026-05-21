from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from tabarena_json2vec.results import FIELDNAMES


RESULTS_CSV = EXPERIMENT_DIR / "results.csv"
RF_SCOPE = "matrix_4task_rf"
TASK_IDS = [363621, 363629, 363625, 363626]


def _format_float(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.6g}"


def _load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"results CSV not found: {path}")

    results = pd.read_csv(path)
    missing = [column for column in FIELDNAMES if column not in results.columns]
    if missing:
        raise ValueError(f"results CSV is missing column(s): {', '.join(missing)}")
    return results


def _scope_rows(results: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = results[
        results["scope"].eq(scope)
        & results["tid"].isin(TASK_IDS)
    ].copy()
    if len(rows) != len(TASK_IDS):
        raise ValueError(f"scope {scope!r} must have exactly {len(TASK_IDS)} rows; found {len(rows)}")

    duplicates = rows.duplicated(["tid"], keep=False)
    if duplicates.any():
        duplicate_tasks = rows.loc[duplicates, "tid"].drop_duplicates().astype(str).tolist()
        raise ValueError(f"scope {scope!r} has duplicate task rows: {', '.join(duplicate_tasks)}")

    missing_tasks = sorted(set(TASK_IDS) - set(rows["tid"].astype(int)))
    if missing_tasks:
        raise ValueError(f"scope {scope!r} is missing task(s): {missing_tasks}")

    missing_timings = rows[["time_train_s", "time_infer_s"]].isna().any(axis=1)
    if missing_timings.any():
        details = rows.loc[missing_timings, ["tid", "dataset", "time_train_s", "time_infer_s"]]
        raise ValueError(f"scope {scope!r} has missing timing fields:\n{details.to_string(index=False)}")

    return rows


def compare(candidate_scope: str, *, results_csv: Path = RESULTS_CSV) -> pd.DataFrame:
    results = _load_results(results_csv)
    candidate = _scope_rows(results, candidate_scope)
    rf = _scope_rows(results, RF_SCOPE)

    candidate = candidate.rename(
        columns={
            "metric_error": "candidate_error",
            "metric_error_val": "candidate_val_error",
            "time_train_s": "candidate_train_s",
            "time_infer_s": "candidate_infer_s",
            "model_or_config": "candidate_config",
        }
    )
    rf = rf.rename(
        columns={
            "metric_error": "rf_error",
            "metric_error_val": "rf_val_error",
            "time_train_s": "rf_train_s",
            "time_infer_s": "rf_infer_s",
            "model_or_config": "rf_config",
        }
    )

    joined = candidate[
        [
            "tid",
            "dataset",
            "problem_type",
            "candidate_config",
            "metric",
            "candidate_error",
            "candidate_val_error",
            "candidate_train_s",
            "candidate_infer_s",
        ]
    ].merge(
        rf[["tid", "metric", "rf_config", "rf_error", "rf_val_error", "rf_train_s", "rf_infer_s"]],
        on=["tid", "metric"],
        how="inner",
        validate="one_to_one",
    )

    if len(joined) != len(TASK_IDS):
        raise ValueError("candidate and RF scopes did not join to exactly four comparable rows")

    joined["gap_vs_rf"] = joined["candidate_error"].astype(float) - joined["rf_error"].astype(float)
    joined["wins_rf"] = joined["gap_vs_rf"] < 0.0
    return joined.sort_values("tid").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a fixed json2vec candidate scope against matrix_4task_rf.")
    parser.add_argument("candidate_scope", help="results.csv scope containing exactly four candidate rows")
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    matrix = compare(args.candidate_scope, results_csv=args.results_csv)
    display = matrix.copy()
    display["wins_rf"] = display["wins_rf"].map({True: "yes", False: ""})
    for column in [
        "candidate_error",
        "candidate_val_error",
        "candidate_train_s",
        "candidate_infer_s",
        "rf_error",
        "rf_val_error",
        "rf_train_s",
        "rf_infer_s",
        "gap_vs_rf",
    ]:
        display[column] = display[column].map(_format_float)

    print(f"Candidate scope: {args.candidate_scope}")
    print(
        display[
            [
                "tid",
                "dataset",
                "problem_type",
                "candidate_config",
                "metric",
                "candidate_error",
                "rf_error",
                "gap_vs_rf",
                "candidate_train_s",
                "candidate_infer_s",
                "wins_rf",
            ]
        ].to_string(index=False)
    )

    wins = int(matrix["wins_rf"].sum())
    aggregate = pd.DataFrame(
        [
            {
                "wins": wins,
                "losses": len(matrix) - wins,
                "median_candidate_train_s": matrix["candidate_train_s"].median(),
                "median_candidate_infer_s": matrix["candidate_infer_s"].median(),
            }
        ]
    )
    for column in ["median_candidate_train_s", "median_candidate_infer_s"]:
        aggregate[column] = aggregate[column].map(_format_float)

    print()
    print("Aggregate")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
