from __future__ import annotations

from typing import Any

from relbench_json2vec.baselines import BaselineTaskResult, evaluate_baselines
from relbench_json2vec.config import JSON2VecConfig, RelBenchRunConfig
from relbench_json2vec.model import JSON2VecTaskResult, fit_and_evaluate_json2vec
from relbench_json2vec.paths import configure_import_paths
from relbench_json2vec.results import append_results_csv


def _load_task(dataset_name: str, task_name: str, *, download: bool, cache_dir: str | None = None) -> Any:
    from relbench.tasks import get_task

    kwargs: dict[str, Any] = {"download": download}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    try:
        return get_task(dataset_name, task_name, **kwargs)
    except TypeError:
        kwargs.pop("cache_dir", None)
        return get_task(dataset_name, task_name, **kwargs)


def _rows_from_result(
    *,
    scope: str,
    notes: str,
    result: JSON2VecTaskResult | BaselineTaskResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_notes = notes
    warnings = getattr(result, "warnings", [])
    if warnings:
        warning_note = "; ".join(warnings[:5])
        extra_notes = f"{notes}; warnings: {warning_note}" if notes else f"warnings: {warning_note}"

    for split, metrics in result.metrics_by_split.items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "scope": scope,
                    "dataset": result.dataset,
                    "task": result.task,
                    "task_type": result.task_type,
                    "model_or_config": result.model_or_config,
                    "split": split,
                    "metric": metric,
                    "value": value,
                    "time_train_s": result.time_train_s,
                    "time_infer_s": result.time_infer_s.get(split, ""),
                    "notes": extra_notes,
                }
            )
    return rows


def _print_result(result: JSON2VecTaskResult | BaselineTaskResult) -> None:
    print(
        {
            "dataset": result.dataset,
            "task": result.task,
            "task_type": result.task_type,
            "model_or_config": result.model_or_config,
            "metrics_by_split": result.metrics_by_split,
            "time_train_s": result.time_train_s,
            "time_infer_s": result.time_infer_s,
        },
        flush=True,
    )


def run_json2vec_from_env(run_config: RelBenchRunConfig, model_config: JSON2VecConfig) -> list[JSON2VecTaskResult]:
    configure_import_paths()
    import json2vec as j2v

    print(
        "Running json2vec RelBench benchmark "
        f"tasks={run_config.tasks}, d_model={model_config.d_model}, batch_size={model_config.batch_size}, "
        f"max_epochs={model_config.max_epochs}, history_limit={model_config.history_limit}",
        flush=True,
    )
    results: list[JSON2VecTaskResult] = []
    rows: list[dict[str, Any]] = []
    for dataset_name, task_name in run_config.tasks:
        task = _load_task(
            dataset_name,
            task_name,
            download=run_config.download,
            cache_dir=str(run_config.cache_dir / dataset_name / task_name),
        )
        result = fit_and_evaluate_json2vec(
            j2v=j2v,
            dataset_name=dataset_name,
            task_name=task_name,
            task=task,
            config=model_config,
        )
        _print_result(result)
        results.append(result)
        rows.extend(_rows_from_result(scope=run_config.results_scope, notes=run_config.results_notes, result=result))

    if run_config.results_csv is not None and rows:
        append_results_csv(run_config.results_csv, rows)
        print(f"Wrote result rows to {run_config.results_csv}", flush=True)
    return results


def run_baselines_from_env(run_config: RelBenchRunConfig) -> list[BaselineTaskResult]:
    print(f"Running RelBench simple baselines tasks={run_config.tasks}", flush=True)
    results: list[BaselineTaskResult] = []
    rows: list[dict[str, Any]] = []
    for dataset_name, task_name in run_config.tasks:
        task = _load_task(
            dataset_name,
            task_name,
            download=run_config.download,
            cache_dir=str(run_config.cache_dir / dataset_name / task_name),
        )
        task_results = evaluate_baselines(dataset_name=dataset_name, task_name=task_name, task=task)
        for result in task_results:
            _print_result(result)
            results.append(result)
            rows.extend(
                _rows_from_result(scope=run_config.results_scope, notes=run_config.results_notes, result=result)
            )
    if run_config.results_csv is not None and rows:
        append_results_csv(run_config.results_csv, rows)
        print(f"Wrote result rows to {run_config.results_csv}", flush=True)
    return results
