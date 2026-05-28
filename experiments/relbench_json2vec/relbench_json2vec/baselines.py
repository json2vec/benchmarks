from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BaselineTaskResult:
    dataset: str
    task: str
    task_type: str
    model_or_config: str
    metrics_by_split: dict[str, dict[str, float]]
    time_train_s: float
    time_infer_s: dict[str, float]


def _task_type_value(task: Any) -> str:
    return getattr(task.task_type, "value", str(task.task_type))


def _entity_col(task: Any, train_table: Any) -> str | None:
    if hasattr(task, "entity_col"):
        return task.entity_col
    keys = list(train_table.fkey_col_to_pkey_table.keys())
    return keys[0] if keys else None


def _entity_stat_prediction(train_table: Any, pred_table: Any, *, task: Any, statistic: str, fill: float) -> np.ndarray:
    entity_col = _entity_col(task, train_table)
    if entity_col is None or entity_col not in pred_table.df:
        return np.full(len(pred_table), fill, dtype=np.float32)
    grouped = train_table.df.groupby(entity_col).agg({task.target_col: statistic})
    grouped = grouped.rename(columns={task.target_col: "__prediction__"})
    merged = pred_table.df.merge(grouped, how="left", on=entity_col)
    return merged["__prediction__"].fillna(fill).astype(float).to_numpy(dtype=np.float32)


def _predict(name: str, task: Any, train_table: Any, pred_table: Any) -> np.ndarray:
    target = train_table.df[task.target_col].dropna()
    task_type = _task_type_value(task)
    if task_type == "binary_classification":
        if name == "majority":
            majority = int(target.mode().iloc[0])
            return np.full(len(pred_table), majority, dtype=np.float32)
        if name == "entity_mean":
            return _entity_stat_prediction(
                train_table,
                pred_table,
                task=task,
                statistic="mean",
                fill=float(target.mean()),
            )
    elif task_type == "regression":
        if name == "global_mean":
            return np.full(len(pred_table), float(target.mean()), dtype=np.float32)
        if name == "global_median":
            return np.full(len(pred_table), float(target.median()), dtype=np.float32)
        if name == "entity_mean":
            return _entity_stat_prediction(
                train_table,
                pred_table,
                task=task,
                statistic="mean",
                fill=float(target.mean()),
            )
        if name == "entity_median":
            return _entity_stat_prediction(
                train_table,
                pred_table,
                task=task,
                statistic="median",
                fill=float(target.median()),
            )
    raise ValueError(f"unsupported baseline {name!r} for task type {task_type!r}")


def baseline_names(task: Any) -> tuple[str, ...]:
    task_type = _task_type_value(task)
    if task_type == "binary_classification":
        return ("majority", "entity_mean")
    if task_type == "regression":
        return ("global_mean", "global_median", "entity_mean", "entity_median")
    raise ValueError(f"unsupported RelBench task type for v1: {task_type}")


def evaluate_baselines(*, dataset_name: str, task_name: str, task: Any) -> list[BaselineTaskResult]:
    train_start = time.time()
    train_table = task.get_table("train", mask_input_cols=False)
    time_train_s = time.time() - train_start
    results: list[BaselineTaskResult] = []

    for name in baseline_names(task):
        metrics_by_split: dict[str, dict[str, float]] = {}
        time_infer_s: dict[str, float] = {}
        for split in ("val", "test"):
            table = task.get_table(split, mask_input_cols=False)
            pred_table = task.get_table(split)
            infer_start = time.time()
            pred = _predict(name, task, train_table, pred_table)
            time_infer_s[split] = time.time() - infer_start
            metrics_by_split[split] = {
                metric_name: float(metric_value)
                for metric_name, metric_value in task.evaluate(
                    pred,
                    target_table=table,
                ).items()
            }
        results.append(
            BaselineTaskResult(
                dataset=dataset_name,
                task=task_name,
                task_type=_task_type_value(task),
                model_or_config=name,
                metrics_by_split=metrics_by_split,
                time_train_s=time_train_s,
                time_infer_s=time_infer_s,
            )
        )
    return results
