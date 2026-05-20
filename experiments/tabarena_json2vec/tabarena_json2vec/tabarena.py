from __future__ import annotations

import openml
import pandas as pd

from tabarena_json2vec.paths import METADATA_PATH


def parse_task_ids(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(task_id.strip()) for task_id in raw.split(",") if task_id.strip()]


def select_tasks(
    *,
    task_ids_env: str | None,
    max_instances: int,
    task_limit: int,
    allow_openml_suite: bool,
) -> list[int]:
    task_ids = parse_task_ids(task_ids_env)
    if task_ids is not None:
        return task_ids

    if max_instances > 0:
        metadata = pd.read_csv(METADATA_PATH)
        metadata = metadata[metadata["NumberOfInstances"].le(max_instances)]
        metadata = metadata.sort_values(["NumberOfInstances", "tid"])
        if task_limit > 0:
            metadata = metadata.head(task_limit)
        print(
            "Selected small TabArena tasks:\n"
            + metadata[["tid", "name", "task_type", "NumberOfInstances", "NumberOfFeatures"]].to_string(index=False),
            flush=True,
        )
        return metadata["tid"].astype(int).tolist()

    if allow_openml_suite:
        return openml.study.get_suite("tabarena-v0.1").tasks[:task_limit]

    raise ValueError("Set TABARENA_TASK_IDS or TABARENA_MAX_INSTANCES")


def print_results(results: list[dict]) -> None:
    print(f"Completed {len(results)} result(s).", flush=True)
    for result in results:
        task = result["task_metadata"]
        print(
            {
                "tid": task["tid"],
                "name": task["name"],
                "problem_type": result["problem_type"],
                "metric": result["metric"],
                "metric_error": float(result["metric_error"]),
                "metric_error_val": float(result.get("metric_error_val", float("nan"))),
                "time_train_s": float(result["time_train_s"]),
                "time_infer_s": float(result["time_infer_s"]),
            },
            flush=True,
        )
