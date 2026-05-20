from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/tmp/tabarena/tabarena")

import pandas as pd

from tabarena.benchmark.experiment import run_experiments_new
from tabarena.models.random_forest.generate import gen_randomforest


OUT_DIR = ROOT / "experiments" / "tabarena_json2vec" / "tabarena_rf_out"
METADATA_PATH = Path("/tmp/tabarena/tabarena/tabarena/nips2025_utils/metadata/task_metadata_tabarena51.csv")


def main() -> None:
    task_limit = int(os.environ.get("TABARENA_TASK_LIMIT", "1"))
    max_instances = int(os.environ.get("TABARENA_MAX_INSTANCES", "0"))
    task_ids_env = os.environ.get("TABARENA_TASK_IDS")
    num_bag_folds = int(os.environ.get("RF_NUM_BAG_FOLDS", "2"))
    num_random_configs = int(os.environ.get("RF_NUM_RANDOM_CONFIGS", "0"))
    cache_mode = os.environ.get("TABARENA_CACHE_MODE", "default")

    if task_ids_env:
        task_ids = [int(task_id.strip()) for task_id in task_ids_env.split(",") if task_id.strip()]
    elif max_instances > 0:
        metadata = pd.read_csv(METADATA_PATH)
        metadata = metadata[metadata["NumberOfInstances"].le(max_instances)]
        metadata = metadata.sort_values(["NumberOfInstances", "tid"])
        if task_limit > 0:
            metadata = metadata.head(task_limit)
        task_ids = metadata["tid"].astype(int).tolist()
        print(
            "Selected small TabArena tasks:\n"
            + metadata[["tid", "name", "task_type", "NumberOfInstances", "NumberOfFeatures"]].to_string(index=False),
            flush=True,
        )
    else:
        raise ValueError("Set TABARENA_TASK_IDS or TABARENA_MAX_INSTANCES")

    experiments = gen_randomforest.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        num_bag_folds=num_bag_folds,
        fold_fitting_strategy="sequential_local",
        time_limit=None,
    )

    print(
        "Running TabArena RF TabArena-Lite baseline "
        f"with tasks={task_ids}, num_random_configs={num_random_configs}, "
        f"num_bag_folds={num_bag_folds}",
        flush=True,
    )
    results = run_experiments_new(
        output_dir=str(OUT_DIR),
        model_experiments=experiments,
        tasks=task_ids,
        repetitions_mode="TabArena-Lite",
        cache_mode=cache_mode,
        raise_on_failure=True,
        debug_mode=True,
    )
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


if __name__ == "__main__":
    main()
