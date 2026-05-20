from __future__ import annotations

import os
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from tabarena_json2vec.paths import EXPERIMENT_DIR as OUTPUT_ROOT
from tabarena_json2vec.paths import configure_import_paths

configure_import_paths()

from tabarena.benchmark.experiment import run_experiments_new
from tabarena.models.random_forest.generate import gen_randomforest
from tabarena_json2vec.tabarena import print_results, select_tasks


OUT_DIR = OUTPUT_ROOT / "tabarena_rf_out"


def main() -> None:
    num_bag_folds = int(os.environ.get("RF_NUM_BAG_FOLDS", "2"))
    num_random_configs = int(os.environ.get("RF_NUM_RANDOM_CONFIGS", "0"))
    tasks = select_tasks(
        task_ids_env=os.environ.get("TABARENA_TASK_IDS"),
        max_instances=int(os.environ.get("TABARENA_MAX_INSTANCES", "0")),
        task_limit=int(os.environ.get("TABARENA_TASK_LIMIT", "1")),
        allow_openml_suite=False,
    )
    experiments = gen_randomforest.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        num_bag_folds=num_bag_folds,
        fold_fitting_strategy="sequential_local",
        time_limit=None,
    )

    print(
        "Running TabArena RF TabArena-Lite baseline "
        f"with tasks={tasks}, num_random_configs={num_random_configs}, "
        f"num_bag_folds={num_bag_folds}",
        flush=True,
    )
    results = run_experiments_new(
        output_dir=str(OUT_DIR),
        model_experiments=experiments,
        tasks=tasks,
        repetitions_mode="TabArena-Lite",
        cache_mode=os.environ.get("TABARENA_CACHE_MODE", "default"),
        raise_on_failure=True,
        debug_mode=True,
    )
    print_results(results)


if __name__ == "__main__":
    main()
