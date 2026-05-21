from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tabarena.benchmark.experiment import run_experiments_new
from tabarena.utils.config_utils import ConfigGenerator, generate_holdout_experiments

from tabarena_json2vec import JSON2VecTabArenaModel
from tabarena_json2vec.config import JSON2VecConfig, RandomForestConfig, TabArenaRunConfig
from tabarena_json2vec.results import append_results_csv
from tabarena_json2vec.tabarena import print_results, select_tasks


def json2vec_experiments(config: JSON2VecConfig) -> Sequence[Any]:
    params = config.autogluon_params()
    if config.holdout:
        return generate_holdout_experiments(
            model_cls=JSON2VecTabArenaModel,
            configs=[params],
            name_id_prefix="c",
            time_limit=None,
        )

    generator = ConfigGenerator(
        model_cls=JSON2VecTabArenaModel,
        manual_configs=[params],
        search_space={},
    )
    return generator.generate_all_bag_experiments(
        num_random_configs=0,
        num_bag_folds=config.num_bag_folds,
        fold_fitting_strategy="sequential_local",
        time_limit=None,
    )


def random_forest_experiments(config: RandomForestConfig) -> Sequence[Any]:
    from tabarena.models.random_forest.generate import gen_randomforest

    return gen_randomforest.generate_all_bag_experiments(
        num_random_configs=config.num_random_configs,
        num_bag_folds=config.num_bag_folds,
        fold_fitting_strategy="sequential_local",
        time_limit=None,
    )


def run_tabarena(
    *,
    run_config: TabArenaRunConfig,
    model_experiments: Sequence[Any],
    allow_openml_suite: bool,
    model_or_config: str,
) -> list[dict]:
    tasks = select_tasks(
        task_ids_env=run_config.task_ids,
        max_instances=run_config.max_instances,
        task_limit=run_config.task_limit,
        allow_openml_suite=allow_openml_suite,
    )
    results = run_experiments_new(
        output_dir=str(run_config.output_dir),
        model_experiments=model_experiments,
        tasks=tasks,
        repetitions_mode=run_config.repetitions_mode,
        cache_mode=run_config.cache_mode,
        raise_on_failure=run_config.raise_on_failure,
        debug_mode=run_config.debug_mode,
    )
    print_results(results)
    if run_config.results_csv is not None:
        append_results_csv(
            run_config.results_csv,
            scope=run_config.results_scope,
            model_or_config=model_or_config,
            results=results,
            notes=run_config.results_notes,
        )
        print(f"Wrote result rows to {run_config.results_csv}", flush=True)
    return results


def run_json2vec_from_env(run_config: TabArenaRunConfig, model_config: JSON2VecConfig) -> list[dict]:
    print(
        "Running json2vec TabArena benchmark "
        f"with output_dir={run_config.output_dir}, repetitions_mode={run_config.repetitions_mode}, "
        f"max_epochs={model_config.max_epochs}, d_model={model_config.d_model}, "
        f"batch_size={model_config.batch_size}, lr={model_config.lr}, "
        f"attention={model_config.attention}, pretrain_epochs={model_config.pretrain_epochs}, "
        f"pretrain_p_mask={model_config.pretrain_p_mask}, "
        f"num_bag_folds={model_config.num_bag_folds}, holdout={model_config.holdout}",
        flush=True,
    )
    return run_tabarena(
        run_config=run_config,
        model_experiments=json2vec_experiments(model_config),
        allow_openml_suite=True,
        model_or_config=model_config.label(),
    )


def run_random_forest_from_env(run_config: TabArenaRunConfig, model_config: RandomForestConfig) -> list[dict]:
    print(
        "Running TabArena RandomForest baseline "
        f"with output_dir={run_config.output_dir}, repetitions_mode={run_config.repetitions_mode}, "
        f"num_random_configs={model_config.num_random_configs}, "
        f"num_bag_folds={model_config.num_bag_folds}",
        flush=True,
    )
    return run_tabarena(
        run_config=run_config,
        model_experiments=random_forest_experiments(model_config),
        allow_openml_suite=False,
        model_or_config=model_config.label(),
    )
