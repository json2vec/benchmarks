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
from tabarena.utils.config_utils import ConfigGenerator, generate_holdout_experiments
from tabarena_json2vec import JSON2VecTabArenaModel
from tabarena_json2vec.tabarena import print_results, select_tasks


OUT_DIR = OUTPUT_ROOT / "tabarena_out"


def json2vec_config_from_env() -> dict:
    lr = float(os.environ.get("JSON2VEC_LR", "0.001"))
    return {
        "d_model": int(os.environ.get("JSON2VEC_D_MODEL", "16")),
        "batch_size": int(os.environ.get("JSON2VEC_BATCH_SIZE", "128")),
        "max_epochs": int(os.environ.get("JSON2VEC_MAX_EPOCHS", "2")),
        "lr": lr,
        "weight_decay": float(os.environ.get("JSON2VEC_WEIGHT_DECAY", "0.0")),
        "pretrain_epochs": int(os.environ.get("JSON2VEC_PRETRAIN_EPOCHS", "0")),
        "pretrain_p_mask": float(os.environ.get("JSON2VEC_PRETRAIN_P_MASK", "0.0")),
        "pretrain_lr": float(os.environ.get("JSON2VEC_PRETRAIN_LR", str(lr))),
        "max_cat_vocab_size": 512,
        "accelerator": os.environ.get("JSON2VEC_ACCELERATOR", "mps"),
        "attention": os.environ.get("JSON2VEC_ATTENTION", "none"),
        "dropout": float(os.environ.get("JSON2VEC_DROPOUT", "0.0")),
        "p_mask": float(os.environ.get("JSON2VEC_P_MASK", "0.0")),
        "p_target": float(os.environ.get("JSON2VEC_P_TARGET", "0.0")),
        "n_layers": int(os.environ.get("JSON2VEC_N_LAYERS", "1")),
        "n_linear": int(os.environ.get("JSON2VEC_N_LINEAR", "1")),
        "n_heads": int(os.environ.get("JSON2VEC_N_HEADS", "4")),
        "random_seed": int(os.environ.get("JSON2VEC_RANDOM_SEED", "0")),
    }


def experiments_from_config(config: dict):
    if os.environ.get("JSON2VEC_HOLDOUT", "0").lower() in {"1", "true", "yes"}:
        return generate_holdout_experiments(
            model_cls=JSON2VecTabArenaModel,
            configs=[config],
            name_id_prefix="c",
            time_limit=None,
        )

    generator = ConfigGenerator(
        model_cls=JSON2VecTabArenaModel,
        manual_configs=[config],
        search_space={},
    )
    return generator.generate_all_bag_experiments(
        num_random_configs=0,
        num_bag_folds=int(os.environ.get("JSON2VEC_NUM_BAG_FOLDS", "2")),
        fold_fitting_strategy="sequential_local",
        time_limit=None,
    )


def main() -> None:
    config = json2vec_config_from_env()
    tasks = select_tasks(
        task_ids_env=os.environ.get("TABARENA_TASK_IDS"),
        max_instances=int(os.environ.get("TABARENA_MAX_INSTANCES", "0")),
        task_limit=int(os.environ.get("TABARENA_TASK_LIMIT", "1")),
        allow_openml_suite=True,
    )

    print(
        "Running json2vec TabArena-Lite "
        f"with tasks={tasks}, max_epochs={config['max_epochs']}, "
        f"d_model={config['d_model']}, batch_size={config['batch_size']}, "
        f"lr={config['lr']}, attention={config['attention']}, "
        f"pretrain_epochs={config['pretrain_epochs']}, "
        f"pretrain_p_mask={config['pretrain_p_mask']}",
        flush=True,
    )
    results = run_experiments_new(
        output_dir=str(OUT_DIR),
        model_experiments=experiments_from_config(config),
        tasks=tasks,
        repetitions_mode="TabArena-Lite",
        cache_mode=os.environ.get("TABARENA_CACHE_MODE", "default"),
        raise_on_failure=True,
        debug_mode=True,
    )
    print_results(results)


if __name__ == "__main__":
    main()
