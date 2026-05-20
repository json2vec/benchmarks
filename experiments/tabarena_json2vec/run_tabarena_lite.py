from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSON2VEC_REPO = Path(os.environ.get("JSON2VEC_REPO", ROOT.parent / "json2vec"))
if (JSON2VEC_REPO / "src").exists():
    sys.path.insert(0, str(JSON2VEC_REPO / "src"))
elif (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/tmp/tabarena/tabarena")

import openml
import pandas as pd

from json2vec_ag_model import JSON2VecTabArenaModel
from tabarena.benchmark.experiment import run_experiments_new
from tabarena.utils.config_utils import ConfigGenerator, generate_holdout_experiments


OUT_DIR = ROOT / "experiments" / "tabarena_json2vec" / "tabarena_out"
METADATA_PATH = Path("/tmp/tabarena/tabarena/tabarena/nips2025_utils/metadata/task_metadata_tabarena51.csv")


def main() -> None:
    task_limit = int(os.environ.get("TABARENA_TASK_LIMIT", "1"))
    max_instances = int(os.environ.get("TABARENA_MAX_INSTANCES", "0"))
    task_ids_env = os.environ.get("TABARENA_TASK_IDS")
    max_epochs = int(os.environ.get("JSON2VEC_MAX_EPOCHS", "2"))
    d_model = int(os.environ.get("JSON2VEC_D_MODEL", "16"))
    batch_size = int(os.environ.get("JSON2VEC_BATCH_SIZE", "128"))
    lr = float(os.environ.get("JSON2VEC_LR", "0.001"))
    weight_decay = float(os.environ.get("JSON2VEC_WEIGHT_DECAY", "0.0"))
    pretrain_epochs = int(os.environ.get("JSON2VEC_PRETRAIN_EPOCHS", "0"))
    pretrain_p_mask = float(os.environ.get("JSON2VEC_PRETRAIN_P_MASK", "0.0"))
    pretrain_lr = float(os.environ.get("JSON2VEC_PRETRAIN_LR", str(lr)))
    attention = os.environ.get("JSON2VEC_ATTENTION", "none")
    dropout = float(os.environ.get("JSON2VEC_DROPOUT", "0.0"))
    p_mask = float(os.environ.get("JSON2VEC_P_MASK", "0.0"))
    p_target = float(os.environ.get("JSON2VEC_P_TARGET", "0.0"))
    n_layers = int(os.environ.get("JSON2VEC_N_LAYERS", "1"))
    n_linear = int(os.environ.get("JSON2VEC_N_LINEAR", "1"))
    n_heads = int(os.environ.get("JSON2VEC_N_HEADS", "4"))
    random_seed = int(os.environ.get("JSON2VEC_RANDOM_SEED", "0"))
    num_bag_folds = int(os.environ.get("JSON2VEC_NUM_BAG_FOLDS", "2"))
    use_holdout = os.environ.get("JSON2VEC_HOLDOUT", "0").lower() in {"1", "true", "yes"}
    cache_mode = os.environ.get("TABARENA_CACHE_MODE", "default")

    config = {
        "d_model": d_model,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "pretrain_epochs": pretrain_epochs,
        "pretrain_p_mask": pretrain_p_mask,
        "pretrain_lr": pretrain_lr,
        "max_cat_vocab_size": 512,
        "accelerator": "mps",
        "attention": attention,
        "dropout": dropout,
        "p_mask": p_mask,
        "p_target": p_target,
        "n_layers": n_layers,
        "n_linear": n_linear,
        "n_heads": n_heads,
        "random_seed": random_seed,
    }

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
        task_ids = openml.study.get_suite("tabarena-v0.1").tasks[:task_limit]

    generator = ConfigGenerator(
        model_cls=JSON2VecTabArenaModel,
        manual_configs=[config],
        search_space={},
    )
    if use_holdout:
        experiments = generate_holdout_experiments(
            model_cls=JSON2VecTabArenaModel,
            configs=[config],
            name_id_prefix="c",
            time_limit=None,
        )
    else:
        experiments = generator.generate_all_bag_experiments(
            num_random_configs=0,
            num_bag_folds=num_bag_folds,
            fold_fitting_strategy="sequential_local",
            time_limit=None,
        )

    print(
        "Running json2vec TabArena-Lite smoke run "
        f"with tasks={task_ids}, max_epochs={max_epochs}, d_model={d_model}, "
        f"batch_size={batch_size}, lr={lr}, attention={attention}, "
        f"pretrain_epochs={pretrain_epochs}, pretrain_p_mask={pretrain_p_mask}, "
        f"num_bag_folds={num_bag_folds}, holdout={use_holdout}",
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
