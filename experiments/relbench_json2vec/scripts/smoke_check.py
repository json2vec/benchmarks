from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

from relbench_json2vec.config import JSON2VecConfig
from relbench_json2vec.model import fit_and_evaluate_json2vec
from relbench_json2vec.paths import configure_import_paths
from tests.fakes import make_binary_task, make_regression_task


def run_case(name: str, task) -> None:
    configure_import_paths()
    import json2vec as j2v

    result = fit_and_evaluate_json2vec(
        j2v=j2v,
        dataset_name="fake",
        task_name=name,
        task=task,
        config=JSON2VecConfig(
            d_model=8,
            batch_size=2,
            max_epochs=1,
            lr=0.01,
            weight_decay=0.0,
            accelerator="cpu",
            attention="none",
            n_layers=1,
            n_linear=1,
            n_heads=4,
            random_seed=0,
            history_limit=3,
            relation_depth=1,
            max_cat_vocab_size=32,
        ),
    )
    print(
        {
            "case": name,
            "task_type": result.task_type,
            "splits": sorted(result.metrics_by_split),
            "metrics": result.metrics_by_split,
        },
        flush=True,
    )


def main() -> None:
    run_case("binary", make_binary_task())
    run_case("regression", make_regression_task())


if __name__ == "__main__":
    main()
