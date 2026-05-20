from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from tabarena_json2vec.paths import configure_import_paths

configure_import_paths()

from tabarena_json2vec import JSON2VecTabArenaModel


def run_case(name: str, extra_params: dict) -> None:
    X = pd.DataFrame(
        {
            "num": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["a", "a", "b", "b", "a", "b"],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1])

    model = JSON2VecTabArenaModel(
        problem_type="binary",
        hyperparameters={
            "d_model": 8,
            "batch_size": 2,
            "max_epochs": 1,
            "lr": 0.01,
            "accelerator": "cpu",
            "attention": "none",
            "random_seed": 0,
            **extra_params,
        },
    )
    model.fit(X=X, y=y)

    proba = model.predict_proba(X)
    pred = model.predict(X)
    print(
        {
            "case": name,
            "proba_shape": tuple(proba.shape),
            "pred_len": len(pred),
            "first_probabilities": proba[:3].round(4).tolist(),
        }
    )


def main() -> None:
    run_case("supervised", {})
    run_case(
        "masked_pretraining",
        {
            "pretrain_epochs": 1,
            "pretrain_p_mask": 0.8,
            "pretrain_lr": 0.01,
        },
    )


if __name__ == "__main__":
    main()
