from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tabarena_json2vec.paths import EXPERIMENT_DIR


TRUE_VALUES = {"1", "true", "yes", "on"}


def _bool_env(name: str, default: bool = False, env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return values.get(name, str(int(default))).strip().lower() in TRUE_VALUES


def _int_env(name: str, default: int, env: Mapping[str, str] | None = None) -> int:
    values = os.environ if env is None else env
    return int(values.get(name, str(default)))


def _float_env(name: str, default: float, env: Mapping[str, str] | None = None) -> float:
    values = os.environ if env is None else env
    return float(values.get(name, str(default)))


def _str_env(name: str, default: str, env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return values.get(name, default)


@dataclass(frozen=True)
class TabArenaRunConfig:
    task_ids: str | None
    max_instances: int
    task_limit: int
    cache_mode: str
    repetitions_mode: str
    output_dir: Path
    debug_mode: bool
    raise_on_failure: bool

    @classmethod
    def from_env(
        cls,
        *,
        default_output_dir: Path,
        default_task_limit: int = 1,
        env: Mapping[str, str] | None = None,
    ) -> "TabArenaRunConfig":
        values = os.environ if env is None else env
        return cls(
            task_ids=values.get("TABARENA_TASK_IDS"),
            max_instances=_int_env("TABARENA_MAX_INSTANCES", 0, values),
            task_limit=_int_env("TABARENA_TASK_LIMIT", default_task_limit, values),
            cache_mode=_str_env("TABARENA_CACHE_MODE", "default", values),
            repetitions_mode=_str_env("TABARENA_REPETITIONS_MODE", "TabArena-Lite", values),
            output_dir=Path(values.get("TABARENA_OUTPUT_DIR", str(default_output_dir))),
            debug_mode=_bool_env("TABARENA_DEBUG_MODE", True, values),
            raise_on_failure=_bool_env("TABARENA_RAISE_ON_FAILURE", True, values),
        )


@dataclass(frozen=True)
class JSON2VecConfig:
    d_model: int
    batch_size: int
    max_epochs: int
    lr: float
    weight_decay: float
    pretrain_epochs: int
    pretrain_p_mask: float
    pretrain_lr: float
    max_cat_vocab_size: int
    accelerator: str
    attention: str
    dropout: float
    p_mask: float
    p_target: float
    n_layers: int
    n_linear: int
    n_heads: int
    random_seed: int
    num_bag_folds: int
    holdout: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "JSON2VecConfig":
        values = os.environ if env is None else env
        lr = _float_env("JSON2VEC_LR", 0.001, values)
        return cls(
            d_model=_int_env("JSON2VEC_D_MODEL", 16, values),
            batch_size=_int_env("JSON2VEC_BATCH_SIZE", 128, values),
            max_epochs=_int_env("JSON2VEC_MAX_EPOCHS", 2, values),
            lr=lr,
            weight_decay=_float_env("JSON2VEC_WEIGHT_DECAY", 0.0, values),
            pretrain_epochs=_int_env("JSON2VEC_PRETRAIN_EPOCHS", 0, values),
            pretrain_p_mask=_float_env("JSON2VEC_PRETRAIN_P_MASK", 0.0, values),
            pretrain_lr=_float_env("JSON2VEC_PRETRAIN_LR", lr, values),
            max_cat_vocab_size=_int_env("JSON2VEC_MAX_CAT_VOCAB_SIZE", 512, values),
            accelerator=_str_env("JSON2VEC_ACCELERATOR", "mps", values),
            attention=_str_env("JSON2VEC_ATTENTION", "none", values),
            dropout=_float_env("JSON2VEC_DROPOUT", 0.0, values),
            p_mask=_float_env("JSON2VEC_P_MASK", 0.0, values),
            p_target=_float_env("JSON2VEC_P_TARGET", 0.0, values),
            n_layers=_int_env("JSON2VEC_N_LAYERS", 1, values),
            n_linear=_int_env("JSON2VEC_N_LINEAR", 1, values),
            n_heads=_int_env("JSON2VEC_N_HEADS", 4, values),
            random_seed=_int_env("JSON2VEC_RANDOM_SEED", 0, values),
            num_bag_folds=_int_env("JSON2VEC_NUM_BAG_FOLDS", 2, values),
            holdout=_bool_env("JSON2VEC_HOLDOUT", False, values),
        )

    def autogluon_params(self) -> dict:
        return {
            "d_model": self.d_model,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "pretrain_epochs": self.pretrain_epochs,
            "pretrain_p_mask": self.pretrain_p_mask,
            "pretrain_lr": self.pretrain_lr,
            "max_cat_vocab_size": self.max_cat_vocab_size,
            "accelerator": self.accelerator,
            "attention": self.attention,
            "dropout": self.dropout,
            "p_mask": self.p_mask,
            "p_target": self.p_target,
            "n_layers": self.n_layers,
            "n_linear": self.n_linear,
            "n_heads": self.n_heads,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class RandomForestConfig:
    num_bag_folds: int
    num_random_configs: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RandomForestConfig":
        values = os.environ if env is None else env
        return cls(
            num_bag_folds=_int_env("RF_NUM_BAG_FOLDS", 2, values),
            num_random_configs=_int_env("RF_NUM_RANDOM_CONFIGS", 0, values),
        )


def default_tabarena_output_dir(name: str) -> Path:
    return EXPERIMENT_DIR / name
