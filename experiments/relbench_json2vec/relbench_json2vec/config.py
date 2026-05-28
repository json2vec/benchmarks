from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from relbench_json2vec.paths import EXPERIMENT_DIR


DEFAULT_TASKS = (
    "rel-f1:driver-dnf",
    "rel-f1:driver-top3",
    "rel-f1:driver-position",
    "rel-f1:results-position",
)
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


def parse_task_specs(raw: str | None) -> tuple[tuple[str, str], ...]:
    specs = DEFAULT_TASKS if raw is None or raw.strip() == "" else tuple(part.strip() for part in raw.split(","))
    out: list[tuple[str, str]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"RelBench task spec must be dataset:task, got {spec!r}")
        dataset, task = spec.split(":", 1)
        if not dataset.strip() or not task.strip():
            raise ValueError(f"RelBench task spec must be dataset:task, got {spec!r}")
        out.append((dataset.strip(), task.strip()))
    return tuple(out)


@dataclass(frozen=True)
class RelBenchRunConfig:
    tasks: tuple[tuple[str, str], ...]
    download: bool
    cache_dir: Path
    results_csv: Path | None
    results_scope: str
    results_notes: str

    @classmethod
    def from_env(cls, *, default_results_scope: str, env: Mapping[str, str] | None = None) -> "RelBenchRunConfig":
        values = os.environ if env is None else env
        results_csv = values.get("RELBENCH_RESULTS_CSV")
        if results_csv is None:
            results_csv_path = EXPERIMENT_DIR / "results.csv"
        elif results_csv.strip() == "":
            results_csv_path = None
        else:
            results_csv_path = Path(results_csv)

        return cls(
            tasks=parse_task_specs(values.get("RELBENCH_TASKS")),
            download=_bool_env("RELBENCH_DOWNLOAD", False, values),
            cache_dir=Path(_str_env("RELBENCH_CACHE_DIR", str(EXPERIMENT_DIR / "relbench_cache"), values)),
            results_csv=results_csv_path,
            results_scope=_str_env("RELBENCH_RESULTS_SCOPE", default_results_scope, values),
            results_notes=_str_env("RELBENCH_RESULTS_NOTES", "", values),
        )


@dataclass(frozen=True)
class JSON2VecConfig:
    d_model: int
    batch_size: int
    max_epochs: int
    lr: float
    weight_decay: float
    accelerator: str
    attention: str
    n_layers: int
    n_linear: int
    n_heads: int
    random_seed: int
    history_limit: int
    relation_depth: int
    max_cat_vocab_size: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "JSON2VecConfig":
        values = os.environ if env is None else env
        return cls(
            d_model=_int_env("JSON2VEC_D_MODEL", 32, values),
            batch_size=_int_env("JSON2VEC_BATCH_SIZE", 64, values),
            max_epochs=_int_env("JSON2VEC_MAX_EPOCHS", 3, values),
            lr=_float_env("JSON2VEC_LR", 0.001, values),
            weight_decay=_float_env("JSON2VEC_WEIGHT_DECAY", 0.0, values),
            accelerator=_str_env("JSON2VEC_ACCELERATOR", "mps", values),
            attention=_str_env("JSON2VEC_ATTENTION", "mha", values),
            n_layers=_int_env("JSON2VEC_N_LAYERS", 1, values),
            n_linear=_int_env("JSON2VEC_N_LINEAR", 1, values),
            n_heads=_int_env("JSON2VEC_N_HEADS", 4, values),
            random_seed=_int_env("JSON2VEC_RANDOM_SEED", 0, values),
            history_limit=_int_env("JSON2VEC_HISTORY_LIMIT", 128, values),
            relation_depth=_int_env("JSON2VEC_RELATION_DEPTH", 1, values),
            max_cat_vocab_size=_int_env("JSON2VEC_MAX_CAT_VOCAB_SIZE", 2048, values),
        )

    def label(self) -> str:
        return f"json2vec_d{self.d_model}_b{self.batch_size}_lr{self.lr:g}_e{self.max_epochs}_h{self.history_limit}"
