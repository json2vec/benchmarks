from __future__ import annotations

import os
import sys
from pathlib import Path


BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = BENCHMARKS_ROOT / "experiments"
REL_BENCH_EXPERIMENT_DIR = EXPERIMENTS_DIR / "relbench_json2vec"
TABARENA_EXPERIMENT_DIR = EXPERIMENTS_DIR / "tabarena_json2vec"


def add_sys_path(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def add_experiment_path(experiment_dir: Path) -> None:
    add_sys_path(experiment_dir)


def add_json2vec_path() -> None:
    json2vec_repo = Path(os.environ.get("JSON2VEC_REPO", BENCHMARKS_ROOT.parent / "json2vec"))
    for src_path in (json2vec_repo / "src", BENCHMARKS_ROOT / "src"):
        if src_path.exists():
            add_sys_path(src_path)
            return


def configure_tabarena_paths() -> None:
    add_experiment_path(TABARENA_EXPERIMENT_DIR)
    add_json2vec_path()
    tabarena_root = Path(os.environ.get("TABARENA_ROOT", "/tmp/tabarena"))
    add_sys_path(tabarena_root / "tabarena")


def configure_relbench_paths() -> None:
    add_experiment_path(REL_BENCH_EXPERIMENT_DIR)
    add_json2vec_path()
