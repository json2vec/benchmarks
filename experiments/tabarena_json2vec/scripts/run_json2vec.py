from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from tabarena_json2vec.config import JSON2VecConfig, TabArenaRunConfig, default_tabarena_output_dir
from tabarena_json2vec.paths import configure_import_paths

configure_import_paths()

from tabarena_json2vec.runner import run_json2vec_from_env


def main() -> None:
    run_config = TabArenaRunConfig.from_env(
        default_output_dir=default_tabarena_output_dir("tabarena_out"),
        default_results_scope="json2vec",
    )
    model_config = JSON2VecConfig.from_env()
    run_json2vec_from_env(run_config=run_config, model_config=model_config)


if __name__ == "__main__":
    main()
