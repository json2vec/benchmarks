from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from relbench_json2vec.config import JSON2VecConfig, RelBenchRunConfig
from relbench_json2vec.runner import run_json2vec_from_env


def main() -> None:
    run_config = RelBenchRunConfig.from_env(default_results_scope="relbench_json2vec_clean_matrix")
    model_config = JSON2VecConfig.from_env()
    run_json2vec_from_env(run_config=run_config, model_config=model_config)


if __name__ == "__main__":
    main()
