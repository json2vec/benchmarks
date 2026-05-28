from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

from relbench_json2vec.config import RelBenchRunConfig
from relbench_json2vec.runner import run_baselines_from_env


def main() -> None:
    run_config = RelBenchRunConfig.from_env(default_results_scope="relbench_simple_baselines")
    run_baselines_from_env(run_config=run_config)


if __name__ == "__main__":
    main()
