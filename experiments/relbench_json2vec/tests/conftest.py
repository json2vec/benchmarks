from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

from relbench_json2vec.paths import configure_import_paths

configure_import_paths()
