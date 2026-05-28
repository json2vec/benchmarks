from __future__ import annotations

import os
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]


def configure_import_paths() -> None:
    json2vec_repo = Path(os.environ.get("JSON2VEC_REPO", REPO_ROOT.parent / "json2vec"))
    candidate_src_paths = [json2vec_repo / "src", REPO_ROOT / "src"]
    for src_path in candidate_src_paths:
        if src_path.exists():
            sys.path.insert(0, str(src_path))
            break
