#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARKS_DIR="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
JSON2VEC_REPO="${JSON2VEC_REPO:-${BENCHMARKS_DIR}/../json2vec}"
VENV_DIR="${RELBENCH_JSON2VEC_VENV:-${EXPERIMENT_DIR}/.venv}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${JSON2VEC_REPO}/.venv/bin/python" ]]; then
    PYTHON_BIN="${JSON2VEC_REPO}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit(f"json2vec requires Python >=3.12, got {sys.version.split()[0]}")
PY

"${PYTHON_BIN}" -m venv --clear "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e "${JSON2VEC_REPO}"
"${VENV_DIR}/bin/python" -m pip install "relbench==2.1.2"
"${VENV_DIR}/bin/python" -m pip uninstall -y torchvision || true

"${VENV_DIR}/bin/python" - <<'PY'
import importlib

for package in ("json2vec", "lightning", "relbench"):
    importlib.import_module(package)
    print(f"import ok: {package}")
PY

echo "RelBench json2vec environment ready: ${VENV_DIR}"
