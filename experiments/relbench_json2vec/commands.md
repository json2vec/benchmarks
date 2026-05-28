# Reproduction Commands

Run commands from the benchmark repo root:

```bash
cd /Users/teo/Developer/json2vec/benchmarks
```

Use the sibling json2vec virtualenv:

```bash
../json2vec/.venv/bin/python
```

Install RelBench into that environment if needed:

```bash
../json2vec/.venv/bin/python -m pip install "relbench==2.1.2"
```

For benchmark runs, prefer the isolated environment:

```bash
experiments/relbench_json2vec/scripts/setup_env.sh
```

## SDK Dry Run

```bash
experiments/relbench_json2vec/.venv/bin/python \
  experiments/relbench_json2vec/scripts/benchmark.py run --dry-run
```

## Single-Task SDK Run

```bash
experiments/relbench_json2vec/.venv/bin/python \
  experiments/relbench_json2vec/scripts/benchmark.py run \
  --tasks rel-f1:driver-dnf
```

## SDK Report

```bash
experiments/relbench_json2vec/.venv/bin/python \
  experiments/relbench_json2vec/scripts/benchmark.py report \
  --results experiments/relbench_json2vec/runs/<run-id>/results.csv
```

## Smoke Check

```bash
../json2vec/.venv/bin/python experiments/relbench_json2vec/scripts/smoke_check.py
```

## Clean Matrix json2vec Run

```bash
RELBENCH_DOWNLOAD=true \
RELBENCH_RESULTS_SCOPE=relbench_json2vec_clean_matrix \
RELBENCH_RESULTS_NOTES="clean fixed-config matrix; no tuning" \
JSON2VEC_D_MODEL=32 \
JSON2VEC_BATCH_SIZE=64 \
JSON2VEC_MAX_EPOCHS=3 \
JSON2VEC_LR=0.001 \
JSON2VEC_HISTORY_LIMIT=128 \
JSON2VEC_RELATION_DEPTH=1 \
../json2vec/.venv/bin/python experiments/relbench_json2vec/scripts/run_json2vec.py
```

## Clean Matrix Baselines

```bash
RELBENCH_DOWNLOAD=true \
RELBENCH_RESULTS_SCOPE=relbench_simple_baselines \
RELBENCH_RESULTS_NOTES="simple global/entity-history baselines" \
../json2vec/.venv/bin/python experiments/relbench_json2vec/scripts/run_baselines.py
```
