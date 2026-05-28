# json2vec RelBench Experiment

This folder contains the RelBench v2 integration for `json2vec`.

The first benchmark is a fixed-config clean matrix for entity and autocomplete
tasks. It is not a leaderboard submission and does not include per-task tuning,
link prediction, recommendation, GNN baselines, or hyperparameter search.

## What Is Included

- `relbench_json2vec/materialize.py`: converts each RelBench task row into a nested JSON observation.
- `relbench_json2vec/schema.py`: infers a json2vec schema from materialized records.
- `relbench_json2vec/model.py`: trains json2vec and converts predictions to RelBench-compatible arrays.
- `relbench_json2vec/baselines.py`: simple global/entity-history baselines.
- `relbench_json2vec/runner.py`: shared task loading, execution, printing, and CSV export.
- `relbench_json2vec/sdk.py`: manifest-driven run orchestration and report generation.
- `scripts/benchmark.py`: stable CLI for `run`, `report`, and `inspect`.
- `scripts/setup_env.sh`: isolated environment setup for RelBench runs.
- `scripts/run_json2vec.py`: json2vec runner.
- `scripts/run_baselines.py`: simple baseline runner.
- `scripts/smoke_check.py`: local fake-task smoke check.
- `results.csv`: normalized benchmark rows.

## Default Clean Matrix

```text
rel-f1:driver-dnf
rel-f1:driver-top3
rel-f1:driver-position
rel-f1:results-position
```

Only RelBench entity/autocomplete tasks with binary classification or regression
targets are supported in v1. Unsupported task types fail fast.

## Environment Controls

```text
RELBENCH_TASKS=rel-f1:driver-dnf,rel-f1:driver-top3,rel-f1:driver-position,rel-f1:results-position
RELBENCH_DOWNLOAD=true
RELBENCH_CACHE_DIR=experiments/relbench_json2vec/relbench_cache
RELBENCH_RESULTS_CSV=experiments/relbench_json2vec/results.csv
RELBENCH_RESULTS_SCOPE=relbench_json2vec_clean_matrix
RELBENCH_RESULTS_NOTES=free-form notes

JSON2VEC_D_MODEL=32
JSON2VEC_BATCH_SIZE=64
JSON2VEC_MAX_EPOCHS=3
JSON2VEC_LR=0.001
JSON2VEC_HISTORY_LIMIT=128
JSON2VEC_RELATION_DEPTH=1
```

`RELBENCH_RESULTS_CSV` defaults to this experiment's `results.csv`. Set it to
an empty string to disable CSV export.

Run-local SDK outputs are written under ignored timestamped directories in
`runs/`. The tracked `results.csv` is appended only when `benchmark.py run` is
called with `--append-results`.

## Result Schema

```text
scope,dataset,task,task_type,model_or_config,split,metric,value,time_train_s,time_infer_s,notes
```

Exploratory tuning should be written to a separate `search_results.csv` or a
separate scope that is clearly marked as search output.
