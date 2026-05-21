# json2vec TabArena Experiment

This folder contains the local TabArena integration used to test `json2vec` as an AutoGluon-compatible model.

## What Is Included

- `tabarena_json2vec/ag_model.py`: AutoGluon `AbstractModel` adapter for json2vec.
- `tabarena_json2vec/config.py`: typed environment-backed run configuration.
- `tabarena_json2vec/runner.py`: shared TabArena experiment construction, execution, and CSV result export.
- `tabarena_json2vec/results.py`: normalized CSV writer for TabArena result payloads.
- `tabarena_json2vec/paths.py`: shared repository, json2vec, and TabArena path setup.
- `tabarena_json2vec/tabarena.py`: shared TabArena task selection and result printing helpers.
- `scripts/run_json2vec.py`: TabArena-Lite runner for json2vec.
- `scripts/run_random_forest.py`: TabArena RandomForest baseline runner using TabArena's built-in RF generator.
- `scripts/smoke_check.py`: tiny local adapter check covering supervised training and masked pretraining.
- `results_summary.csv`: curated benchmark result snapshots.
- `commands.md`: exact commands/configs for reproducing the main runs.
- `best_blood_config.env`: env-file form of the best focused benchmark config.

Generated outputs are intentionally ignored by git:

- `AutogluonModels/`
- `experiments/tabarena_json2vec/tabarena_out/`
- `experiments/tabarena_json2vec/tabarena_rf_out/`

## Environment Notes

The runs were performed on a MacBook with PyTorch MPS enabled. TabArena was cloned locally at:

```text
/tmp/tabarena
```

The local working Python environment is the sibling json2vec repo `.venv`. From the `benchmarks` repo, we used:

```bash
../json2vec/.venv/bin/python
```

instead of `uv run` after dependency resolution, because repeated `uv run` calls attempted to resync dependencies and could reintroduce incompatible packages.

The runners import json2vec from `JSON2VEC_REPO/src` when `JSON2VEC_REPO` is set. If it is not set, they look for a sibling `../json2vec/src` checkout and then fall back to the installed package.

## Running Benchmarks Correctly

The TabArena runners intentionally route every benchmark through TabArena's
own experiment API:

- `run_experiments_new(...)` owns task loading, splits, metrics, and result
  payloads.
- `JSON2VecTabArenaModel` is an AutoGluon `AbstractModel`, so json2vec is
  evaluated inside the same AutoGluon model lifecycle as other TabArena models.
- Bagging is created by TabArena/AutoGluon config utilities, not custom split
  code in this repository.
- Test labels are never read by json2vec-specific code. The adapter only sees
  the `X` and `y` that AutoGluon passes into `_fit`, and it predicts through
  `_predict_proba`/`_predict`.
- Use `TABARENA_CACHE_MODE=ignore` for reproduction runs where you want a fresh
  fit. Use the default cache mode only when intentionally reusing existing
  TabArena outputs.

The scripts are thin entrypoints. Shared policy and configuration live in the
`tabarena_json2vec` package so additional TabArena configs, baselines, or future
benchmark suites can reuse the same structure instead of duplicating runner
logic.

Common environment controls:

```text
TABARENA_TASK_IDS=363621,363629
TABARENA_MAX_INSTANCES=1500
TABARENA_TASK_LIMIT=3
TABARENA_CACHE_MODE=ignore
TABARENA_REPETITIONS_MODE=TabArena-Lite
TABARENA_OUTPUT_DIR=experiments/tabarena_json2vec/tabarena_out
TABARENA_RESULTS_CSV=experiments/tabarena_json2vec/results.csv
TABARENA_RESULTS_SCOPE=experiment_name
TABARENA_RESULTS_NOTES=free-form notes for each emitted row
```

`TABARENA_RESULTS_CSV` defaults to `experiments/tabarena_json2vec/results.csv`.
Set it to an empty string to disable CSV export for a run.

json2vec-specific controls:

```text
JSON2VEC_D_MODEL=8
JSON2VEC_BATCH_SIZE=16
JSON2VEC_LR=0.022
JSON2VEC_MAX_EPOCHS=5
JSON2VEC_ATTENTION=none
JSON2VEC_NUM_BAG_FOLDS=2
JSON2VEC_RANDOM_SEED=0
JSON2VEC_PRETRAIN_EPOCHS=0
JSON2VEC_PRETRAIN_P_MASK=0.0
```

## Result Files

Run-level result rows are written to CSV with this schema:

```text
scope,tid,dataset,problem_type,model_or_config,metric,metric_error,metric_error_val,time_train_s,time_infer_s,notes
```

Use `results.csv` for raw appended runner output and `results_summary.csv` for
curated snapshots that should be preserved in git. README files should describe
protocols and commands, not embed benchmark statistics.

## Development Notes

Known areas for adapter improvement:

- Encode columns as feature tokens rather than a single row JSON array with pooled fields.
- Add a classification-specific target/probability head.
- Improve inference batching and model reload overhead.
- Pretrain on more rows/tasks rather than only fold-local datasets.
- Revisit feature normalization and categorical handling.
