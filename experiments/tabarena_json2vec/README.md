# json2vec TabArena Experiment

This folder contains the local TabArena integration used to test `json2vec` as an AutoGluon-compatible model.

## What Is Included

- `tabarena_json2vec/ag_model.py`: AutoGluon `AbstractModel` adapter for json2vec.
- `tabarena_json2vec/config.py`: typed environment-backed run configuration.
- `tabarena_json2vec/runner.py`: shared TabArena experiment construction and execution.
- `tabarena_json2vec/paths.py`: shared repository, json2vec, and TabArena path setup.
- `tabarena_json2vec/tabarena.py`: shared TabArena task selection and result printing helpers.
- `scripts/run_json2vec.py`: TabArena-Lite runner for json2vec.
- `scripts/run_random_forest.py`: TabArena RandomForest baseline runner using TabArena's built-in RF generator.
- `scripts/smoke_check.py`: tiny local adapter check covering supervised training and masked pretraining.
- `results_summary.csv`: key benchmark results from the local MacBook runs.
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
```

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

## Main Finding

The first broad pass across 11 small TabArena datasets showed that the current json2vec tabular adapter is not competitive with a simple RandomForest baseline overall.

However, after focusing on one benchmark, `blood-transfusion-service-center` (`tid=363621`), json2vec beat the TabArena RandomForest baseline on the same TabArena-Lite split.

Best result on `blood-transfusion-service-center`:

```text
json2vec metric_error      = 0.233947
TabArena RF metric_error   = 0.314211
```

For TabArena `metric_error`, lower is better. For `roc_auc`, this is effectively `1 - AUC`.

The TabArena RF number is from `RandomForest_c1_BAG_L1`, using TabArena's built-in `gen_randomforest` manual config with 2 bag folds and no random RF configs.

Best config:

```bash
TABARENA_TASK_IDS=363621
JSON2VEC_D_MODEL=8
JSON2VEC_BATCH_SIZE=16
JSON2VEC_LR=0.022
JSON2VEC_MAX_EPOCHS=5
JSON2VEC_ATTENTION=none
JSON2VEC_NUM_BAG_FOLDS=2
JSON2VEC_RANDOM_SEED=0
```

## Fixed Mini-Suite Result

A three-task TabArena-Lite sanity suite was run with one fixed json2vec config
and no per-task tuning:

```bash
TABARENA_TASK_IDS=363621,363629,363625
JSON2VEC_D_MODEL=16
JSON2VEC_BATCH_SIZE=32
JSON2VEC_LR=0.001
JSON2VEC_MAX_EPOCHS=1
JSON2VEC_ATTENTION=none
JSON2VEC_NUM_BAG_FOLDS=2
JSON2VEC_RANDOM_SEED=0
```

Results:

| task | metric | json2vec metric_error | RF metric_error |
| --- | --- | ---: | ---: |
| blood-transfusion-service-center | roc_auc | 0.418509 | 0.314211 |
| diabetes | roc_auc | 0.380656 | 0.165228 |
| concrete_compressive_strength | rmse | 16.986025 | 6.084826 |

This mini-suite validates that the adapter runs through TabArena's binary
classification and regression paths, but it also confirms that the current
fixed json2vec tabular adapter is not competitive with the RandomForest
baseline without task-specific tuning.

## Pretraining Result

Masked feature reconstruction pretraining was implemented and tested. It runs end-to-end, but on the tuned `blood-transfusion` benchmark it did not improve the supervised-only result.

Best masked-pretraining run:

```text
pretrain_epochs = 2
pretrain_p_mask = 0.20
pretrain_lr = 0.01
metric_error = 0.259035
```

Supervised-only remains better:

```text
supervised-only best = 0.233947
best pretraining     = 0.259035
```

## Interpretation

The adapter works and MPS acceleration is usable on a MacBook, but broad benchmark performance is currently limited by the adapter structure.

Most likely next improvements are structural:

- Encode columns as feature tokens rather than a single row JSON array with pooled fields.
- Add a classification-specific target/probability head.
- Improve inference batching and model reload overhead.
- Pretrain on more rows/tasks, not only one small fold-local dataset.
- Revisit feature normalization and categorical handling.
