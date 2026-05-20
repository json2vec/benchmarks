# json2vec TabArena Experiment

This folder contains the local TabArena integration used to test `json2vec` as an AutoGluon-compatible model.

## What Is Included

- `json2vec_ag_model.py`: AutoGluon `AbstractModel` adapter for json2vec.
- `run_tabarena_lite.py`: TabArena-Lite runner for json2vec.
- `run_tabarena_rf_baseline.py`: TabArena RandomForest baseline runner using TabArena's built-in RF generator.
- `smoke_check.py`: tiny local adapter check covering supervised training and masked pretraining.
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
../.venv/bin/python
```

instead of `uv run` after dependency resolution, because repeated `uv run` calls attempted to resync dependencies and could reintroduce incompatible packages.

The runners import json2vec from `JSON2VEC_REPO/src` when `JSON2VEC_REPO` is set. If it is not set, they look for a sibling `../json2vec/src` checkout and then fall back to the installed package.

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
