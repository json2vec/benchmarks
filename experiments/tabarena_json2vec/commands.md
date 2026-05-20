# Reproduction Commands

Run commands from the repo root:

```bash
cd /Users/teo/Desktop/json2vec/benchmarks
```

Use the existing sibling json2vec virtualenv:

```bash
../.venv/bin/python
```

## Local Adapter Smoke Check

```bash
../.venv/bin/python experiments/tabarena_json2vec/scripts/smoke_check.py
```

This trains a tiny binary classifier twice: once supervised-only and once with masked feature pretraining.

## Best Focused json2vec Run

```bash
TABARENA_CACHE_MODE=ignore \
TABARENA_TASK_IDS=363621 \
JSON2VEC_D_MODEL=8 \
JSON2VEC_BATCH_SIZE=16 \
JSON2VEC_LR=0.022 \
JSON2VEC_MAX_EPOCHS=5 \
JSON2VEC_ATTENTION=none \
JSON2VEC_NUM_BAG_FOLDS=2 \
JSON2VEC_RANDOM_SEED=0 \
../.venv/bin/python experiments/tabarena_json2vec/scripts/run_json2vec.py
```

Expected result from local run:

```text
metric_error = 0.233947
```

## Best Masked-Pretraining Run

```bash
TABARENA_CACHE_MODE=ignore \
TABARENA_TASK_IDS=363621 \
JSON2VEC_D_MODEL=8 \
JSON2VEC_BATCH_SIZE=16 \
JSON2VEC_LR=0.022 \
JSON2VEC_MAX_EPOCHS=5 \
JSON2VEC_ATTENTION=none \
JSON2VEC_NUM_BAG_FOLDS=2 \
JSON2VEC_RANDOM_SEED=0 \
JSON2VEC_PRETRAIN_EPOCHS=2 \
JSON2VEC_PRETRAIN_P_MASK=0.20 \
JSON2VEC_PRETRAIN_LR=0.01 \
../.venv/bin/python experiments/tabarena_json2vec/scripts/run_json2vec.py
```

Expected result from local run:

```text
metric_error = 0.259035
```

## RandomForest Baseline

```bash
TABARENA_CACHE_MODE=ignore \
TABARENA_TASK_IDS=363621 \
RF_NUM_BAG_FOLDS=2 \
RF_NUM_RANDOM_CONFIGS=0 \
../.venv/bin/python experiments/tabarena_json2vec/scripts/run_random_forest.py
```

Expected result from local run:

```text
metric_error = 0.314211
```

## 11 Small-Dataset json2vec Run

```bash
TABARENA_CACHE_MODE=ignore \
TABARENA_MAX_INSTANCES=1500 \
TABARENA_TASK_LIMIT=0 \
JSON2VEC_D_MODEL=64 \
JSON2VEC_BATCH_SIZE=32 \
JSON2VEC_LR=0.0003 \
JSON2VEC_ATTENTION=none \
JSON2VEC_MAX_EPOCHS=1 \
JSON2VEC_NUM_BAG_FOLDS=2 \
JSON2VEC_RANDOM_SEED=0 \
../.venv/bin/python experiments/tabarena_json2vec/scripts/run_json2vec.py
```

## 11 Small-Dataset RF Baseline

```bash
TABARENA_CACHE_MODE=ignore \
TABARENA_MAX_INSTANCES=1500 \
TABARENA_TASK_LIMIT=0 \
RF_NUM_BAG_FOLDS=2 \
RF_NUM_RANDOM_CONFIGS=0 \
../.venv/bin/python experiments/tabarena_json2vec/scripts/run_random_forest.py
```
