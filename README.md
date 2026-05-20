# json2vec Benchmarks

This repository contains benchmark integrations and local result summaries for json2vec.

## Layout

- `experiments/tabarena_json2vec/`: TabArena-Lite benchmark for the json2vec AutoGluon adapter.
- `experiments/tabarena_json2vec/tabarena_json2vec/`: shared Python package for the TabArena adapter, paths, and utility code.
- `experiments/tabarena_json2vec/scripts/`: runnable entry points for json2vec, RandomForest, and smoke checks.
- `experiments/tabarena_json2vec/results_summary.csv`: recorded local results.

Generated model artifacts and TabArena caches are ignored by git.
