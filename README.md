# json2vec Benchmarks

This repository contains benchmark integrations and local result summaries for json2vec.

## Layout

- `experiments/tabarena_json2vec/`: TabArena-Lite benchmark for the json2vec AutoGluon adapter.
- `experiments/tabarena_json2vec/tabarena_json2vec/`: shared Python package for the TabArena adapter, typed config, runner logic, paths, and utility code.
- `experiments/tabarena_json2vec/scripts/`: runnable entry points for json2vec, RandomForest, and smoke checks.
- `experiments/tabarena_json2vec/results_summary.csv`: recorded local results.

Generated model artifacts and TabArena caches are ignored by git.

Each benchmark suite should keep one thin `scripts/` layer and move reusable
configuration, runner, adapter, and result logic into an importable package
under that experiment directory. That keeps benchmark rules and model code
auditable as more suites are added.
