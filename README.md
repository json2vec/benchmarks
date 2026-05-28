# json2vec Benchmarks

This repository contains benchmark integrations and result-export tooling for json2vec.

## Layout

- `scripts/benchmark.py`: unified CLI dispatcher for TabArena and RelBench runs.
- `benchmark_common/`: shared CLI, environment parsing, and import-path helpers.
- `experiments/tabarena_json2vec/`: TabArena-Lite benchmark for the json2vec AutoGluon adapter.
- `experiments/relbench_json2vec/`: RelBench entity/autocomplete benchmark for nested relational records.
- `experiments/tabarena_json2vec/tabarena_json2vec/`: shared Python package for the TabArena adapter, typed config, runner logic, paths, and utility code.
- `experiments/tabarena_json2vec/scripts/`: runnable entry points for json2vec, RandomForest, and smoke checks.
- `experiments/tabarena_json2vec/results.csv`: benchmark result rows.

## Unified CLI

Run commands from this benchmark repo root. CLI flags override matching
environment variables; omitted flags keep each suite's existing env/default
behavior.

```bash
python scripts/benchmark.py tabarena run --model json2vec
python scripts/benchmark.py tabarena run --model random-forest
python scripts/benchmark.py relbench run --model json2vec
python scripts/benchmark.py relbench run --model baselines
python scripts/benchmark.py relbench inspect
python scripts/benchmark.py relbench report --results <csv>
```

The legacy experiment-local scripts are still supported for compatibility.

Generated model artifacts and TabArena caches are ignored by git.

Each benchmark suite should keep one thin `scripts/` layer and move reusable
configuration, runner, adapter, and result logic into an importable package
under that experiment directory. That keeps benchmark rules and model code
auditable as more suites are added.
